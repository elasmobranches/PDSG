# hooks/iter_logger_hook.py
# Segmentation Train/Val Loss와 mIoU를 CSV 파일로 기록하는 Hook
#
# Ported verbatim from the research repository's file of the same name — the
# first two lines above are its own header. No body was modified on the way
# in. It writes the per-iteration learning_curve.csv every recorded run has
# beside its checkpoints, which is what makes a retrain's loss curve
# comparable to a recorded one step by step (docs/VERIFICATION.md).
#
# The Korean in this file is entirely comments and docstrings, carried over
# with the body. Every string it actually logs during a run is English; there
# is nothing here to translate that a user would ever see.

import os
import csv
import time
import torch
from mmengine.hooks import Hook
from mmseg.registry import HOOKS

def _sum_loss_dict(loss_dict):
    """loss dict에서 숫자(텐서)만 골라 합계 반환"""
    total = 0.0
    for v in loss_dict.values():
        try:
            total += float(v.item()) if hasattr(v, "item") else float(v)
        except Exception:
            continue
    return total

@HOOKS.register_module()
class IterLoggerHook(Hook):
    """
    Segmentation Train/Val Loss와 mIoU를 CSV 파일로 기록하는 Custom Hook

    이 Hook은:
    - Train iteration마다 train loss를 CSV에 기록
    - Val epoch마다 val mIoU를 CSV에 기록
    - 별도의 시각화 스크립트로 그래프 생성 가능

    Args:
        out_csv (str): CSV 파일 저장 경로 (None이면 자동으로 work_dir/learning_curve.csv)
        flush_secs (int): CSV flush 간격 (초)

    사용법:
        Config 파일에 다음과 같이 추가:
        ```python
        custom_imports = dict(
            imports=['hooks.iter_logger_hook'],
            allow_failed_imports=False
        )

        custom_hooks = [
            dict(
                type='IterLoggerHook',
                out_csv=None,  # 자동으로 work_dir/learning_curve.csv
                flush_secs=10
            )
        ]
        ```
    """

    def __init__(self, out_csv=None, flush_secs=10):
        self.out_csv = out_csv  # None이면 나중에 before_run에서 자동 설정
        self.flush_secs = flush_secs
        self._last_flush = time.time()
        self._csv_initialized = False

    def before_run(self, runner):
        """학습 시작 전 호출 - CSV 경로 자동 설정 및 초기화"""
        # out_csv가 None이면 work_dir 기반으로 자동 생성
        if self.out_csv is None:
            self.out_csv = os.path.join(runner.work_dir, 'learning_curve.csv')

        # CSV 파일 디렉토리 생성
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)

        # CSV 헤더 생성 (파일이 없으면)
        # Segmentation 전용: iter, train_loss, val_mIoU
        if not os.path.exists(self.out_csv):
            with open(self.out_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iter', 'train_loss', 'val_mIoU'])

        self._csv_initialized = True
        runner.logger.info(f'IterLoggerHook: Saving learning curve to {self.out_csv}')

    def after_train_iter(self, runner, batch_idx, data_batch, outputs):
        """Train Iteration 종료 후 호출"""
        global_iter = runner.iter

        # Train loss 추출
        train_loss = None

        # 방법 1: outputs dict에서 직접 가져오기 (가장 확실)
        if isinstance(outputs, dict) and 'loss' in outputs:
            train_loss = float(outputs['loss'])

        # 방법 2: message_hub에서 현재 loss 가져오기
        elif hasattr(runner, 'message_hub'):
            try:
                loss_buffer = runner.message_hub.get_scalar('train/loss')
                if loss_buffer is not None:
                    if hasattr(loss_buffer, 'current'):
                        train_loss = float(loss_buffer.current())
                    elif hasattr(loss_buffer, 'mean'):
                        train_loss = float(loss_buffer.mean())
            except Exception:
                pass

        # 방법 3: log_buffer 확인 (Fallback)
        if train_loss is None and hasattr(runner, 'log_buffer'):
            out = runner.log_buffer.output
            if 'loss' in out:
                try:
                    train_loss = float(out['loss'])
                except Exception:
                    pass

        # CSV에 기록
        if train_loss is not None:
            self._append_csv(global_iter, train_loss=train_loss)

        # Flush 체크
        now = time.time()
        if now - self._last_flush > self.flush_secs:
            self._last_flush = now

    def after_val_epoch(self, runner, metrics=None):
        """Validation Epoch 종료 후 호출 - mIoU 기록"""
        global_iter = runner.iter

        # Validation mIoU 가져오기
        val_miou = None

        # Method 1: metrics 파라미터에서 먼저 가져오기
        if metrics is not None and isinstance(metrics, dict):
            for key in ['mIoU', 'seg/mIoU', 'val/mIoU', 'IoUMetric/mIoU']:
                if key in metrics:
                    val_miou = float(metrics[key])
                    break

        # Method 2: message_hub에서 가져오기 (Fallback)
        if val_miou is None and hasattr(runner, 'message_hub'):
            try:
                for key in ['mIoU', 'seg/mIoU', 'val/mIoU']:
                    try:
                        metric_buffer = runner.message_hub.get_scalar(key)
                        if metric_buffer is not None:
                            if hasattr(metric_buffer, 'current'):
                                val_miou = float(metric_buffer.current())
                                break
                    except (KeyError, AttributeError):
                        continue
            except Exception as e:
                runner.logger.debug(f'Failed to get mIoU from message_hub: {e}')

        # 디버그 로깅
        if val_miou is not None:
            runner.logger.info(f'IterLoggerHook: Validation @ iter {global_iter} - mIoU={val_miou:.4f}')

        # CSV에 기록
        self._append_csv(global_iter, train_loss=None, val_mIoU=val_miou)

    def _append_csv(self, global_iter, train_loss=None, val_mIoU=None):
        """간소화된 CSV 기록 - Segmentation 전용 (iter, train_loss, val_mIoU)"""
        # None이면 빈 문자열로 처리
        str_train = f"{train_loss:.6f}" if train_loss is not None else ""
        str_val_miou = f"{val_mIoU:.6f}" if val_mIoU is not None else ""

        row = [global_iter, str_train, str_val_miou]

        with open(self.out_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
