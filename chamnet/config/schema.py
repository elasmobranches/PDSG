"""YAML 레시피를 읽고 검증한다. 학습 설정의 유일한 출처."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass(frozen=True)
class Data:
    root: str
    splits: dict
    size: tuple
    keep_ratio: bool


@dataclass(frozen=True)
class Augmentation:
    hflip_p: float
    brightness: float
    contrast: float
    photometric_p: float


@dataclass(frozen=True)
class Loss:
    ce_weight: float
    class_weight: list
    dice_weight: float
    dice_eps: float
    aux_weight: float


@dataclass(frozen=True)
class Optim:
    type: str
    lr: float
    weight_decay: float
    clip_grad: dict
    paramwise: dict
    per_backbone: dict


@dataclass(frozen=True)
class Schedule:
    max_iters: int
    warmup: dict
    poly: dict


@dataclass(frozen=True)
class Runtime:
    batch_size: int
    num_workers: int
    val_interval: int
    early_stopping: dict
    save_best: str
    seeds: list


@dataclass(frozen=True)
class Hd:
    depth_pretrained: bool


@dataclass(frozen=True)
class Recipe:
    data: Data
    augmentation: Augmentation
    loss: Loss
    optim: Optim
    schedule: Schedule
    runtime: Runtime
    hd: Hd

    def optim_for(self, backbone: str) -> dict:
        """per_backbone 을 병합한 optimizer + paramwise 설정을 돌려준다."""
        over = self.optim.per_backbone.get(backbone, {})
        opt = dict(type=self.optim.type, lr=self.optim.lr,
                   weight_decay=self.optim.weight_decay)
        for k, v in over.items():
            if k == 'paramwise':
                continue
            opt[k] = tuple(v) if k == 'betas' else v
        paramwise = _deep_merge(self.optim.paramwise, over.get('paramwise', {}))
        opt['paramwise_cfg'] = dict(custom_keys=paramwise)
        return opt


def _recipe_text(name_or_path: str | Path) -> str:
    """Resolve a recipe reference to its YAML text.

    A bare name with no path separators and no file suffix (e.g. 'paper_v13')
    names a recipe shipped inside the chamnet package
    (chamnet/recipes/<name>.yaml) and is resolved via importlib.resources, so
    it keeps working after `pip install chamnet` regardless of the caller's
    cwd — recipes/ living at the repo root was not picked up by the wheel
    (only `chamnet*` is packaged), so a cwd-relative default path broke for
    anyone but a checkout. Anything else — a relative/absolute filesystem
    path, or a name that already ends in .yaml/.yml — is read directly, so a
    user's own custom recipe file still works.
    """
    p = Path(name_or_path)
    if p.suffix == '' and len(p.parts) == 1:
        packaged = resources.files('chamnet').joinpath('recipes', f'{p.name}.yaml')
        if packaged.is_file():
            return packaged.read_text()
    return p.read_text()


def load_recipe(path: str | Path) -> Recipe:
    raw = yaml.safe_load(_recipe_text(path))
    d, a = raw['data'], raw['augmentation']
    ls, o, s, r = raw['loss'], raw['optim'], raw['schedule'], raw['runtime']
    return Recipe(
        data=Data(d['root'], d['splits'], tuple(d['size']), d['keep_ratio']),
        augmentation=Augmentation(a['hflip']['p'], a['photometric']['brightness'],
                                  a['photometric']['contrast'], a['photometric']['p']),
        loss=Loss(ls['cross_entropy']['weight'], list(ls['cross_entropy']['class_weight']),
                  ls['dice']['weight'], ls['dice']['eps'], ls['auxiliary']['weight']),
        optim=Optim(o['type'], float(o['lr']), o['weight_decay'], o['clip_grad'],
                    o['paramwise'], o.get('per_backbone', {})),
        schedule=Schedule(s['max_iters'], s['warmup'], s['poly']),
        runtime=Runtime(r['batch_size'], r['num_workers'], r['val_interval'],
                        r['early_stopping'], r['save_best'], list(r['seeds'])),
        hd=Hd(raw['hd']['depth_pretrained']),
    )
