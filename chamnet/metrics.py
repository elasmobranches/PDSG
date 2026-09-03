"""Evaluation metric that reports per-class IoU alongside the aggregates."""
from mmseg.evaluation import IoUMetric
from mmseg.registry import METRICS


@METRICS.register_module()
class IoUMetricWithPerClass(IoUMetric):
    """IoUMetric that also puts each class's IoU into the returned dict.

    Vanilla IoUMetric.compute_metrics only returns the aggregate keys
    (aAcc, mIoU, mAcc, mDice, mFscore, ...) — per-class numbers are printed
    to the log table but never handed back to the caller. Both callers that
    have to write the campaign's CSV schema need them back: the sweep fills
    that schema's eight `val_IoU_*`/`test_IoU_*` columns, and the replay
    compares `IoU.pillar` against the recorded per-class column. So recompute
    the same per-class breakdown IoUMetric already does internally
    (`total_area_to_metrics`) and add it to the dict this method returns.

    Aggregate values are untouched — this is additive, and a run evaluated
    with it scores exactly what it would have scored with plain IoUMetric.
    That matters because the recorded numbers were produced with the plain
    one; the per-class values here are the same numbers the original run
    printed to its log and had scraped back out of it.

    Emitted key format is ``'<metric>.<class name>'`` (e.g. ``'IoU.pillar'``),
    which cannot collide with the aggregate keys — those carry no dot.
    """

    def compute_metrics(self, results):
        metrics = super().compute_metrics(results)
        areas = tuple(zip(*results))
        per_class = self.total_area_to_metrics(
            sum(areas[0]), sum(areas[1]), sum(areas[2]), sum(areas[3]),
            self.metrics, self.nan_to_num, self.beta)
        per_class.pop('aAcc', None)
        class_names = self.dataset_meta['classes']
        for key, values in per_class.items():
            for name, v in zip(class_names, values):
                metrics[f'{key}.{name}'] = round(float(v) * 100, 2)
        return metrics
