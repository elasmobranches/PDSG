"""recipes/paper_v13.yaml's augmentation: block is documentation, not wiring —
_pipelines() never reads r.augmentation; ChamNetOnlineAugmentation's own
constructor defaults are what actually run. This test keeps the YAML honest:
if a default in augmentation.py ever changes, this fails instead of leaving
the YAML lying about training behaviour.
"""
from chamnet.config.schema import load_recipe
from chamnet.datasets.transforms.augmentation import ChamNetOnlineAugmentation


def test_recipe_augmentation_values_match_transform_defaults():
    r = load_recipe('paper_v13')
    aug = ChamNetOnlineAugmentation()

    hflip = aug.geometry.transforms[0]
    bc = aug.photometric.transforms[0]
    assert hflip.p == r.augmentation.hflip_p
    assert bc.brightness_limit == (-r.augmentation.brightness, r.augmentation.brightness)
    assert bc.contrast_limit == (-r.augmentation.contrast, r.augmentation.contrast)
    assert bc.p == r.augmentation.photometric_p
