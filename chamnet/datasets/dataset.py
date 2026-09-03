# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class ChamNet(BaseSegDataset):
    """Chamoe Custom Dataset.

    Custom dataset for chamoe segmentation with 8 classes:
    - 0: background
    - 1: chamoe (참외)
    - 2: heatpipe (히트파이프)
    - 3: path (경로)
    - 4: pillar (기둥)
    - 5: topdownfarm (수직재배단)
    - 6: ceiling (천창)
    - 7: duct (덕트)
    """
    METAINFO = dict(
        classes=('background', 'chamoe', 'heatpipe', 'path', 'pillar', 
                 'topdownfarm', 'ceiling', 'duct'),
        palette=[[0, 0, 0],          # background - black
                 [255, 255, 0],      # chamoe - yellow
                 [255, 0, 0],        # heatpipe - red
                 [0, 255, 0],        # path - green
                 [0, 0, 255],        # pillar - blue
                 [255, 0, 255],      # topdownfarm - magenta
                 [128, 128, 128],    # ceiling - gray
                 [0, 255, 255]       # duct - cyan
                 ]    
    )

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
