import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SODADataset(CustomDataset):
    """ThermalB dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('_background_', 'person', 'building', 'tree', 'road', 'pole', 'grass', 'door', 'table', 'chair', 'car', 'bicycle', 'lamp', 'monitor', 'trafficCone', 'trash can', 'animal', 'fence', 'sky', 'river', 'sidewalk')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
               [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
               [0, 102, 200]
               ]

    def __init__(self, **kwargs):
        super(SODADataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)
        assert osp.exists(self.img_dir)