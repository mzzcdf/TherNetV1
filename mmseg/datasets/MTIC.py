import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
from .multi_custom import Multi_CustomDataset
from .multi_gt_custom import Multi_GT_CustomDataset


@DATASETS.register_module()
class MTICDataset(Multi_CustomDataset):
    """ThermalB dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('rest', 'road', 'sidewalk', 'person', 'rider', 'passenger cars',
               'commercial vehicle', 'tricycle', 'motorcycle and bicycle', 'building', 'fence and guard rail',
               'bridge', 'pole', 'traffic sign', 'traffic light', 'vegetation', 'terrain', 'sky')

    PALETTE = [[0, 0, 0], [128, 64, 128], [244, 35, 232], [220, 20, 60], [255, 0, 0],
               [0, 0, 142], [0, 0, 70], [0, 60, 100], [119, 11, 32],
               [70, 70, 70], [190, 153, 153], [150, 100, 100], [153, 153, 153],
               [220, 220, 0], [250, 170, 30], [107, 142, 35], [152, 251, 152],
               [70, 130, 180]]

    def __init__(self, **kwargs):
        super(MTICDataset, self).__init__(
            img_suffix='.bmp', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)
