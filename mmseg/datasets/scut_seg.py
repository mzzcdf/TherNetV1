import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
from .multi_custom import Multi_CustomDataset


@DATASETS.register_module()
class SCUTDataset(CustomDataset):
    """ThermalB dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """

    CLASSES = ('background', 'road', 'fence', 'pole', 'tree', 'person', 'rider','car', 'truck', 'bus')

    PALETTE = [[0, 0, 0], [128, 64, 128], [153, 153, 190], [153, 153, 153], [35, 142, 107], [60, 20, 220], [0, 0, 255], [142, 0, 0], [70, 0, 0], [100, 60, 0]]

    def __init__(self, **kwargs):
        super(SCUTDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        assert osp.exists(self.img_dir)