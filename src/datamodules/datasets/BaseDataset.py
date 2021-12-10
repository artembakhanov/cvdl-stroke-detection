import os
from typing import Optional, Callable, Dict, Tuple, Any

import pandas as pd
from torchvision.datasets import VisionDataset


class BaseDataset(VisionDataset):
    """
    Default dataset.
    """
    meta_file = ".download"

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            orphan: bool = False,
            image_size: int = 128,
            debug: bool = False
    ) -> None:
        super(BaseDataset, self).__init__(root, transform=transform,
                                          target_transform=target_transform)
        self.image_size = image_size
        self.debug = debug

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def _raw_folder(self):
        return os.path.join(self.root, "raw")

    def _download(self) -> None:
        """Download the data if it doesn't exist in processed_folder/image_size already."""
        raise NotImplementedError

    def _create_meta(self) -> None:
        open(os.path.join(self.root, self.meta_file), 'w').close()

    def _check_exists(self) -> bool:
        return os.path.exists(os.path.join(self.root,
                                           self.meta_file))

    @classmethod
    def root_folder(cls, data_dir, **kwargs):
        return os.path.join(data_dir, cls.__name__)
