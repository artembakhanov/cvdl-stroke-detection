import glob
import os
import shutil
from typing import Optional, Callable
from urllib.error import URLError

import cv2
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive

from src.datamodules.datasets.BaseDataset import BaseDataset
from src.utils import utils
import pathlib

log = utils.get_logger(__name__)


class AtlasDataset(BaseDataset):
    file_name = "atlas.zip"
    file_ids = {10: "1WnNGaCaZGR8CV-LBM3r-yuDhtJ9DQ664",
                7: "1nJLjvpzOk6ZPKP3vC8a9UWa6Iu_wanEM",
                5: "1nm3078vMwI7CzwfdWxxV1lfrKA4KzwQV"}

    folder_names = {
        10: "ATLAS_REFORMED_10",
        7: "ATLAS_REFORMED_TRIPLED_7",
        5: "ATLAS_REFORMED_TIMES4_5"
    }
    classes = {'no_stroke': 0, 'stroke': 1}

    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 download: bool = False,
                 orphan: bool = False,
                 image_size: int = 64,
                 debug: bool = False,
                 rotate: bool = False,
                 rot_seq_prob: float = 0.4,
                 test: bool = False,
                 sequence_size: int = 10):
        self.sequence_size = sequence_size

        super(AtlasDataset, self).__init__(root, transform, target_transform, download, orphan, image_size,
                                           debug)

        if orphan:
            return
        self.rotate = rotate
        self.rot_seq_prob = rot_seq_prob
        self.test = test

        self.images_sequences = {}
        self.idx_to_seq = {}
        self.rot_seq_prob = rot_seq_prob
        self.distributions = []
        self.rotate = rotate

        self.samples = []
        self.idx_to_cls = []

        self._get_samples()

    def __getitem__(self, idx):
        seq = []
        for i in range(1, self.sequence_size + 1):
            seq.append(cv2.imread(str(self.samples[idx] / f"{i}.png"), 0))

        seq = list(map(
            lambda img: self.transform(image=img)["image"] if self.transform else img,
            seq
        ))

        return seq, self.idx_to_cls[idx]

    def __len__(self):
        return len(self.samples)

    def _get_samples(self):
        root = pathlib.Path(self.root) / ('test' if self.test else 'train')

        for cls in self.classes:
            seqs = [f for f in (root / cls).iterdir() if f.is_dir()]
            self.samples.extend(seqs)
            self.idx_to_cls.extend([self.classes[cls]] * len(seqs))

    def _download(self) -> None:
        """Download the brain strokes data if it doesn't exist already."""

        if self._check_exists():
            log.info(f"{self.__class__.__name__} is already downloaded. No need to download")
            return

        # download files
        try:
            log.info("Downloading {}...".format(self.file_name))
            download_file_from_google_drive(self.file_ids[self.sequence_size], self._raw_folder,
                                            filename=self.file_name)
            log.info("Extracting {}...".format(self.file_name))
            extract_archive(os.path.join(self._raw_folder, self.file_name), self.root)

            files = glob.glob(f"{self.root}/{self.folder_names[self.sequence_size]}/*")
            for f in files:
                shutil.move(f, self.root)

            shutil.rmtree(f"{self.root}/{self.folder_names[self.sequence_size]}/")
            self._create_meta()
        except URLError as error:
            log.error(
                "Failed to download (trying next):\n{}".format(error)
            )
        finally:
            pass

    @classmethod
    def root_folder(cls, data_dir, sequence_size=10, **kwargs):
        return os.path.join(super(AtlasDataset, cls).root_folder(data_dir), f"seq{sequence_size}")
