from typing import Optional, Dict

from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
import albumentations as A
from torch.utils.data import Dataset, DataLoader

from src.datamodules.datasets.BrainDataset import BrainDataset
from src.utils.utils import batch_to_tensor


class BrainDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str = "data/",
            image_size: int = 64,
            batch_size: int = 16,
            num_workers: int = 0,
            pin_memory: bool = False,
            rotate: Dict[str, bool] = {"train": False, "test": False},
            rot_seq_prob: Dict[str, float] = {"train": 0.4, "test": 0.4},
            sequence_size: Dict[str, int] = {"train": 15, "test": 15}
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.static_transform = A.Compose([
            A.Resize(image_size, image_size)
        ])
        self.test_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(0.5, 0.3),
            ToTensorV2(),
        ])

        self.dynamic_transform = A.Compose([
            A.GaussNoise(var_limit=(1, 9)),
            A.GaussianBlur(blur_limit=(1, 7)),
            A.GridDistortion(num_steps=3, p=0.3),
            A.OpticalDistortion(p=0.3),
            A.Normalize(0.5, 0.3),
            A.CoarseDropout(p=0.5),  # TRY: apply CourseDropout after Normalize
            ToTensorV2(),
        ])

        self.dims = (3, image_size, image_size)

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU."""
        BrainDataset(BrainDataset.root_folder(self.hparams.data_dir), download=True, orphan=True)

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_test:
            self.data_train = BrainDataset(BrainDataset.root_folder(self.hparams.data_dir),
                                           test=False,
                                           static_transform=self.static_transform,
                                           dynamic_transform=self.dynamic_transform,
                                           image_size=self.hparams["image_size"],
                                           rotate=self.hparams["rotate"]["train"],
                                           sequence_size=self.hparams["sequence_size"]["train"],
                                           rot_seq_prob=self.hparams["rot_seq_prob"]["train"],
                                           )
            self.data_test = BrainDataset(BrainDataset.root_folder(self.hparams.data_dir),
                                          test=True,
                                          static_transform=self.test_transform,
                                          dynamic_transform=None,
                                          image_size=self.hparams["image_size"],
                                          rotate=self.hparams["rotate"]["test"],
                                          sequence_size=self.hparams["sequence_size"]["test"],
                                          rot_seq_prob=self.hparams["rot_seq_prob"]["test"],
                                          )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            collate_fn=batch_to_tensor,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            collate_fn=batch_to_tensor,
            shuffle=False,
        )
