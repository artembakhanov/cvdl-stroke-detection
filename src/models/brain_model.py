from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torchmetrics import Accuracy, Precision, Recall, F1, MaxMetric

from src.models.modules.brain_net import BrainNet


class BrainModel(LightningModule):
    def __init__(self,
                 input_size: int = 64,
                 sequence_size: int = 15,
                 hidden_dim: int = 64,
                 n_lstm_layers: int = 1,
                 lr: float = 0.001,
                 weight_decay: float = 0.0005,
                 ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = BrainNet(hparams=self.hparams)

        # loss function
        self.criterion = nn.BCEWithLogitsLoss()

        self.train_acc = Accuracy()
        self.test_acc = Accuracy()

        self.train_precision = Precision()
        self.test_precision = Precision()

        self.train_recall = Recall()
        self.test_recall = Recall()

        self.train_f1 = F1()
        self.test_f1 = F1()

        self.val_acc_best = MaxMetric()
        self.train_metrics = [self.train_acc, self.train_recall, self.train_precision, self.train_f1]
        self.test_metrics = [self.test_acc, self.test_recall, self.test_precision, self.test_f1]

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        sequences, targets = batch
        sequences = sequences.float()
        targets = targets.float()
        targets = torch.unsqueeze(targets, -1)
        out = self.forward(sequences)
        loss = self.criterion(out, targets)
        preds = out >= 0
        return loss, preds.int(), targets.int()

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        recall = self.train_recall(preds, targets)
        precision = self.train_precision(preds, targets)
        f1 = self.train_f1(preds, targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recall", recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/precision", precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/f1", f1, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "labels": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.test_acc(preds, targets)
        recall = self.test_recall(preds, targets)
        precision = self.test_precision(preds, targets)
        f1 = self.test_f1(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", recall, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/f1", f1, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets, "test/acc": acc}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("test/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        for metric in self.train_metrics + self.test_metrics:
            metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
