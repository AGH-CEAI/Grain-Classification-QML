import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class MLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim=100, output_dim=3, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # TorchMetrics
        metrics = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes=3),
                "f1": MulticlassF1Score(num_classes=3),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        logits = self(x)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, labels)
        self.log("train_loss", loss)
        return loss

    def on_train_epoch_end(self):
        # compute returns dictionary: {"train_acc": x, "train_f1": y}
        metrics = self.train_metrics.compute()
        # log each metric
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        logits = self(x)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.val_metrics.update(preds, labels)

    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute()
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)
        self.val_metrics.reset()

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore
