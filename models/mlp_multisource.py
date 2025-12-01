import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class MLPMultiSource(pl.LightningModule):

    def __init__(
        self,
        input_dim_feat=12,
        hidden_dim_feat=100,
        output_dim_feat=3,
        in_channels=1,
        hidden_channels=16,
        hidden2_channels=8,
        output_img_dim=3,
        kernel_size=3,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Linear(input_dim_feat, hidden_dim_feat),
            nn.ReLU(),
            nn.Linear(hidden_dim_feat, output_dim_feat),
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(hidden_channels, hidden2_channels, kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32)),
            nn.Flatten(),
            nn.Linear(32 * 32 * hidden2_channels, output_img_dim),
        )

        self.classifier = nn.Linear(output_img_dim + output_dim_feat, 3)  # 18 params

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

    def forward(self, features, image):
        features1 = self.model(features)
        features2 = self.cnn(image)
        logits = self.classifier(torch.cat([features1, features2], dim=1))
        return logits

    def training_step(self, batch, batch_idx):
        features, images, labels = batch
        logits = self(features, images)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.train_metrics.update(preds, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False)

        return loss

    def on_train_epoch_end(self):
        # compute returns dictionary: {"train_acc": x, "train_f1": y}
        metrics = self.train_metrics.compute()
        # log each metric
        for k, v in metrics.items():
            self.log(k, v, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        features, images, labels = batch
        logits = self(features, images)
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
        features, images, _ = batch
        logits = self(features, images)
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore
