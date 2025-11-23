import torch
from torch import nn
import pytorch_lightning as pl


class MLPMultiSource(pl.LightningModule):

    def __init__(
        self,
        input_dim_feat,
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

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features, image):
        features1 = self.model(features)
        features2 = self.cnn(image)
        logits = self.classifier(torch.cat([features1, features2], dim=0))
        return logits

    def training_step(self, batch, batch_idx):
        features, images, labels = batch
        logits = self(features, images)
        loss = self.loss_fn(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        features, images, _ = batch
        logits = self(features, images)
        return torch.argmax(logits, dim=1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # type: ignore
