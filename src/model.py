
from torch import Tensor
from helpers.klib import kdict
import pytorch_lightning as pl
from torch import optim, nn, sigmoid
from torchmetrics import Accuracy, F1
from sentence_transformers import SentenceTransformer


class TransformerClassifier(pl.LightningModule):
    def __init__(self, lr=0.001, num_classes=1) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.transformer = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
        print(self.transformer)
        self.classifier = nn.Sequential(
            nn.Linear(768, 334),
            nn.ReLU(),
            nn.Linear(334, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
        )
        self.loss = nn.BCEWithLogitsLoss()

        shared_metrics = kdict(accuracy=Accuracy(num_classes=num_classes),
                               f1=F1(num_classes=num_classes))
        self.metrics = kdict(
            train=shared_metrics.copy(),
            val=shared_metrics.copy(),
            test=shared_metrics.copy())

    def forward(self, x):
        embeddings = self.transformer.encode(
            x, convert_to_tensor=True, device=self.device)
        # print(embeddings)
        return self.classifier(embeddings)

    def _log_metrics(self, step_type: str, predictions: Tensor, labels: Tensor):
        metrics = self.metrics[step_type]
        for name, metric in metrics.items():
            self.log(f"{step_type}/{name}", metric(predictions, labels))

    def _step(self, step_type: str, batch):
        data, labels = batch
        print(data, labels, self.metrics)
        logits = self.forward(data).squeeze(1)
        loss = self.loss(logits, labels.type_as(logits))
        self.log(f'{step_type}/loss', loss)
        print(logits)
        self._log_metrics(step_type, sigmoid(logits), labels)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)
