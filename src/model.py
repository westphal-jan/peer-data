from numpy import true_divide
from sentence_transformers.util import batch_to_device
from torch import Tensor
import pytorch_lightning as pl
from torch import nn, sigmoid, tensor
from torchmetrics import Accuracy, F1, Recall, Precision, MatthewsCorrcoef, StatScores
from sentence_transformers import SentenceTransformer
import transformers
import torch
from copy import deepcopy

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .helpers import F1Loss

class TransformerClassifier(pl.LightningModule):
    def __init__(self, lr=2e-5, num_classes=1, accepted_class_weight=1, weight_decay=0.01, dropout_p=0, simple_linear_layer=False, f1_loss=False) -> None:
        super().__init__()
        self.save_hyperparameters()
   
        self.transformer = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
        self.transformer.max_seq_length = 512
        # print(self.transformer)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(768, 334),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(334, 1),
        )
        if simple_linear_layer is True:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(768, 1),
            )

        extra_weight_on_accepted = tensor(accepted_class_weight)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=extra_weight_on_accepted)
        if f1_loss is True:
            self.loss = F1Loss()

        shared_metrics = nn.ModuleDict(dict(accuracy=Accuracy(num_classes=num_classes),
                                            f1=F1(num_classes=num_classes),
                                            recall=Recall(
                                                num_classes=num_classes),
                                            precision=Precision(
                                                num_classes=num_classes),
                                            mcc=MatthewsCorrcoef(
                                                num_classes=2),
                                            ))
        self.metrics = nn.ModuleDict(dict(_train=deepcopy(shared_metrics),  # the `train` and `training` keywords cause an error with nn.ModuleDict
                                          val=deepcopy(shared_metrics),
                                          test=deepcopy(shared_metrics)))

        self.confusions = nn.ModuleDict(dict(_train=StatScores(num_classes=num_classes),
                                             val=StatScores(num_classes=num_classes), test=StatScores(num_classes=num_classes)))

    def forward(self, x):
        features = self.transformer.tokenize(x)
        features = batch_to_device(features, self.device)
        embeddings = self.transformer(features)['sentence_embedding']
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # embeddings = self.transformer.encode(
        #     x, convert_to_tensor=True, device=self.device)

        return self.classifier(embeddings)

    def _log_metrics(self, step_type: str, predictions: Tensor, labels: Tensor):
        metrics = self.metrics[step_type]
        for name, metric in metrics.items():
            metric(predictions, labels)
            self.log(f"{step_type}/{name}",
                     metric, on_epoch=True, on_step=False)
        self.confusions[step_type](predictions, labels)


    def _step(self, step_type: str, batch):
        data, labels = batch
        logits = self.forward(data).squeeze(1)
        loss = self.loss(logits, labels.type_as(logits))
        sync_dist = step_type in ['val', 'test']
        self.log(f'{step_type}/loss', loss, sync_dist=sync_dist)
        self.log(f'{step_type}/logits', logits.mean(), sync_dist=sync_dist)
        self._log_metrics(step_type, sigmoid(logits), labels)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log_confusion_matrix("_train")
        return super().on_epoch_end()

    def on_validation_epoch_end(self) -> None:
        self.log_confusion_matrix("val")
        return super().on_epoch_end()

    def on_test_epoch_end(self) -> None:
        self.log_confusion_matrix("test")
        return super().on_epoch_end()

    def log_confusion_matrix(self, step_type):
        confusion_metric = self.confusions[step_type]
        TP, FP, TN, FN, _ = confusion_metric.compute()
        confusion_metric.reset()
        self.log(f"{step_type}/TP",
                 TP, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{step_type}/FP",
                 FP, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{step_type}/TN",
                 TN, on_epoch=True, on_step=False, sync_dist=True)
        self.log(f"{step_type}/FN",
                 FN, on_epoch=True, on_step=False, sync_dist=True)

    def training_step(self, batch, batch_idx):
        return self._step("_train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):
        # From SentenceTransformer.fit function
        model_params = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(
                nd in n for nd in no_decay)], 'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in model_params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(
            optimizer_grouped_parameters, lr=self.hparams.lr)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, 2, eta_min=1e-8)
        return {'optimizer': optimizer}
