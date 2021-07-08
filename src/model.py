from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_info, _get_rank
from sentence_transformers.util import batch_to_device
from torch import Tensor
from torchmetrics.classification.stat_scores import StatScores
import pytorch_lightning as pl
from torch import nn, sigmoid
from torchmetrics import Accuracy, F1
from sentence_transformers import SentenceTransformer
import transformers
from copy import deepcopy
class TransformerClassifier(pl.LightningModule):
    def __init__(self, lr=2e-5, num_classes=1) -> None:
        super().__init__()
        self.save_hyperparameters()
   
        self.transformer = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
        self.transformer.max_seq_length = 512
        # print(self.transformer)

        # self.classifier = nn.Sequential(
        #     nn.Linear(768, 334),
        #     nn.ReLU(),
        #     nn.Linear(334, 1)
        # )
        self.classifier = nn.Linear(768, 1)

        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = F1Loss()

        shared_metrics = nn.ModuleDict(dict(accuracy=Accuracy(num_classes=num_classes),
                                            f1=F1(num_classes=num_classes)))
        self.metrics = nn.ModuleDict(dict(_train=deepcopy(shared_metrics), # the `train` and `training` keywords cause an error with nn.ModuleDict
                                          val=deepcopy(shared_metrics),
                                          test=deepcopy(shared_metrics)))

        self.val_confusion = StatScores(num_classes=num_classes)
        self.class_balance_check = []


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
            self.log(f"{step_type}/{name}",
                     metric(predictions, labels))

    def _step(self, step_type: str, batch):
        data, labels = batch
        logits = self.forward(data).squeeze(1)
        loss = self.loss(logits, labels.type_as(logits))
        self.log(f'{step_type}/loss', loss)
        self.log(f'{step_type}/logits', logits.mean())
        self._log_metrics(step_type, sigmoid(logits), labels)

        if step_type == 'val':
            self.val_confusion(sigmoid(logits), labels)
        if step_type == '_train':
            self.class_balance_check.extend(labels)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("_train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)

    def validation_epoch_end(self, outputs) -> None:
        print(self.val_confusion.compute())
        self.val_confusion.reset()

        print(len([val for val in self.class_balance_check if val == 0]),
              len([val for val in self.class_balance_check if val == 1]))
        self.class_balance_check = []
        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):
        # From SentenceTransformer.fit function
        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return transformers.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
