
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.classification import (
    stat_scores_multiple_classes
)
from pytorch_lightning.trainer import optimizers
from pytorch_lightning.utilities.distributed import rank_zero_only
from sentence_transformers.util import batch_to_device
from torch import Tensor
from torchmetrics.classification.stat_scores import StatScores
from klib import kdict
import pytorch_lightning as pl
from torch import optim, nn, sigmoid
from torchmetrics import Accuracy, F1, MatthewsCorrcoef
from sentence_transformers import SentenceTransformer, models
import torch
import transformers
from copy import deepcopy
class F1Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    #sklearn.metrics.f1_score
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-10, reduction='mean', pos_weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, predicted, actual):
        if predicted.ndim != 1 or actual.ndim != 1:
            print(
                f"Loss calculation, encountered following input shapes: predicted: {predicted.shape} {predicted}, actual: {actual.shape} {actual}")
        assert predicted.ndim <= 1
        assert actual.ndim <= 1

        predicted = torch.sigmoid(predicted)

        tp = (actual * predicted).sum().to(torch.float32)
        # tn = ((1 - actual) * (1 - predicted)).sum().to(torch.float32)
        fp = ((1 - actual) * predicted).sum().to(torch.float32)
        fn = (actual * (1 - predicted)).sum().to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        # f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)

        # return 1 - f1 to minimize
        if self.reduction == 'mean':
            # f1 is already mean
            return 1 - f1
        elif self.reduction == "sum":
            # simulate seperate loss for each observation in input (:= "sum")
            return (1 - f1) * len(predicted)
        else:
            return 1 - f1


class MCC(Metric):
    r"""
    Computes `Mathews Correlation Coefficient <https://en.wikipedia.org/wiki/Matthews_correlation_coefficient>`_:
    Forward accepts
    - ``preds`` (float or long tensor): ``(N, ...)`` or ``(N, C, ...)`` where C is the number of classes
    - ``target`` (long tensor): ``(N, ...)``
    If preds and target are the same shape and preds is a float tensor, we use the ``self.threshold`` argument.
    This is the case for binary and multi-label logits.
    If preds has an extra dimension as in the case of multi-class scores we perform an argmax on ``dim=1``.
    Args:
        labels: Classes in the dataset.
        pos_label: Treats it as a binary classification problem with given label as positive.
    """

    def __init__(
        self,
        labels,
        pos_label=None,
        compute_on_step=True,
        dist_sync_on_step=False,
        process_group=None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        self.labels = labels
        self.num_classes = len(labels)
        self.idx = None

        if pos_label is not None:
          self.idx = labels.index(pos_label)

        self.add_state("matthews_corr_coef",
                       default=torch.tensor(0), dist_reduce_fx="mean")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        tps, fps, tns, fns, _ = stat_scores_multiple_classes(
            pred=preds, target=target, num_classes=self.num_classes)

        if self.idx is not None:
          tps, fps, tns, fns = tps[self.idx], fps[self.idx], tns[self.idx], fns[self.idx]

        numerator = (tps * tns) - (fps * fns)
        denominator = torch.sqrt(
            ((tps + fps) * (tps + fns) * (tns + fps) * (tns + fns)))

        self.matthews_corr_coef = numerator / denominator
        #Replacing any NaN values with 0
        self.matthews_corr_coef[torch.isnan(self.matthews_corr_coef)] = 0

        self.total += 1

    def compute(self):
        """
        Computes Matthews Correlation Coefficient over state.
        """
        return self.matthews_corr_coef / self.total
class TransformerClassifier(pl.LightningModule):
    def __init__(self, lr=2e-5, num_classes=1) -> None:
        super().__init__()
        self.save_hyperparameters()
        # transformer_backbone = models.Transformer('paraphrase-TinyBERT-L6-v2')
        # pooling_model = models.Pooling(transformer_backbone.get_word_embedding_dimension())
        # dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

        self.transformer = SentenceTransformer('paraphrase-TinyBERT-L6-v2')
        self.transformer.max_seq_length = 512
        # print(self.transformer)

        self.classifier = nn.Sequential(
            nn.Linear(768, 334),
            nn.ReLU(),
            nn.Linear(334, 1)
        )
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = F1Loss()

        shared_metrics = nn.ModuleDict(dict(accuracy=Accuracy(num_classes=num_classes),
                                            f1=F1(num_classes=num_classes),
                                            mcc=MCC(labels=[1])),
                               )
        self.metrics = nn.ModuleDict(dict(_train=deepcopy(shared_metrics),
                                          val=deepcopy(shared_metrics),
                                          test=deepcopy(shared_metrics)))

        self.val_confusion = StatScores(num_classes=num_classes)
        self.test = []

    # def setup(self, stage):
        # self.transformer = self.transformer.to(self.device)
        # print('train start', self.device)

    def forward(self, x):
        # self.transformer = self.transformer.to(self.device)
        features = self.transformer.tokenize(x)
        features = batch_to_device(features, self.device)
        embeddings = self.transformer(features)['sentence_embedding']
        # embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # print(embeddings['sentence_embedding'], embeddings['sentence_embedding'].shape, embeddings['cls_token_embeddings'], embeddings['cls_token_embeddings'].shape)
        # embeddings = self.transformer.encode(
        #     x, convert_to_tensor=True, device=self.device)
        # print(embeddings)

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
            self.test.extend(labels)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("_train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)
    
    def validation_epoch_end(self, outputs) -> None:
        # if rank_zero_only.rank == 0:
        #     print(self.val_confusion.compute())
        print(self.val_confusion.compute())
        print(len([val for val in self.test if val == 0]),
              len([val for val in self.test if val == 1]))
        self.test = []
        self.val_confusion.reset()
        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        print("no weight decay layers", len([p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)]))
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return transformers.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr)
