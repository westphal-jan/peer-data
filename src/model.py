
from pytorch_lightning.utilities.distributed import rank_zero_only
from sentence_transformers.util import batch_to_device
from torch import Tensor
from torchmetrics.classification.stat_scores import StatScores
from klib import kdict
import pytorch_lightning as pl
from torch import optim, nn, sigmoid
from torchmetrics import Accuracy, F1
from sentence_transformers import SentenceTransformer, models
import torch
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
class TransformerClassifier(pl.LightningModule):
    def __init__(self, lr=0.001, num_classes=1) -> None:
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

        shared_metrics = kdict(accuracy=Accuracy(num_classes=num_classes, dist_sync_on_step=True),
                               f1=F1(num_classes=num_classes, dist_sync_on_step=True))
        self.metrics = kdict(
            train=shared_metrics.copy(),
            val=shared_metrics.copy(),
            test=shared_metrics.copy())

        self.val_confusion = StatScores(num_classes=num_classes, dist_sync_on_step=True)
        self.test = []

    # def setup(self, stage):
        # self.transformer = self.transformer.to(self.device)
        # print('train start', self.device)

    def forward(self, x):
        # self.transformer = self.transformer.to(self.device)
        features = self.transformer.tokenize(x)
        features = batch_to_device(features, self.device)
        embeddings = self.transformer(features)['sentence_embedding']
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # print(embeddings['sentence_embedding'], embeddings['sentence_embedding'].shape, embeddings['cls_token_embeddings'], embeddings['cls_token_embeddings'].shape)
        # embeddings = self.transformer.encode(
        #     x, convert_to_tensor=True, device=self.device)
        # print(embeddings)

        return self.classifier(embeddings)

    def _log_metrics(self, step_type: str, predictions: Tensor, labels: Tensor):
        metrics = self.metrics[step_type]
        for name, metric in metrics.items():
            self.log(f"{step_type}/{name}",
                     metric(predictions.cpu(), labels.cpu()))

    def _step(self, step_type: str, batch):
        data, labels = batch
        logits = self.forward(data).squeeze(1)
        loss = self.loss(logits, labels.type_as(logits))
        self.log(f'{step_type}/loss', loss)
        self.log(f'{step_type}/logits', logits.mean())
        self._log_metrics(step_type, sigmoid(logits), labels)

        if step_type == 'val':
            self.val_confusion(sigmoid(logits), labels)
        if step_type == 'train':
            self.test.extend(labels)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch)
    
    def validation_epoch_end(self, outputs) -> None:
        # if rank_zero_only.rank == 0:
        #     print(self.val_confusion.compute())
        print(self.val_confusion.compute())

        self.val_confusion.reset()
        return super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch)

    def configure_optimizers(self):

        return optim.Adam(self.parameters(), lr=self.hparams.lr)
