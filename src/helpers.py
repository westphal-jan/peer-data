import torch
from torch import nn


class F1Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

   Inspired by Michal Haltuf on Kaggle.

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