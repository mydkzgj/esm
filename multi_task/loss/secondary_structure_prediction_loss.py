from torch import nn

class SecondaryStructurePredictionLoss(nn.Module):
    """Multi-label Cross entropy loss.
    Reference: None
    Args: None
    """
    def __init__(self):
        super(SecondaryStructurePredictionLoss, self).__init__()

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: multi-label binary
        """
        loss = 0

        return loss
