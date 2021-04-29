import torch
from torch import nn

class DistancePredictionLoss(nn.Module):
    def __init__(self):
        super(DistancePredictionLoss, self).__init__()
        self.MSE = torch.nn.MSELoss(reduction="none")

    def forward(self, pd_contacts, gt_contacts):
        """
        Args:
            pd_contacts: prediction matrix (before softmax) with shape (batch_size, num_classes)
            gt_contacts: multi-label binary (int rather than binary)
        """

        loss_list = []
        for index, gt_contact in enumerate(gt_contacts):
            pd_contact = pd_contacts[index][0:gt_contact.shape[0], 0:gt_contact.shape[1]]
            loss_list.append(self.MSE(pd_contact, gt_contact).mean().unsqueeze(0))

        loss_list = torch.cat(loss_list, dim=0)
        loss = loss_list.mean()


        return loss
