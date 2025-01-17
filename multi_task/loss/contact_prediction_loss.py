import torch
from torch import nn

class ContactPredictionLoss(nn.Module):
    def __init__(self):
        super(ContactPredictionLoss, self).__init__()
        self.BCE = torch.nn.BCELoss(reduction="none")

    def forward(self, pd_contacts, gt_contacts):
        """
        Args:
            pd_contacts: prediction matrix (before softmax) with shape (batch_size, num_classes)
            gt_contacts: multi-label binary (int rather than binary)
        """

        loss_list = []
        for index, gt_contact in enumerate(gt_contacts):
            pd_contact = pd_contacts[index][0:gt_contact.shape[0], 0:gt_contact.shape[1]]
            gt_contact = gt_contact.gt(0.07).float()  # th
            loss_list.append(self.BCE(pd_contact, gt_contact).mean().unsqueeze(0))

        loss_list = torch.cat(loss_list, dim=0)
        loss = loss_list.mean()


        return loss
