import torch.nn as nn
import torch

# --------------------------------------------
# TV loss
# --------------------------------------------
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        """
        Total variation loss
        https://github.com/jxgu1016/Total_Variation_Loss.pytorch
        Args:
            tv_loss_weight (int):
        """
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# --------------------------------------------
# Charbonnier loss
# --------------------------------------------
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-9):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + self.eps))
        return loss


# --------------------------------------------
# CTformer loss
# --------------------------------------------
class CTformerLoss(nn.Module):
    """CTformer Loss (L2)"""

    def __init__(self, eps=1e-4):
        super(CTformerLoss, self).__init__()
        self.eps = eps
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        loss = self.criterion(x, y)*100 + 0.0001
        return loss


class TrainLoss:
    def __init__(self, G_lossfn_type):
        self.G_lossfn_type = G_lossfn_type

    def define_loss(self):
        G_lossfn_type = self.G_lossfn_type
        if G_lossfn_type == 'l1':
            G_lossfn = nn.L1Loss().cuda()
        if G_lossfn_type == 'l1sum':
            G_lossfn = nn.L1Loss(reduction='sum').cuda()
        elif G_lossfn_type == 'l2':
            G_lossfn = nn.MSELoss().cuda()
        elif G_lossfn_type == 'l2sum':
            G_lossfn = nn.MSELoss(reduction='sum').cuda()
        elif G_lossfn_type == 'charbonnier':
            G_lossfn = CharbonnierLoss(self.opt_train['G_charbonnier_eps']).cuda()
        elif G_lossfn_type == 'CTformerloss':
            G_lossfn = CTformerLoss().cuda()
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(G_lossfn_type))
        return G_lossfn
