import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel(object):
    def __init__(self, cfg) -> None:
        super(BaseModel, self).__init__()
        self.cfg = cfg

        self.optimizer_list = []
        self.log_dict = {}
    
    def cuda(self):
        raise NotImplemented
    
    def reset_grad(self):
        for opt in self.optimizer_list:
            opt.zero_grad()
    
    def group_step(self, opt_group):
        for opt in opt_group:
            opt.step()
        self.reset_grad()

    def set_optimizer(self, which_opt='Adam', lr=0.001):
        r"""Set the optimizers for each module here

        ::
            if which_opt == 'Adam':
                self.opt_backbone = optim.Adam(self.backbone.get_parameters(), lr=lr, betas=(0.9, 0.99))
                self.optimizer_lit.append(self.opt_backbone)

        :param which_opt: Optimizer type, defaults to 'Adam'
        :type which_opt: str, optional
        :param lr: Learing rate, defaults to 0.001
        :type lr: float, optional
        """
        raise NotImplemented

    def save_checkpoint(self, epoch, filename):
        """Save current model into state dict

        :param epoch: Current epoch
        :type epoch: int
        :param filename: Output file path
        :type filename: str
        """
        raise NotImplemented
    
    def train(self, epoch, train_loader, writer_dict):
        """Train function for one epoch

        :param epoch: Epoch number
        :type epoch: int
        :param train_loader: The dataloader for providing training samples
        :type train_loader: torch.utils.data.DataLoader
        :param writer_dict: Log output dict tensorboard writer etc.
        :type writer_dict: dict
        """
        raise NotImplemented

    def eval(self, epoch, val_loader, writer_dict):
        """Evaluation function for provided data loader

        :param epoch: Epoch number
        :type epoch: int
        :param val_loader: The dataloader for providing evaluation samples
        :type val_loader: torch.utils.data.DataLoader
        :param writer_dict: Log output dict tensorboard writer ect.
        :type writer_dict: dict
        """
        raise NotImplemented
