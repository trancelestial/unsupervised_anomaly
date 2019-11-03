import logging
import torch.nn as nn
import numpy as np

class BaseNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.latent_dim = None # latent dimension

    def forward(self, *input):
        raise NotImplementedError

    def summary(self):
        net_parameters = filter(lambda p: p.requres_grad, self.parameters())
        print(f'[INFO] net_parameters: {net_parameters}')
        params = sum([np.prod(p.size())] for p in net_parameters)
        self.logger.info(f'Trainable parameters: {params}')
        self.logger.info(self)