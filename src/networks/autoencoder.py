import torch
import torch.nn as nn
# import torch.nn.functional as F

from src.base.base_net import BaseNet

class Encoder(BaseNet):

    def __init__(self, input_dim, units, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.blocks = []
        self.blocks.append(nn.ModuleList([nn.Linear(self.input_dim, units[0], bias=False),
                          nn.LeakyReLU(),
                          nn.BatchNorm1d(units[0], affine=False)]))
        for i in range(1,len(units)):
            self.blocks.append(nn.ModuleList([nn.Linear(units[i-1], units[i], bias=False),
                              nn.LeakyReLU(),
                              nn.BatchNorm1d(units[i], affine=False)]))
        self.blocks = nn.ModuleList(self.blocks)
        self.latent = nn.Linear(units[-1], self.latent_dim, bias=False)

    def forward(self, x):
        for fc, relu, bn in self.blocks:
            x = bn(relu(fc(x)))
        x = self.latent(x)

        return x

class AutoEncoder(BaseNet):

    def __init__(self, input_dim=28, units_enc=(64,32), units_dec=(32,64), latent_dim=(16)):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.blocks_enc = []
        self.blocks_enc.append(nn.ModuleList([nn.Linear(self.input_dim, units_enc[0], bias=False),
                                              nn.LeakyReLU(),
                                              nn.BatchNorm1d(units_enc[0], affine=False)]))
        for i in range(1, len(units_enc)):
            self.blocks_enc.append(nn.ModuleList([nn.Linear(units_enc[i - 1], units_enc[i], bias=False),
                                                  nn.LeakyReLU(),
                                                  nn.BatchNorm1d(units_enc[i], affine=False)]))
        self.blocks_enc = nn.ModuleList(self.blocks_enc)
        self.latent = nn.Linear(units_enc[-1], self.latent_dim, bias=False)

        # Decoder
        self.blocks_dec = []
        self.blocks_dec.append(nn.ModuleList([nn.Linear(self.latent_dim, units_dec[0], bias=False),
                                              nn.LeakyReLU(),
                                              nn.BatchNorm1d(units_dec[0], affine=False)]))
        for i in range(1, len(units_dec)):
            self.blocks_dec.append(nn.ModuleList([nn.Linear(units_dec[i - 1], units_dec[i], bias=False),
                                                  nn.LeakyReLU(),
                                                  nn.BatchNorm1d(units_dec[i], affine=False)]))
        self.blocks_dec = nn.ModuleList(self.blocks_dec)
        self.output = nn.Linear(units_dec[-1], self.input_dim, bias=False)

    def encode(self, input):
        x = input
        for fc, relu, bn in self.blocks_enc:
            x = fc(x)
            x = bn(relu(x))
        return self.latent(x)

    def decode(self, latent):
        x = latent
        for fc, relu, bn in self.blocks_dec:
            x = fc(x)
            x = bn(relu(x))
        return self.output(x)

    def forward(self, x):
        # for fc, relu, bn in self.blocks_enc:
        #     x = fc(x)
        #     x = bn(relu(x))
        # x = self.latent(x)
        # for fc, relu, bn in self.blocks_dec:
        #     x = bn(relu(fc(x)))
        # x = self.output(x)
        # return x
        latent = self.encode(x)
        return self.decode(latent)

