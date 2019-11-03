import torch
import torch.nn as nn
import math
import numpy as np

from src.base.base_net import BaseNet

class VariationalAutoencoder(BaseNet):

    def __init__(self, input_dim=28, units_enc=(36,), units_dec=(32,), latent_dim=16, bias=True, proba=False, proba_nsample=None):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.proba = proba
        self.proba_nsample = proba_nsample

        #Encoder
        self.block_enc = []
        self.block_enc.append(nn.ModuleList([nn.Linear(self.input_dim, units_enc[0], bias=bias),
                                             nn.LeakyReLU(units_enc[0]),
                                             nn.BatchNorm1d(units_enc[0], affine=bias)]))
        for i in range(1, len(units_enc)):
            self.block_enc.append(nn.ModuleList([nn.Linear(units_enc[i - 1], units_enc[i], bias=bias),
                                                  nn.LeakyReLU(),
                                                  nn.BatchNorm1d(units_enc[i], affine=bias)]))
        self.block_enc = nn.ModuleList(self.block_enc)
        self.mu = nn.Linear(units_enc[-1], self.latent_dim, bias=bias)
        self.logvar = nn.Linear(units_enc[-1], self.latent_dim, bias=bias)

        #Decoder
        self.block_dec = []
        self.block_dec.append(nn.ModuleList([nn.Linear(self.latent_dim, units_dec[0], bias=bias),
                                             nn.LeakyReLU(units_dec[0]),
                                             nn.BatchNorm1d(units_dec[0], affine=bias)]))
        for i in range(1, len(units_dec)):
            self.block_dec.append(nn.ModuleList([nn.Linear(units_dec[i - 1], units_dec[i], bias=bias),
                                                  nn.LeakyReLU(),
                                                  nn.BatchNorm1d(units_dec[i], affine=bias)]))
        self.block_dec = nn.ModuleList(self.block_dec)
        if self.proba:
            self.output_mu = nn.Linear(units_dec[-1], self.input_dim, bias=bias)
            self.output_logvar = nn.Linear(units_dec[-1], self.input_dim, bias=bias)
        else:
            self.output = nn.Linear(units_dec[-1], self.input_dim, bias=bias)

    def encode(self, input):
        x = input
        for fc, relu, bn in self.block_enc:
            x = fc(x)
            x = bn(relu(x))
        return self.mu(x), self.logvar(x)

    def decode(self, latent):
        x = latent
        for fc, relu, bn in self.block_dec:
            x = fc(x)
            x = bn(relu(x))
        if self.proba:
            return self.output_mu(x), self.output_logvar(x)
        else:
            return self.output(x)

    def reparametrize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + torch.exp(0.5 * logvar) * eps

    def likelihood(self, input, mu, logvar):
        var = torch.exp(logvar)
        std_dev = torch.sqrt(torch.exp(logvar))
        z = input - mu
        # print(f'std-dev shape: {std_dev.shape}\nz shape: {z.shape}')
        # print(f'std-dev: {std_dev}\nmu: {mu}')
        # print(f'VARS: {var}')
        if torch.isnan(var).any():
            print(f'KURAAAAC VAR')
        if torch.isnan(std_dev).any():
            print(f'KURAAAAC STD_DEV')
        if torch.isnan(z).any():
            print(f'KURAAAAC Z')

        left = torch.sqrt(2*math.pi ** self.input_dim * torch.prod(var, dim=-1))
        right = torch.sum(torch.sqrt(torch.exp(z * 1 / std_dev * z)), dim=-1)

        prod = left*right

        lhood = 1. / prod


        if torch.isnan(left).any():
            print(f'KURAAAAC LEFT')
        if torch.isnan(right).any():
            print(f'KURAAAAC RIGHT')
        if torch.isnan(lhood).any():
            print(f'KURAAAAC LHOOD')
        if torch.isnan(prod).any():
            print(f'right shape: {right.shape}')
            print(f'left shape: {left.shape}')
            print(f'prod shape: {prod.shape}')
            print(f'KURAAAAC prod')
            print(left[torch.isnan(prod)])
            print(right[torch.isnan(prod)])
            exit()

        scores = torch.mean(lhood, dim=0)
        nanmask = torch.isnan(scores)
        if nanmask.any():
            print(f'---------------------------->Nasao!')

            # print(f'NaNs count: {torch.sum(nanmask)}\nmus: {mu[:,nanmask]}\nstd_devs: {std_dev[:,nanmask]}\n')

        return scores

    def forward(self, input):
        z_mu, z_logvar = self.encode(input)

        latent = self.reparametrize(z_mu, z_logvar)
        if self.proba:
            if self.proba_nsample == None:
                self.logger.error(f'Number of times to sample from latent space has to be provided when using probabilistic mode!')
                exit()
            latent = [self.reparametrize(z_mu, z_logvar) for i in range(self.proba_nsample)]
            # out_mu, out_logvar = self.decode(latent)
            out_params = torch.stack([torch.stack(self.decode(l)) for l in latent]) # make torch tensors of list of tensors
            output = torch.stack([self.reparametrize(out_mu, out_logvar) for out_mu, out_logvar in out_params])
        else:
            out_params = None
            output = self.decode(latent)
        return z_mu, z_logvar, output, out_params