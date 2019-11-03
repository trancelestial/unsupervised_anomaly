import torch
import json
from src.base.base_dataset import BaseDataset
from src.networks.variational import VariationalAutoencoder
from src.optim.vae_trainer import VAETrainer

class VAE(object):

    def __init__(self, proba=False, proba_nsamples=None):
        self.vae_net = VariationalAutoencoder(28, (40,28), (28,40), 16, proba=proba, proba_nsample=proba_nsamples)

        self.vae_trainer = None
        self.vae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None
        }

    def train(self,  dataset: BaseDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        # self.vae_net = VariationalAutoencoder(28, (40,28), (28,40), 16)
        # self.vae_net = self.vae_net.to(device)
        self.vae_optimizer_name = optimizer_name
        self.vae_trainer = VAETrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                                      n_jobs_dataloader)

        self.vae_net = self.vae_trainer.train(dataset, self.vae_net)

    def test(self, dataset: BaseDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        if self.vae_trainer is None:
            print(f'DJE JE TRENER BRE ALO?!')
            # self.trainer = VAETrainer(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
            #                           n_jobs_dataloader)
        self.vae_trainer.test(dataset, self.vae_net)
        self.results['test_auc'] = self.vae_trainer.test_auc
        self.results['test_time'] = self.vae_trainer.test_time
        self.results['test_scores'] = self.vae_trainer.test_score