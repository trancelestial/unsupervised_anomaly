import torch
import json
from src.base.base_dataset import BaseDataset
from src.networks.autoencoder import AutoEncoder, Encoder
from src.optim.deepSVDD_trainer import DeepSVDTrainer
from src.optim.ae_trainer import AETrainer

class DeepSVDD(object):

    def __init__(self, objective: str = 'one-class', nu: float = 0.1):
        assert objective in ('one-class', 'soft-boundary'), '[ASSERT FAILED] Invalid objective'
        self.objective = objective
        assert (0 < nu) & (nu <= 1), '[ASSERT FAILED] nu must be in (0,1] range.'
        self.nu = nu
        self.R = 0.0
        self.c = None

        self.net_name = None
        self.net = None

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None
        }

    def set_network(self, net_name):
        self.net_name = net_name
        self.net = Encoder(input_dim=28, units=(32,32), latent_dim=2)

    def train(self, dataset: BaseDataset, optimizer_name: str = 'adam', lr: float = 1e-3, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):

        self.optimizer_name = optimizer_name
        self.trainer = DeepSVDTrainer(self.objective, self.R, self.c, self.nu, optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())
        self.c = self.trainer.c.cpu().data.numpy().tolist()
        self.results['train_time'] = self.trainer.train_time

    def test(self, dataset: BaseDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        if self.trainer is None:
            self.trainer = DeepSVDTrainer(self.objective, self.R, self.c, self.nu, device=device,
                                          n_jobs_dataloader=n_jobs_dataloader)
        self.trainer.test(dataset, self.net)
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self, dataset: BaseDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        self.ae_net = AutoEncoder(input_dim=28, units_enc=(32,32), units_dec=(32,32), latent_dim=2)
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net)
        self.ae_trainer.test(dataset, self.ae_net)
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()
        # print(f'net_dict shape: {net_dict["fc3.weight"].shape}\n net_dict: {net_dict["fc3.weight"]}')
        ae_net_dict = self.ae_net.state_dict()

        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        # print(f'net_dict updated shape: {net_dict["fc3.weight"].shape}\n net_dict updated: {net_dict["fc3.weight"]}')
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({
            'R': self.R,
            'c': self.c,
            'net_dict': net_dict,
            'ae_net_dict': ae_net_dict
        }, export_model)

    def load_model(self, model_path, load_ae=False):
        model_dict = torch.load(model_path)

        self.R = model_dict['R']
        self.c = model_dict['c']
        self.ae_net.load_state_dict(model_dict['ae_net_dict'])
        if load_ae:
            if self.ae_net is None:
                self.ae_net = AutoEncoder()
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        with open(export_json, 'w') as f:
            json.dump(self.results, f)