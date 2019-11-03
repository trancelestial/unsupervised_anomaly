from src.base.base_trainer import BaseTrainer
from src.base.base_dataset import BaseDataset
from src.base.base_net import BaseNet
from sklearn.metrics import average_precision_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

class VAETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 1e-3, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def vae_loss(self, inputs, outputs, mu, logvar):
        scores = torch.sum((outputs - inputs) ** 2, dim=-1)
        # print(scores)
        # probabilistic - additional dimension for number of samples took from distribution
        if outputs.dim() == 3:
            # mean over distribution samples
            scores = torch.mean(scores, dim=0)
            # print(scores)
        r_loss = torch.mean(scores)
        # print(r_loss)

        kl_mat = -0.5 * torch.sum(1. + logvar - torch.exp(logvar) - mu ** 2, -1) # last param to 1
        kl_loss = torch.mean(kl_mat)
        # print(f'R-loss: {r_loss}\nKL-loss: {kl_loss}')

        return r_loss + kl_loss

    def train(self, dataset: BaseDataset, vae_net: BaseNet):
        logger = logging.getLogger()

        vae_net = vae_net.to(self.device)
        vae_net.double()

        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        optimizer = optim.Adam(vae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self.lr_milestones, gamma=0.1)

        logger.info('Starting VAE training...')
        start_time = time.time()
        vae_net.train()
        for epoch in range(self.n_epochs):
            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info(f'   LR scheduler: new learning rate is {scheduler.get_lr()[0]}')

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                optimizer.zero_grad()

                mu, logvar, outputs, output_dparams = vae_net(inputs)
                loss = self.vae_loss(inputs, outputs, mu, logvar)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'  Epoch {epoch + 1}/{self.n_epochs}\t Time: {epoch_train_time:.3f}\t Loss: {loss_epoch / n_batches:.8f}')

        train_time = time.time() - start_time
        logger.info(f'Training time {train_time:.3f}')
        logger.info('Finished training.')

        return vae_net

    def test(self, dataset: BaseDataset, vae_net: BaseNet):
        logger = logging.getLogger()

        vae_net = vae_net.to(self.device)
        vae_net = vae_net.double()

        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        logger.info('Testing VAE...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        vae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                _, _, outputs, out_params = vae_net(inputs)
                # print(f'------>{outputs.shape} --- {out_params.shape}')
                # print(f'---> {inputs.shape} {out_params[0][0].shape} {out_params[0][1].shape}')
                if vae_net.proba:
                    scores = vae_net.likelihood(inputs,out_params[:,0],out_params[:,1])
                    # print(f'Scores shape: {scores.shape}\nscores: {scores}')
                    # exit()
                else:
                    scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                    print(f'Scores shape: {scores.shape}')
                    loss = torch.mean(scores)
                    loss_epoch += loss.item()
                    n_batches += 1
                # exit()

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))


        if not vae_net.proba:
            logger.info(f'Test set Loss: {loss_epoch / n_batches:.3f}')

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        if np.any(np.isnan(scores)):
            print(f'NaNs in scores!')
        if np.any(np.isinf(scores)):
            print(f'Infinity in scores!')


        auc_pr = average_precision_score(labels, scores)
        logger.info(f'Test set AUC-PR: {auc_pr:.8f}')

        test_time = time.time() - start_time
        logger.info(f'Testing time: {test_time:.3f}')

        self.test_auc = auc_pr
        self.test_time = test_time
        self.test_score = scores

        logger.info('Finished testing.')

