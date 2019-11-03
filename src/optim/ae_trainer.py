from src.base.base_trainer import BaseTrainer
from src.base.base_dataset import BaseDataset
from src.base.base_net import BaseNet
from sklearn.metrics import average_precision_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np

class AETrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 1e-3, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

    def train(self, dataset: BaseDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        ae_net = ae_net.to(self.device)
        ae_net = ae_net.double()

        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        logger.info('Starting pretraining...')
        start_time = time.time()
        ae_net.train()
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

                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                n_batches += 1

            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'  Epoch {epoch+1}/{self.n_epochs}\t Time: {epoch_train_time:.3f}\t Loss: {loss_epoch/n_batches:.8f}')

        pretrain_time = time.time() - start_time
        logger.info(f'Pretraining time {pretrain_time:.3f}')
        logger.info('Finished pretraining.')

        return ae_net

    def test(self, dataset: BaseDataset, ae_net: BaseNet):
        logger = logging.getLogger()

        ae_net = ae_net.to(self.device)
        ae_net = ae_net.double()

        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        logger.info('Testing autoencoder...')
        loss_epoch = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = ae_net(inputs)
                scores = torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))
                loss = torch.mean(scores)

                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                loss_epoch += loss.item()
                n_batches += 1

        logger.info(f'Test set Loss: {loss_epoch/n_batches:.3f}')

        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        auc_pr = average_precision_score(labels, scores)
        logger.info(f'Test set AUC-PR: {auc_pr:.8f}')

        test_time = time.time() - start_time
        logger.info(f'Autoencoder testing time: {test_time:.3f}')
        logger.info('Finished testing autoencoder.')