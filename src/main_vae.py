import torch
import logging

from src.datasets.kaggle_old import OldKaggle
from src.VAE import VAE

LR = 1e-3
EPOCHS = 1
MILESTONES = (30,60,90)
BATCH_SIZE = 128
WEIGHT_DECAY = 1e-6

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = 'log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
    logger.info(f'Computation device: {device}')

    dataset = OldKaggle()

    # logger.info(f'--------------------VAE REGULAR')
    #
    # vae = VAE(proba=False)
    # # vae.vae_net = vae.vae_net.to(device)
    # vae.train(dataset=dataset,
    #           optimizer_name='adam',
    #           lr=LR,
    #           n_epochs=EPOCHS,
    #           lr_milestones=MILESTONES,
    #           batch_size=BATCH_SIZE,
    #           weight_decay=WEIGHT_DECAY,
    #           device=device,
    #           n_jobs_dataloader=4)
    #
    #
    # vae.test(dataset, device=device, n_jobs_dataloader=4)

    logger.info(f'--------------------VAE PROBABILITY nsamples: 3')
    vae = VAE(proba=True, proba_nsamples=3)
    # vae.vae_net = vae.vae_net.to(device)
    vae.train(dataset=dataset,
              optimizer_name='adam',
              lr=LR,
              n_epochs=EPOCHS,
              lr_milestones=MILESTONES,
              batch_size=BATCH_SIZE,
              weight_decay=WEIGHT_DECAY,
              device=device,
              n_jobs_dataloader=4)
    # logger.info(f'')

    vae.test(dataset, device=device, n_jobs_dataloader=4)


if __name__ == '__main__':
    main()