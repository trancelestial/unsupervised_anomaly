import torch
import logging
import numpy as np
import matplotlib.pyplot as plt

from src.datasets.kaggle_old import OldKaggle
from src.deepSVDD import DeepSVDD

pretraining = True
PRETRAIN_EPOCHS = 50 #(100, 32-20-32, 0.63)
PRETRAIN_MILESTONES = (40,60,90,120,140)

LR = 1e-3
EPOCHS = 20
MILESTONE_RATE = 7
MILESTONES = (20,40,60,90,120,140)
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

    deep_SVDD = DeepSVDD('one-class',nu=0.002)
    deep_SVDD.set_network('encoder')

    logger.info(f'Pretraining {pretraining}')
    # print(f'--->{dataset.train_set[0]}')
    if pretraining:
        deep_SVDD.pretrain(dataset,
                           optimizer_name='adam',
                           lr=LR,
                           n_epochs=PRETRAIN_EPOCHS,
                           lr_milestones=PRETRAIN_MILESTONES,
                           batch_size=BATCH_SIZE,
                           weight_decay=WEIGHT_DECAY,
                           device=device,
                           n_jobs_dataloader=4)

    logger.info(f'Training...')

    deep_SVDD.train(dataset,
                    optimizer_name='adam',
                    lr=LR,
                    n_epochs=EPOCHS,
                    lr_milestones=MILESTONES,
                    batch_size=BATCH_SIZE,
                    weight_decay=WEIGHT_DECAY,
                    device=device,
                    n_jobs_dataloader=4)

    deep_SVDD.test(dataset, device=device, n_jobs_dataloader=4)

    indices, labels, scores = zip(*deep_SVDD.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)

    print(f'R: {deep_SVDD.R}\nc: {deep_SVDD.c}')


    anomalies = scores[np.argwhere(labels)]
    normals = scores[np.argwhere(labels == 0)]
    plt.scatter(np.arange(len(normals)), normals, c='b', s=5)
    plt.scatter(np.linspace(0,len(indices),len(anomalies)), anomalies, c='r', s=5)
    plt.show()

    # idx_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]

    deep_SVDD.save_results(export_json='results.json')
    deep_SVDD.save_model('model.tar')

if __name__ == '__main__':
    main()