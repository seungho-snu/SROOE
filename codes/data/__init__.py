'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset):
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    from data.LQ_dataset import LQDataset as D

    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
