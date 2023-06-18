import numpy as np
import torch
import torch.utils.data as data
import data.util as util


class LQDataset(data.Dataset):
    '''Read LQ images only in the test phase.'''

    def __init__(self, opt):
        super(LQDataset, self).__init__()
        self.opt = opt
        self.paths_LQ = None
        self.LQ_env = None  # environment for lmdb

        # read image list from lmdb or image files
        # self.LQ_env, self.paths_LQ = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(opt['data_type'], opt['dataroot_LQ'])
        assert self.paths_LQ, 'Error: LQ paths are empty.'

    def __getitem__(self, index):
        LQ_path = None

        # get LQ image
        LQ_path = self.paths_LQ[index]
        img_LQ = util.read_img(self.LQ_env, LQ_path)
        H, W, C = img_LQ.shape

        # change color space if necessary
        if self.opt['color']:
            img_LQ = util.channel_convert(C, self.opt['color'], [img_LQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        return {'LQ': img_LQ, 'LQ_path': LQ_path}

    def __len__(self):
        return len(self.paths_LQ)
