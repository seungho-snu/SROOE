import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util


class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.BTMap_on = 1
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb
        self.paths_LPIPS_map=[]
        self.sizes_LPIPS_map=[]
        # self.t_num = 13
        self.t_num = 1

        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb
        self.GT_LPIPS_map_env = None

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])

        if self.BTMap_on==1:
            if self.opt['phase'] == 'train':
                self.paths_LPIPS_map, self.sizes_LPIPS_map_temp = util.get_image_paths(self.data_type, opt['dataroot_T_OOS_map'])
                for idx in range(len(self.paths_LPIPS_map)):
                    self.paths_LPIPS_map[idx] = self.paths_LPIPS_map[idx].replace('\\', '/')

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None) or (self.LQ2_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        GT_LPIPS_map_path = None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]

        if self.opt['phase'] == 'train':
            GT_LPIPS_map_path = self.paths_LPIPS_map[index]

        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT_temp = util.read_img(self.GT_env, GT_path, resolution)

        img_GT_LPIPS_map_temp=None
        if self.BTMap_on == 1:
            if self.opt['phase'] == 'train':
                img_GT_LPIPS_map_temp = util.read_img(self.GT_LPIPS_map_env, GT_LPIPS_map_path, resolution)
                # img_GT_LPIPS_map_temp = util.imresize_np2(img_GT_LPIPS_map_temp, 1 / scale, True)

        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT_temp = util.modcrop(img_GT_temp, scale)

        # change color space if necessary
        if self.opt['color']:
            img_GT_temp = util.channel_convert(img_GT_temp.shape[2], self.opt['color'], [img_GT_temp])[0]

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ_temp = util.read_img(self.LQ_env, LQ_path, resolution)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT_temp.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT_temp), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT_temp.shape
            # using matlab imresize
            img_LQ_temp = util.imresize_np(img_GT_temp, 1 / scale, True)
            if img_LQ_temp.ndim == 2:
                img_LQ_temp = np.expand_dims(img_LQ_temp, axis=2)

        img_GT_LPIPS_map = img_LQ_temp
        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT_temp.shape
            if H < GT_size or W < GT_size:
                img_GT_temp = cv2.resize(np.copy(img_GT_temp), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT_temp, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ_temp.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ_temp[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]

            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT_temp[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'], self.opt['use_rot'])

            if self.BTMap_on == 1:
                img_GT_LPIPS_map = img_GT_LPIPS_map_temp[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
                img_GT_LPIPS_map = util.augment([img_GT_LPIPS_map], self.opt['use_flip'], self.opt['use_rot'])
                img_GT_LPIPS_map = img_GT_LPIPS_map[0]

        else:
            H1_temp, W1_temp, C1_temp = img_GT_temp.shape

            H2 = (H1_temp // scale)
            W2 = (W1_temp // scale)

            H1 = H2 * scale
            W1 = W2 * scale

            img_GT = img_GT_temp[0:H1, 0:W1, :]
            img_LQ = img_LQ_temp[0:H2, 0:W2, :]

        # change color space if necessary
        if self.opt['color']:
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        elif img_GT.shape[2] == 1:
            H, W, C = img_GT.shape
            img_GT = np.repeat(img_GT.reshape(H, W, 1), 3, axis=2)
            H, W, C = img_LQ.shape
            img_LQ = np.repeat(img_LQ.reshape(H, W, 1), 3, axis=2)
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if self.BTMap_on == 1:
            if self.opt['phase'] == 'train':
                img_GT_LPIPS_map = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT_LPIPS_map, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path

        if self.BTMap_on == 1:
            return {'LQ': img_LQ, 'GT': img_GT, 'LPIPS_maps': img_GT_LPIPS_map, 'LQ_path': LQ_path, 'GT_path': GT_path}
        else:
            return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}


    def __len__(self):
        return len(self.paths_GT)
