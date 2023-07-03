import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os.path as osp
import logging
import argparse

import options.options as option
import utils.util as util

from data import create_dataset, create_dataloader
from models import create_model

def esrgan():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--T_ctrl', type=float, default=1.0)
    parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
    opt = parser.parse_args()
    T_ctrl_temp = opt.T_ctrl
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)
    opt.T_ctrl = T_ctrl_temp
    T_ctrl_str = '%03d' % (opt.T_ctrl * 100)
    opt['name'] = opt['name'] + '_t' + T_ctrl_str
    opt['path']['results_root'] = opt['path']['results_root'] + '_t' + T_ctrl_str
    opt['path']['log'] = opt['path']['log'] + '_t' + T_ctrl_str
    opt['T_ctrl'] = opt.T_ctrl

    util.mkdirs(
        (path for key, path in opt['path'].items()
         if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
    util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                      screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
    test_loaders = []
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
        test_loaders.append(test_loader)

    model = create_model(opt)
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info('\nTesting [{:s}]...'.format(test_set_name))
        dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
        util.mkdir(dataset_dir)

        for data in test_loader:
            need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
            model.feed_data(data, need_GT=need_GT)
            img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]

            model.test(opt)
            visuals = model.get_current_visuals(need_GT=need_GT)
            sr_img = util.tensor2img(visuals['SR'])  # uint8
            cm_img = util.tensor2img(visuals['CM'])  # uint8

            # save images
            suffix = opt['suffix']
            if suffix:
                save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
            else:
                save_img_path = osp.join(dataset_dir, img_name + '.png')
            util.save_img(sr_img, save_img_path)

            save_cm_path = os.path.join(dataset_dir, '{:s}_cmap.png'.format(img_name))
            util.save_img(cm_img, save_cm_path)

            logger.info('{:20s}'.format(img_name))

if __name__ == '__main__':
    esrgan()
