# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117

import datetime
import os
import random
import traceback

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from tqdm.autonotebook import tqdm
from pycocotools.coco import COCO

from option import get_args
from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, CustomToTensor, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from dataprocessor import Mosaic, HorizontalFlip, RandomBrightnessContrast, \
    MedianBlur, ToGray, HueSaturationValue, Mixup, GaussianNoise, \
    RandomRotate, VerticalFlip, JpegCompression, RandomSizedCrop


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def train_val_split(opt, params):
    if os.path.isfile(os.path.join(opt.saved_path, 'split_ids.txt')):
        while True:
            print('Split list is found. Resume? (y/n) ', end='')
            reply = input()
            if reply == 'y':
                break
            elif reply == 'n':
                return [], []
            else:
                print('Not recognized, try again.')
        with open(os.path.join(opt.saved_path, 'split_ids.txt'), 'r') as f:
            next(f)
            train_ids = [int(id) for id in f.readline().split(' ')[:-1]]
            next(f)
            val_ids = [int(id) for id in f.readline().split(' ')[:-1]]
    else:
        coco = COCO(os.path.join(opt.data_path,
                                 params.project_name,
                                 'annotations',
                                 params.train_set + '.json'))
        image_ids = coco.getImgIds()
        random.shuffle(image_ids)
        train_ids = image_ids[:int(len(image_ids)*opt.train_split)]
        val_ids = image_ids[int(len(image_ids)*opt.train_split):]
        with open(os.path.join(opt.saved_path, 'split_ids.txt'), 'w') as f:
            f.write('train_ids\n')
            for id in train_ids:
                f.write(str(id) + ' ')
            f.write('\nval_ids\n')
            for id in val_ids:
                f.write(str(id) + ' ')
            f.write('\n')

    return train_ids, val_ids


def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    if opt.force_input_size is not None:
        input_sizes = {opt.compound_coef: opt.force_input_size}

    train_ids, val_ids = train_val_split(opt, params)
    if len(train_ids) == 0 or len(val_ids) == 0:
        return

    augmentations = transforms.Compose([
        RandomSizedCrop((0.1, 1.0), input_sizes[opt.compound_coef], input_sizes[opt.compound_coef]),
        HorizontalFlip(p=opt.aug_prob),
        VerticalFlip(p=opt.aug_prob),
        RandomRotate(p=opt.aug_prob),
        RandomBrightnessContrast(p=opt.aug_prob),
        HueSaturationValue(p=opt.aug_prob),
        ToGray(p=0.005),
        MedianBlur(p=opt.aug_prob),
        GaussianNoise(p=opt.aug_prob),
        JpegCompression(p=opt.aug_prob),
        Mosaic(
            annot_file=os.path.join(opt.data_path, params.project_name, 'annotations/train.json'),
            image_folder=os.path.join(opt.data_path, params.project_name, params.train_set),
            image_ids=train_ids,
            transform=transforms.Compose([
                RandomSizedCrop((0.1, 1.0), input_sizes[opt.compound_coef], input_sizes[opt.compound_coef]),
                HorizontalFlip(p=opt.aug_prob),
                VerticalFlip(p=opt.aug_prob),
                RandomRotate(p=opt.aug_prob),
                RandomBrightnessContrast(p=opt.aug_prob),
                HueSaturationValue(p=opt.aug_prob),
                ToGray(p=0.005),
                MedianBlur(p=opt.aug_prob),
                GaussianNoise(p=opt.aug_prob),
                JpegCompression(p=opt.aug_prob),
            ]),
            p=opt.aug_prob,
        ),
        Mixup(
            annot_file=os.path.join(opt.data_path, params.project_name, 'annotations/train.json'),
            image_folder=os.path.join(opt.data_path, params.project_name, params.train_set),
            image_ids=train_ids,
            transform=transforms.Compose([
                RandomSizedCrop((0.1, 1.0), input_sizes[opt.compound_coef], input_sizes[opt.compound_coef]),
                HorizontalFlip(p=opt.aug_prob),
                VerticalFlip(p=opt.aug_prob),
                RandomRotate(p=opt.aug_prob),
                RandomBrightnessContrast(p=opt.aug_prob),
                HueSaturationValue(p=opt.aug_prob),
                ToGray(p=0.005),
                MedianBlur(p=opt.aug_prob),
                GaussianNoise(p=opt.aug_prob),
                JpegCompression(p=opt.aug_prob),
            ]),
            p=opt.aug_prob,
        ),
        Normalizer(mean=params.mean, std=params.std),
        CustomToTensor(),
    ]) 

    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                               image_ids=train_ids,
                               set=params.train_set,
                               transform=augmentations)
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name),
                          image_ids=val_ids,
                          set=params.train_set,
                          transform=transforms.Compose([
                                                Resizer(input_sizes[opt.compound_coef]),
                                                Normalizer(mean=params.mean, std=params.std),
                                                CustomToTensor(),
                                                ]))
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300*opt.num_epochs) # iter per epoch * num_epoch
    scaler = GradScaler()

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()

                    with autocast():
                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue
                    scaler.scale(loss).backward()

                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
