# Modified

import re
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class WheatDataset(Dataset):
    def __init__(self, root_dir, anno_dir, image_ids, transform=None):

        self.root_dir = root_dir
        self.image_ids = image_ids
        self.annots = self.read_csv(anno_dir)
        self.transform = transform

    def read_csv(self, anno_dir):
        pattern = '(\w+).*\"\[([.\d]+), ([.\d]+), ' \
                '([.\d]+), ([.\d]+)\]\".*'
        annots = {}
        with open(anno_dir, 'r') as f:
            next(f)
            for line in f:
                m = re.search(pattern, line)
                image_id = m.group(1)
                bbox = [float(x) for x in m.group(2, 3, 4, 5)]
                bbox.append(0)  # class label = 0
                if image_id in self.image_ids:
                    if image_id not in annots:
                        annots[image_id] = [np.array(bbox)]
                    else:
                        annots[image_id].append(np.array(bbox))
        for id in self.image_ids:
            if id in annots:
                annots[id] = np.stack(annots[id])
            else:
                annots[id] = np.zeros((0, 5))

        return annots

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(self.image_ids[idx])
        annot = self.annots[self.image_ids[idx]]
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_id):
        path = os.path.join(self.root_dir, image_id + '.jpg')
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
