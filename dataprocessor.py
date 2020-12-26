import pandas as pd
import cv2
import numpy as np
import random


class Mosaic(object):
    '''Apply Mosaic augmentation to passed in image.'''
    def __init__(self, p=1):
        self.proba = p
        self.annot_file = 'global-wheat-detection/train.csv'
        self.image_folder = 'global-wheat-detection/train/'
        self.annot = pd.read_csv(self.annot_file)
        self.image_ids = np.unique(self.annot['image_id'].to_numpy())

    def get_cut(self, image, bboxs, cutx, cuty):
        old_height = image.shape[0]
        old_width = image.shape[1]
        # image
        new_img = image[old_height - cuty:, old_width - cutx:, :]

        # bbox
        extract_bbox = np.zeros((0, 5))
        for bbox in bboxs:
            new_bbox = np.zeros((1, 5))
            if bbox[0] >= old_width - cutx and bbox[1] >= old_height - cuty:
                new_bbox[0, 0] = bbox[0] - (old_width - cutx)
                new_bbox[0, 1] = bbox[1] - (old_height - cuty)
                new_bbox[0, 2] = bbox[2]
                new_bbox[0, 3] = bbox[3]
                new_bbox[0, 4] = bbox[4]

            elif bbox[0] >= old_width - cutx:
                if bbox[1] + bbox[3] > old_height - cuty:
                    new_bbox[0, 0] = bbox[0] - (old_width - cutx)
                    new_bbox[0, 1] = 0
                    new_bbox[0, 2] = bbox[2]
                    new_bbox[0, 3] = bbox[1] + bbox[3] - (old_height - cuty)
                    new_bbox[0, 4] = bbox[4]
                    if new_bbox[0, 2] < 15 or new_bbox[0, 3] < 15:
                        continue
                else:
                    continue

            elif bbox[1] >= old_height - cuty:
                if bbox[0] + bbox[2] > old_width - cutx:
                    new_bbox[0, 0] = 0
                    new_bbox[0, 1] = bbox[1] - (old_height - cuty)
                    new_bbox[0, 2] = bbox[0] + bbox[2] - (old_width - cutx)
                    new_bbox[0, 3] = bbox[3]
                    new_bbox[0, 4] = bbox[4]
                    if new_bbox[0, 2] < 15 or new_bbox[0, 3] < 15:
                        continue
                else:
                    continue

            else:
                if bbox[1] + bbox[3] > old_height - cuty \
                        and bbox[0] + bbox[2] > old_width - cutx:
                    new_bbox[0, 0] = 0
                    new_bbox[0, 1] = 0
                    new_bbox[0, 2] = bbox[0] + bbox[2] - (old_width - cutx)
                    new_bbox[0, 3] = bbox[1] + bbox[3] - (old_height - cuty)
                    new_bbox[0, 4] = bbox[4]
                    if new_bbox[0, 2] < 15 or new_bbox[0, 3] < 15:
                        continue
                else:
                    continue

            extract_bbox = np.concatenate((extract_bbox, new_bbox), axis=0)

        return new_img, extract_bbox

    def __call__(self, sample):
        if random.uniform(0, 1) <= self.proba:
            input_img = sample['img']
            input_bbx = sample['annot']

            # prepare the other 3 images
            np.random.shuffle(self.image_ids)
            mosaic_ids = [self.image_ids[i] for i in range(3)]
            mosaic_imgs = [input_img]
            mosaic_bboxs = [input_bbx]

            for mosaic_id in mosaic_ids:

                # image
                path = self.image_folder + mosaic_id + '.jpg'
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255
                mosaic_imgs.append(image)

                # bbox
                raw_annot = self.annot.loc[self.annot['image_id'] == id]
                processed_annot = np.zeros((0, 5))

                for i in range(len(raw_annot.index)):
                    bbox = raw_annot.iloc[i, :]['bbox']
                    bbox = bbox.strip('[').strip(']').split(', ')
                    for i, ele in enumerate(bbox):  # [left, top, width, height]
                        bbox[i] = float(ele)
                    bbox.append(0)  # class 0
                    bbox = np.expand_dims(np.array(bbox), axis=0)
                    processed_annot = np.concatenate((processed_annot, bbox), axis=0)

                mosaic_bboxs.append(processed_annot)

            # do augmentation
            x_length = input_img.shape[1]
            y_length = input_img.shape[0]
            x_ratio = int(random.uniform(0.2, 0.8) * x_length)
            y_ratio = int(random.uniform(0.2, 0.8) * y_length)

            merge_bboxs = np.zeros((0, 5))
            new_img = np.zeros((y_length, x_length, 3))  # [H, W, C]

            for i in range(4):
                ori_width = mosaic_imgs[i].shape[1]
                ori_height = mosaic_imgs[i].shape[0]
                if i == 0:  # top left
                    cut_img, cut_bbox = self.get_cut(mosaic_imgs[i], mosaic_bboxs[i], x_ratio, y_ratio)
                    new_img[:cut_img.shape[0], :cut_img.shape[1], :] = cut_img
                    merge_bboxs = np.concatenate((merge_bboxs, cut_bbox), axis=0)

                elif i == 1:  # top right
                    flipped_img = mosaic_imgs[i][:, ::-1, :]  # hortizontal flip
                    flipped_bboxs = np.zeros((0, 5))
                    for bbox in mosaic_bboxs[i]:
                        flipped_bbox = np.zeros((1, 5))
                        flipped_bbox[0, 0] = ori_width - bbox[0] - bbox[2]
                        flipped_bbox[0, 1:] = bbox[1:]
                        flipped_bboxs = np.concatenate((flipped_bboxs, flipped_bbox),
                                                       axis=0)

                    cut_img, cut_bbox = self.get_cut(flipped_img, flipped_bboxs,
                                                ori_width - x_ratio, y_ratio)

                    cut_img = cut_img[:, ::-1, :]  # flip back
                    flipped_bboxs = np.zeros((0, 5))
                    for bbox in cut_bbox:
                        flipped_bbox = np.zeros((1, 5))
                        flipped_bbox[0, 0] = cut_img.shape[1] - bbox[0] - bbox[2] + x_ratio
                        flipped_bbox[0, 1:] = bbox[1:]
                        flipped_bboxs = np.concatenate((flipped_bboxs, flipped_bbox),
                                                       axis=0)

                    new_img[:cut_img.shape[0], x_length - cut_img.shape[1]:, :] = cut_img
                    merge_bboxs = np.concatenate((merge_bboxs, flipped_bboxs), axis=0)

                elif i == 2:  # bot left
                    flipped_img = mosaic_imgs[i][::-1, :, :]  # vertical flip
                    flipped_bboxs = np.zeros((0, 5))
                    for bbox in mosaic_bboxs[i]:
                        flipped_bbox = np.zeros((1, 5))
                        flipped_bbox[0, 0] = bbox[0]
                        flipped_bbox[0, 1] = ori_height - bbox[1] - bbox[3]
                        flipped_bbox[0, 2:] = bbox[2:]
                        flipped_bboxs = np.concatenate((flipped_bboxs, flipped_bbox),
                                                       axis=0)

                    cut_img, cut_bbox = self.get_cut(flipped_img, flipped_bboxs,
                                                x_ratio, ori_height - y_ratio)

                    cut_img = cut_img[::-1, :, :]  # flip back
                    flipped_bboxs = np.zeros((0, 5))
                    for bbox in cut_bbox:
                        flipped_bbox = np.zeros((1, 5))
                        flipped_bbox[0, 0] = bbox[0]
                        flipped_bbox[0, 1] = cut_img.shape[0] - bbox[1] - bbox[3] + y_ratio
                        flipped_bbox[0, 2:] = bbox[2:]
                        flipped_bboxs = np.concatenate((flipped_bboxs, flipped_bbox),
                                                       axis=0)

                    new_img[y_length - cut_img.shape[0]:, :cut_img.shape[1], :] = cut_img
                    merge_bboxs = np.concatenate((merge_bboxs, flipped_bboxs), axis=0)

                elif i == 3:  # bot right
                    flipped_img = mosaic_imgs[i][::-1, ::-1, :]  # vertical and horizontal flip
                    flipped_bboxs = np.zeros((0, 5))
                    for bbox in mosaic_bboxs[i]:
                        flipped_bbox = np.zeros((1, 5))
                        flipped_bbox[0, 0] = ori_width - bbox[0] - bbox[2]
                        flipped_bbox[0, 1] = ori_height - bbox[1] - bbox[3]
                        flipped_bbox[0, 2:] = bbox[2:]
                        flipped_bboxs = np.concatenate((flipped_bboxs, flipped_bbox),
                                                       axis=0)

                    cut_img, cut_bbox = self.get_cut(flipped_img, flipped_bboxs,
                                                ori_width - x_ratio, ori_height - y_ratio)

                    cut_img = cut_img[::-1, ::-1, :]  # flip back
                    flipped_bboxs = np.zeros((0, 5))
                    for bbox in cut_bbox:
                        flipped_bbox = np.zeros((1, 5))
                        flipped_bbox[0, 0] = cut_img.shape[1] - bbox[0] - bbox[2] + x_ratio
                        flipped_bbox[0, 1] = cut_img.shape[0] - bbox[1] - bbox[3] + y_ratio
                        flipped_bbox[0, 2:] = bbox[2:]
                        flipped_bboxs = np.concatenate((flipped_bboxs, flipped_bbox),
                                                       axis=0)

                    new_img[y_length - cut_img.shape[0]:, x_length - cut_img.shape[1]:, :] = cut_img
                    merge_bboxs = np.concatenate((merge_bboxs, flipped_bboxs), axis=0)

            sample = {'img': new_img, 'annot': merge_bboxs}

        return sample


class Mixup(object):
    '''Mix passed in image with another.'''
    def __init__(self, p=1):
        self.proba = p
        self.annot_file = 'global-wheat-detection/train.csv'
        self.image_folder = 'global-wheat-detection/train/'
        self.annot = pd.read_csv(self.annot_file)
        self.image_ids = np.unique(self.annot['image_id'].to_numpy())

    def __call__(self, sample):
        if random.uniform(0, 1) <= self.proba:
            input_img = sample['img']
            input_bbx = sample['annot']

            # prepare the other image
            np.random.shuffle(self.image_ids)
            mixup_id = self.image_ids[0]

            mixup_img = cv2.imread(self.image_folder + mixup_id + '.jpg')
            mixup_img = cv2.cvtColor(mixup_img, cv2.COLOR_BGR2RGB)
            mixup_img = mixup_img.astype(np.float32) / 255
            add_annotation = np.zeros((0, 5))

            annot_input = self.annot.loc[self.annot['image_id'] == mixup_id]
            for i in range(len(annot_input.index)):
                bbox = annot_input.iloc[i, :]['bbox']
                bbox = bbox.strip('[').strip(']').split(', ')
                for i, ele in enumerate(bbox):  # [left, top, width, height]
                    bbox[i] = float(ele)
                bbox.append(0)
                bbox = np.expand_dims(np.array(bbox), axis=0)
                add_annotation = np.concatenate((add_annotation, bbox), axis=0)

            # do mixup augmentation
            mixup_ratio = random.uniform(0.35, 0.65)
            mixup_img = mixup_ratio * input_img + (1 - mixup_ratio) * mixup_img
            mixup_annotation = np.concatenate((input_bbx, add_annotation), axis=0)

            sample = {'img':mixup_img, 'annot':mixup_annotation}

        return sample