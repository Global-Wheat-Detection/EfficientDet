import os
import cv2
import numpy as np
import random
import json
from pycocotools.coco import COCO
import albumentations as A


def ltwh2ltrb(bboxs):
    if bboxs.ndim == 2:
        bboxs[:, 2] = bboxs[:, 2] + bboxs[:, 0]
        bboxs[:, 3] = bboxs[:, 3] + bboxs[:, 1]
    elif bboxs.ndim == 1:
        if bboxs.shape[0] != 0:
            bboxs[2] = bboxs[2] + bboxs[0]
            bboxs[3] = bboxs[3] + bboxs[1]

    return bboxs


def ltrb2ltwh(bboxs):
    if bboxs.ndim == 2:
        bboxs[:, 2] = bboxs[:, 2] - bboxs[:, 0]
        bboxs[:, 3] = bboxs[:, 3] - bboxs[:, 1]
    elif bboxs.ndim == 1:
        if bboxs.shape[0] != 0:
            bboxs[2] = bboxs[2] - bboxs[0]
            bboxs[3] = bboxs[3] - bboxs[1]

    return bboxs


def concate_bbox(old, new):
    if new.ndim == 2:
        old = np.concatenate((old, new), axis=0)
    else:
        if new.shape[0] != 0:
            new = np.expand_dims(new, axis=0)
            old = np.concatenate((old, new), axis=0)

    return old


class Mosaic(object):

    def __init__(self, annot_file, image_folder, image_ids=None, transform=None, p=1):
        '''
        Apply Mosaic augmentation to passed in image.
        :param image_ids: ids of training images
        :param annot_file: file path of annotation file
        :param image_folder: folder path of images
        :param transform: transform to apply to other images
        :param p: probability of applying mosaic
        '''
        self.proba = p
        self.image_folder = image_folder
        self.annot = COCO(annot_file)
        if image_ids:
            self.image_ids = image_ids
        else:
            self.image_ids = list(self.annot.imgs.keys())
        self.transform = transform

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

            input_bbx = ltrb2ltwh(input_bbx)

            # prepare the other 3 images
            choice = np.random.choice(len(self.image_ids), 3, replace=False)
            mosaic_ids = [self.image_ids[i] for i in choice]
            mosaic_imgs = [input_img]
            mosaic_bboxs = [input_bbx]

            for mosaic_id in mosaic_ids:
                img_info = self.annot.loadImgs(mosaic_id)

                # image
                path = os.path.join(self.image_folder, img_info[0]['file_name'])
                image = cv2.imread(path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255
                mosaic_imgs.append(image)

                # bbox
                annids = self.annot.getAnnIds(imgIds=mosaic_id)
                anns = self.annot.loadAnns(annids)

                processed_annot = np.zeros((0, 5))

                for instance in anns:
                    bbox = instance['bbox']
                    if len(bbox) == 4:
                        bbox.append(0)  # class 0
                    bbox = np.expand_dims(np.array(bbox), axis=0)
                    processed_annot = np.concatenate((processed_annot, bbox), axis=0)

                processed_annot = ltwh2ltrb(processed_annot)

                mosaic_bboxs.append(processed_annot)

            if self.transform:
                for i in [1, 2, 3]:
                    transformed = self.transform({'img': mosaic_imgs[i], 'annot': mosaic_bboxs[i]})
                    mosaic_imgs[i] = transformed['img']
                    mosaic_bboxs[i] = transformed['annot']

            for i in [1, 2, 3]:
                mosaic_bboxs[i] = ltrb2ltwh(mosaic_bboxs[i])

            # do augmentation
            x_length = input_img.shape[1]
            y_length = input_img.shape[0]
            x_ratio = int(random.uniform(0.2, 0.8) * x_length)
            y_ratio = int(random.uniform(0.2, 0.8) * y_length)

            merge_bboxs = np.zeros((0, 5))
            new_img = np.zeros((y_length, x_length, 3))  # [H, W, C]

            for i in range(4):          # cut and paste images
                ori_width = mosaic_imgs[i].shape[1]
                ori_height = mosaic_imgs[i].shape[0]
                if i == 0:  # top left
                    cut_img, cut_bbox = self.get_cut(mosaic_imgs[i], mosaic_bboxs[i], x_ratio, y_ratio)
                    new_img[:cut_img.shape[0], :cut_img.shape[1], :] = cut_img
                    merge_bboxs = concate_bbox(merge_bboxs, cut_bbox)

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
                    merge_bboxs = concate_bbox(merge_bboxs, flipped_bboxs)

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
                    merge_bboxs = concate_bbox(merge_bboxs, flipped_bboxs)

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
                    merge_bboxs = concate_bbox(merge_bboxs, flipped_bboxs)

            merge_bboxs = ltwh2ltrb(merge_bboxs)

            sample['img'] = new_img
            sample['annot'] = merge_bboxs

        return sample


class Mixup(object):

    def __init__(self, annot_file, image_folder, image_ids=None, transform=None, p=1.0):
        '''
        Mix passed in image with another.
        :param image_ids: ids of training images
        :param annot_file: file path of annotation file
        :param image_folder: folder path of images
        :param transform: transform to apply to the other image
        :param p: probability of applying mixup
        '''
        self.proba = p
        self.image_folder = image_folder
        self.annot = COCO(annot_file)
        if image_ids:
            self.image_ids = image_ids
        else:
            self.image_ids = list(self.annot.imgs.keys())
        self.transform = transform

    def __call__(self, sample):
        if random.uniform(0, 1) <= self.proba:
            input_img = sample['img']
            input_bbx = sample['annot']

            input_bbx = ltrb2ltwh(input_bbx)

            # prepare the other image
            choice = int(np.random.choice(len(self.image_ids), 1, replace=False))
            mixup_id = self.image_ids[choice]
            img_info = self.annot.loadImgs(mixup_id)

            # image
            mixup_img = cv2.imread(os.path.join(self.image_folder, img_info[0]['file_name']))
            mixup_img = cv2.cvtColor(mixup_img, cv2.COLOR_BGR2RGB)
            mixup_img = mixup_img.astype(np.float32) / 255
            add_annotation = np.zeros((0, 5))

            # bbox
            annids = self.annot.getAnnIds(imgIds=mixup_id)
            anns = self.annot.loadAnns(annids)

            for instance in anns:
                bbox = instance['bbox']
                if len(bbox) == 4:
                    bbox.append(0)  # class 0
                bbox = np.expand_dims(np.array(bbox), axis=0)
                add_annotation = np.concatenate((add_annotation, bbox), axis=0)

            add_annotation = ltwh2ltrb(add_annotation)

            if self.transform:
                transformed = self.transform({'img': mixup_img, 'annot': add_annotation})
                mixup_img = transformed['img']
                add_annotation = transformed['annot']

            add_annotation = ltrb2ltwh(add_annotation)

            # do mixup augmentation
            mixup_ratio = random.uniform(0.35, 0.65)
            mixup_img = mixup_ratio * input_img + (1 - mixup_ratio) * mixup_img

            mixup_annotation = concate_bbox(input_bbx, add_annotation)

            mixup_annotation = ltwh2ltrb(mixup_annotation)

            sample['img'] = mixup_img
            sample['annot'] = mixup_annotation

        return sample


class GaussianBlur(object):

    def __init__(self, kernel_szie=(35, 35), p=1.0):
        '''
        Apply Gaussian blur.
        :param kernel_szie: kernel size of blur. tuple
        :param p: probability of applying blur
        '''
        self.kernel_size = kernel_szie
        self.proba = p

    def __call__(self, sample):
        if random.uniform(0, 1) <= self.proba:
            input_img = sample['img']
            input_bbx = sample['annot']

            image_blur = cv2.GaussianBlur(input_img, self.kernel_size, 0)

            sample['img'] = image_blur
            sample['annot'] = input_bbx

        return sample


class GaussianNoise(object):

    def __init__(self, ratio_min=0, ratio_max=0.5, p=1.0):
        '''
        Add Gaussian noise.
        :param ratio_min: minimum ratio of noise, between 0 and 1
        :param ratio_max: maximum ratio of noise, between 0 and 1
        :param p: probability of adding noise
        '''
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.proba = p

    def __call__(self, sample):
        if random.uniform(0, 1) <= self.proba:
            input_img = sample['img']
            input_bbx = sample['annot']

            ratio = random.uniform(self.ratio_min, self.ratio_max)

            image_noise = (1 - ratio) * input_img +\
                          ratio * np.random.normal(0, 0.5, input_img.shape)
            image_noise = np.clip(image_noise, 0.0, 1.0)

            sample['img'] = image_noise
            sample['annot'] = input_bbx

        return sample


class RandomRotate(object):

    def __init__(self, p=1.0):
        '''
        Rotate the image 90 degrees clockwise or counterclockwise, both 50%.
        :param p: probability to rotate image
        '''
        self.proba = p

    def rotate_img(self, image, angle, center=None, scale=1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    def __call__(self, sample):
        if random.uniform(0, 1) <= self.proba:
            input_img = sample['img']
            input_bbx = sample['annot']

            input_bbx = ltrb2ltwh(input_bbx)

            counterclockwise = np.random.randint(2)

            # image
            if counterclockwise:
                rotated_img = self.rotate_img(input_img, 90)
            else:
                rotated_img = self.rotate_img(input_img, -90)

            # bbox
            rotated_bboxs = np.zeros((0, 5))

            width = input_img.shape[1]
            height = input_img.shape[0]
            for bbox in input_bbx:
                rotated_bbox = np.zeros((1, 5))
                if counterclockwise:
                    rotated_bbox[0, 0] = bbox[1]
                    rotated_bbox[0, 1] = width - bbox[0] - bbox[2]
                else:
                    rotated_bbox[0, 0] = height - bbox[1] - bbox[3]
                    rotated_bbox[0, 1] = bbox[0]
                rotated_bbox[0, 2] = bbox[3]
                rotated_bbox[0, 3] = bbox[2]
                rotated_bbox[0, 4] = 0
                rotated_bboxs = np.concatenate((rotated_bboxs, rotated_bbox), axis=0)

            rotated_bboxs = ltwh2ltrb(rotated_bboxs)
            sample['img'] = rotated_img
            sample['annot'] = rotated_bboxs

        return sample


class HorizontalFlip(object):

    def __init__(self, p=1.0):
        '''
        Flip image horizontally.
        :param p: probability to flip
        '''
        self.transform = A.Compose(
            [A.HorizontalFlip(p=p)],
            bbox_params=A.BboxParams(format='coco'),
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        input_bbx = ltrb2ltwh(input_bbx)

        transformed = self.transform(image=input_img, bboxes=input_bbx)

        out_img = transformed['image']
        out_bboxs = np.array(transformed['bboxes'])

        out_bboxs = ltwh2ltrb(out_bboxs)
        sample['img'] = out_img
        sample['annot'] = out_bboxs

        return sample


class VerticalFlip(object):

    def __init__(self, p=1.0):
        '''
        Flip image vertically.
        :param p: probability to flip
        '''
        self.transform = A.Compose(
            [A.VerticalFlip(p=p)],
            bbox_params=A.BboxParams(format='coco'),
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        input_bbx = ltrb2ltwh(input_bbx)

        transformed = self.transform(image=input_img, bboxes=input_bbx)

        out_img = transformed['image']
        out_bboxs = np.array(transformed['bboxes'])

        out_bboxs = ltwh2ltrb(out_bboxs)

        sample['img'] = out_img
        sample['annot'] = out_bboxs

        return sample


class JpegCompression(object):

    def __init__(self, quality_lower=99, quality_upper=100, p=1):
        '''
        Apply JEPG compression.
        :param quality_lower: quality upper bound
        :param quality_upper: quality lower bound
        :param p: probability of applying transformation
        '''
        self.transform = A.Compose(
            [A.JpegCompression(quality_lower=quality_lower,
                               quality_upper=quality_upper, p=p)],
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_img = (input_img * 255).astype(np.uint8)
        input_bbx = sample['annot']

        transformed = self.transform(image=input_img)

        out_img = transformed['image'] / 255

        sample['img'] = out_img
        sample['annot'] = input_bbx

        return sample


class MedianBlur(object):

    def __init__(self, blur_limit=11, p=1):
        '''
        Apply MedianBlur blur.
        :param p: probability of applying transformation
        '''
        self.p = p
        self.transform = A.Compose(
            [A.MedianBlur(blur_limit=blur_limit, p=p)],
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_img = (input_img * 255).astype(np.uint8)
        input_bbx = sample['annot']

        transformed = self.transform(image=input_img)

        out_img = transformed['image'] / 255

        sample['img'] = out_img
        sample['annot'] = input_bbx

        return sample


class RandomCrop(object):

    def __init__(self, height, width, p=1):
        '''
        Randomly crop from an image.
        :param height: height of cropped image
        :param width: width of cropped image
        :param p: probability of applying transformation
        '''
        self.transform = A.Compose(
            [A.RandomCrop(height, width, p=p)],
            bbox_params=A.BboxParams(format='coco'),
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        input_bbx = ltrb2ltwh(input_bbx)

        transformed = self.transform(image=input_img, bboxes=input_bbx)

        out_img = transformed['image']
        out_bboxs = transformed['bboxes']
        out_bboxs = np.array(out_bboxs)

        if out_bboxs.ndim == 1:
            if out_bboxs.shape[0] != 0:
                out_bboxs = np.expand_dims(out_bboxs, axis=0)
            else:
                out_bboxs = np.zeros((0, 5))

        out_bboxs = ltwh2ltrb(out_bboxs)

        sample['img'] = out_img
        sample['annot'] = out_bboxs

        return sample


class ToGray(object):

    def __init__(self, p=1):
        '''
        Transform image to grayscale.
        :param p: probability of applying transformation
        '''
        self.transform = A.Compose(
            [A.ToGray(p=p)],
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        input_img = (input_img * 255).astype(np.uint8)
        transformed = self.transform(image=input_img)

        out_img = transformed['image'] / 255

        sample['img'] = out_img
        sample['annot'] = input_bbx

        return sample


class HueSaturationValue(object):

    def __init__(self, hue_shift_limit=20, sat_shift_limit=30,
                 val_shift_limit=20, p=1):
        '''
        Randomly change hue, saturation and value of the input image.
        :param hue_shift_limit: range for changing hue.
            If hue_shift_limit is a single int, the range will be
            (-hue_shift_limit, hue_shift_limit). Default: (-20, 20).
        :param sat_shift_limit: range for changing saturation.
            If sat_shift_limit is a single int, the range will be
            (-sat_shift_limit, sat_shift_limit). Default: (-30, 30).
        :param val_shift_limit: range for changing value.
            If val_shift_limit is a single int, the range will be
            (-val_shift_limit, val_shift_limit). Default: (-20, 20).
        :param p: probability of applying transformation
        '''
        self.transform = A.Compose(
            [A.HueSaturationValue(hue_shift_limit, sat_shift_limit,
                                  val_shift_limit, p=p)],
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        input_img = (input_img * 255).astype(np.uint8)
        transformed = self.transform(image=input_img)

        out_img = transformed['image'] / 255

        sample['img'] = out_img
        sample['annot'] = input_bbx

        return sample


class RandomBrightnessContrast(object):

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2,
                 brightness_by_max=True, p=1):
        '''
        Randomly change brightness and contrast of the input image.
        :param brightness_limit: factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit).
            Default: (-0.2, 0.2).
        :param contrast_limit: factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit).
            Default: (-0.2, 0.2).
        :param brightness_by_max: If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        :param p: probability of applying transformation
        '''
        self.transform = A.Compose(
            [A.RandomBrightnessContrast(brightness_limit, contrast_limit,
                                  brightness_by_max, p=p)],
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        input_img = (input_img * 255).astype(np.uint8)
        transformed = self.transform(image=input_img)

        out_img = transformed['image'] / 255

        sample['img'] = out_img
        sample['annot'] = input_bbx

        return sample


class Sharpen(object):
    # Unsolved:
    # AttributeError: module 'albumentations' has no attribute 'Sharpen'

    def __init__(self):
        '''
        Sharpen the image and overlays the result with the original image.
        :param p: probability to apply transformation
        '''
        self.transform = A.Compose(
            [A.Sharpen(alpha=(0.2, 0.5),
            lightness=(0.5, 1.0), always_apply=False, p=1.0)] ,
            bbox_params=A.BboxParams(format='coco'),
        )

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        transformed = self.transform(image=input_img, bboxes=input_bbx)

        out_img = transformed['image']
        out_bboxs = np.array(transformed['bboxes'])

        sample['img'] = out_img
        sample['annot'] = out_bboxs

        return sample


class RandomSizedCrop(object):

    def __init__(self, min_max_height, height, width, w2h_ratio=1.0, p=1):
        '''
        Crop a random part of the input and rescale it to some size.
        :param min_max_height: crop size limits. tuple of float (0 to 1).
        :param height: height after crop and resize.
        :param width: width after crop and resize.
        :param w2h_ratio: aspect ratio of crop. float.
        :param p: probability of applying transformation.
        '''
        self.min_max_height = min_max_height
        self.height = height
        self.width = width
        self.w2h_ratio = w2h_ratio
        self.p = p

    def __call__(self, sample):
        input_img = sample['img']
        input_bbx = sample['annot']

        input_bbx = ltrb2ltwh(input_bbx)

        min_max_height = (int(self.min_max_height[0] * input_img.shape[0]),
                          int(self.min_max_height[1] * input_img.shape[0]))
        transform = A.Compose(
            [A.RandomSizedCrop(min_max_height=min_max_height,
                               height=self.height, width=self.width,
                               w2h_ratio=self.w2h_ratio, p=self.p)],
            bbox_params=A.BboxParams(format='coco'),
        )
        transformed = transform(image=input_img, bboxes=input_bbx)

        for i, bbox in enumerate(transformed['bboxes']):
            if bbox[2] < 15 or bbox[3] < 15:
                transformed['bboxes'].pop(i)

        out_img = transformed['image']
        out_bboxs = transformed['bboxes']
        out_bboxs = np.array(out_bboxs)

        if out_bboxs.ndim == 1:
            if out_bboxs.shape[0] != 0:
                out_bboxs = np.expand_dims(out_bboxs, axis=0)
            else:
                out_bboxs = np.zeros((0, 5))

        out_bboxs = ltwh2ltrb(out_bboxs)

        sample['img'] = out_img
        sample['annot'] = out_bboxs

        return sample