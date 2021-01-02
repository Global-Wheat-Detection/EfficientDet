import os
import re
import json


annots = []
images = {}


def read_wheat(root):
    pattern = '(\w+),(\d+),(\d+),\"\[([.\d]+), ([.\d]+), ' \
              '([.\d]+), ([.\d]+)\]\".*'
    with open(root, 'r') as f:
        next(f)
        for line in f:
            m = re.search(pattern, line)
            annot = {}
            res = {}
            annot['file_name'] = m.group(1)
            res['width'] = int(m.group(2))
            res['height'] = int(m.group(3))
            annot['bbox'] = [round(float(x)) for x in m.group(4, 5, 6, 7)]

            # https://www.kaggle.com/raininbox/check-clean-big-small-bboxes
            if annot['bbox'][2]*annot['bbox'][3] < 300 or \
               annot['bbox'][0] < 0 or annot['bbox'][1] < 0 or \
               annot['bbox'][2] < 10 or annot['bbox'][3] < 10:
                continue

            if annot['file_name'] not in images:
                images[annot['file_name']] = res
            annots.append(annot)


def read_tsv(root):
    for tsv in os.listdir(root):
        name = tsv.split('.')[0]
        with open(os.path.join(root, tsv), 'r') as f:
            for line in f:
                bbox = [int(x) for x in line[:-1].split()]
                bbox[2] = bbox[2] - bbox[0]
                bbox[3] = bbox[3] - bbox[1]
                annot = {}
                annot['file_name'] = name
                annot['bbox'] = bbox

                # https://www.kaggle.com/raininbox/check-clean-big-small-bboxes
                if annot['bbox'][2]*annot['bbox'][3] < 300 or \
                   annot['bbox'][0] < 0 or annot['bbox'][1] < 0 or \
                   annot['bbox'][2] < 10 or annot['bbox'][3] < 10:
                    continue

                if annot['file_name'] not in images:
                    images[annot['file_name']] = {'width': None, 'height': None}
                annots.append(annot)


def read_csv(root):
    for csv in os.listdir(root):
        with open(os.path.join(root, csv), 'r') as f:
            next(f)
            for line in f:
                bbox_name = line.split(',')
                annot = {}
                annot['file_name'] = bbox_name[-1][:-1]
                annot['bbox'] = [int(round(float(x))) for x in bbox_name[:-1]]
                annot['bbox'][2] = annot['bbox'][2] - annot['bbox'][0]
                annot['bbox'][3] = annot['bbox'][3] - annot['bbox'][1]

                # https://www.kaggle.com/raininbox/check-clean-big-small-bboxes
                if annot['bbox'][2]*annot['bbox'][3] < 300 or \
                   annot['bbox'][0] < 0 or annot['bbox'][1] < 0 or \
                   annot['bbox'][2] < 10 or annot['bbox'][3] < 10:
                    continue

                if annot['file_name'] not in images:
                    images[annot['file_name']] = {'width': None, 'height': None}
                annots.append(annot)



def write_json():
    json_dict = {}

    # types
    json_dict['type'] = 'instances'

    # images
    json_dict['images'] = []
    name2id = {} # save for later use
    for idx, file_name in enumerate(images):
        info = {}
        info['file_name'] = file_name + '.jpg'
        info['height'] = images[file_name]['height']
        info['width'] = images[file_name]['width']
        info['id'] = idx
        json_dict['images'].append(info)
        name2id[file_name] = info['id']

    # categories
    json_dict['categories'] = []
    json_dict['categories'].append({
        'supercategory': 'none',
        'name': 'wheat',
        'id': 0,
    })

    # annotations
    json_dict['annotations'] = []
    for idx, annot in enumerate(annots):
        coco_annot = {}
        coco_annot['id'] = idx + 1
        coco_annot['bbox'] = annot['bbox']
        coco_annot['image_id'] = name2id[annot['file_name']]
        coco_annot['segmentation'] = []
        coco_annot['ignore'] = 0
        coco_annot['area'] = annot['bbox'][2]*annot['bbox'][3]
        coco_annot['iscrowd'] = 0
        coco_annot['category_id'] = 0
        json_dict['annotations'].append(coco_annot)

    root = 'datasets/global-wheat-detection/annotations'
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, 'train.json'), 'w') as f:
        json.dump(json_dict, f)


if __name__=='__main__':
    read_wheat('datasets/train.csv')
    read_tsv('datasets/bbox_tsv')
    read_csv('datasets/csv')

    write_json()
