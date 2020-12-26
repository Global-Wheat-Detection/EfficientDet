import os
import re
import json


def read_csv(anno_dir):
    pattern = '(\w+),(\d+),(\d+),\"\[([.\d]+), ([.\d]+), ' \
              '([.\d]+), ([.\d]+)\]\".*'
    annots = []
    images = {}
    with open(anno_dir, 'r') as f:
        next(f)
        for line in f:
            m = re.search(pattern, line)
            annot = {}
            res = {}
            annot['file_name'] = m.group(1)
            res['width'] = int(m.group(2))
            res['height'] = int(m.group(3))
            annot['bbox'] = [round(float(x)) for x in m.group(4, 5, 6, 7)]

            if annot['file_name'] not in images:
                images[annot['file_name']] = res
            annots.append(annot)

    return images, annots


def write_json(images, annots):
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
    input_path = 'datasets/global-wheat-detection/train.csv'
    images, annots = read_csv(input_path)
    write_json(images, annots)
