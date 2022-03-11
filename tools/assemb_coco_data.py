# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   assemb_coco_data.py
# @Time   :   2021/10/22 15:04

from os import path
import os
import json
import torch


def format_coco_infer_res(custom_infer_out_root):
    """
    调整格式
    Args:
        custom_infer_out_root:

    Returns:

    """
    print('processing {}'.format(custom_infer_out_root))
    info = json.load(open(os.path.join(custom_infer_out_root, 'custom_data_info.json')))

    data_info = {}
    data_info['img_idx'] = {path.basename(info['idx_to_files'][i]): i for i in range(len(info['idx_to_files']))}
    data_info['ind_to_classes'] = info['ind_to_classes']
    data_info['ind_to_predicates'] = info['ind_to_predicates']

    with open(os.path.join(custom_infer_out_root, 'coco_sgg_info.json'), 'w') as f:
        json.dump(data_info, f)

    predictions = torch.load(path.join(custom_infer_out_root, 'custom_prediction.pth'))
    # json.load(open(os.path.join(custom_infer_out_root, 'custom_prediction.json')))
    # pred_data = [predictions[str(i)] for i in range(len(info['idx_to_files']))]
    #
    # torch.save(pred_data, os.path.join(custom_infer_out_root, 'coco_sgg_data.pth'))

    return data_info, predictions


def assemble_coco_data(infer_root):
    datasets = ['train2014', 'val2014', 'test2015']
    total_map = {}
    total_data = []
    for dataset in datasets:
        map, pred = format_coco_infer_res(path.join(infer_root, dataset))
        if not total_map:
            total_map = map
        else:
            start_idx = len(total_data)
            map['img_idx'] = {k: v + start_idx for k, v in map['img_idx'].items()}
            total_map['img_idx'].update(map['img_idx'])
        total_data.extend(pred)

    info_file = path.join(infer_root, 'coco_sgg_data_info.json')
    json.dump(total_map, open(info_file, 'w+'))
    print("File {} saved!".format(info_file))

    data_file = path.join(infer_root, 'coco_sgg_data.pth')
    torch.save(total_data, data_file)
    print("File {} saved!".format(data_file))


if __name__ == '__main__':
    infer_root = '/public/data1/users/chenchao278/model/sgdet/gt_base/coco_infer'
    assemble_coco_data(infer_root)
