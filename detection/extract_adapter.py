# -*- encoding: utf-8 -*-
'''
@File       :   extract_adapter.py   
@Desciption :   从checkpoint提取adapter
 
@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
2023/9/6 15:22   Xue Zongyao      1.0         None
'''
import numpy as np
import torch
from collections import OrderedDict
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('extract_adapter', add_help=False)
    parser.add_argument('--full_checkpoint', default=r'C:\Users\15339\Downloads\mask_rcnn_mae_adapter_base_lsj_fpn_25ep_coco.pth.tar', type=str)  # 完整checkpoint路径
    parser.add_argument('--adapter', default=r'D:\data\mae_adapter_base_lsj_fpn_25ep_coco.pth', type=str)  # 提取的adapter路径
    return parser


def main(args):
    full_checkpoint_path = args.full_checkpoint
    adapter_path = args.adapter
    check_point = torch.load(full_checkpoint_path,map_location='cpu')['state_dict']
    adapter = OrderedDict()
    filter_list = ['backbone.pos_embed','backbone.patch_embed','backbone.blocks','roi_head']

    for k,v in check_point.items():
        filtered = False
        for filter in filter_list:
            if k.startswith(filter):
                filtered=True
                break

        if filtered:
            continue

        adapter[k] = v

    adapter = {'state_dict':adapter}
    torch.save(adapter,adapter_path)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)