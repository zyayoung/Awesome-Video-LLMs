import argparse
import json
import os

import clip
# import numpy as np
import torch
# import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()
def cal_retrival_acc(file_name, output_name, clip_model='RN50', cut_length=77, top_k=(1, 5)):
    print('Loading CLIP')
    model, preprocess = clip.load(clip_model, device='cuda')
    with open(file_name, 'r') as f:
        result_pred = json.load(f)

    gt_set = set()
    for result_item in result_pred:
        gt_set.add(result_item['gt'][0])
    gt_set = list(gt_set)
    answer2id = {ans:idx for idx, ans in enumerate(gt_set)}

    print('Building GT Embedding List')
    gt_embbed_list = list()
    for gt in tqdm(gt_set):
        gt = clip.tokenize(gt).cuda()
        gt_feats = model.encode_text(gt)
        gt_embbed_list.append(gt_feats)
    gt_embbed_list = torch.cat(gt_embbed_list, dim=0)

    print('Getting Accuracy')
    acc_count = [0 for _ in top_k]
    for result_item in tqdm(result_pred):
        gt = result_item['gt'][0]
        pred = result_item['answer'].replace('.', '')
        if len(pred) > cut_length:
            pred = pred[:cut_length]
        
        pred = clip.tokenize(pred).cuda()
        pred_feats = model.encode_text(pred)
        acc_list = torch.cosine_similarity(pred_feats, gt_embbed_list)
        _, c_idx = torch.topk(acc_list, max(top_k), dim=0)
        for idx, k in enumerate(top_k):
            correct_k = c_idx[:k].eq(answer2id[gt]).view(-1).float().sum().item()
            acc_count[idx] += correct_k

    final_acc = [_ / len(result_pred) for _ in acc_count]
    json_output = {}
    for idx, k in enumerate(top_k):
        print("The Top-{} Accuracy of retrival is {}, {}/{}.".format(k, final_acc[idx], int(acc_count[idx]), len(result_pred)))
        json_output['retrival_accuracy_top{}'.format(k)] = final_acc[idx]
    with open(output_name, 'w') as f:
        json.dump(json_output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default='/data/projects/code_yc/mmgpt/checkpoints/msvd_eval_results/')
    parser.add_argument('--exp_name', type=str, default='webvid_0711_03ep_vtype-v2_qadata_testval_ckpt2000_gpu8')
    args = parser.parse_args()

    exp_folder = os.path.join(args.exp_root, args.exp_name)
    file_name = os.path.join(exp_folder, 'results_final.json')
    output_name = os.path.join(exp_folder, 'clip_retrival_accuracy.json')
    cal_retrival_acc(file_name, output_name)
