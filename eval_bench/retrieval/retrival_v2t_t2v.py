import argparse
import json
import os

import clip
# import numpy as np
import torch
# import torch.nn.functional as F
from tqdm import tqdm
from refile import smart_open
import sys

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@torch.no_grad()
def cal_retrival_acc(file_name, output_name, clip_model='RN50', gt_path='', cut_length=77, top_k=(1, 5)):
    '''
    Calculate v2t, t2v
    1. embedding of gt, pred
    2. first calculate v2t: pred to all gt; then t2v: gt to all pred
    3. print and output all results
    '''

    logging.info('Loading CLIP')
    model, preprocess = clip.load(clip_model, device='cuda')
    with smart_open(file_name, 'r') as f:
        result_pred = json.load(f)
    # optional: gt from gt_path
    if gt_path != '':
        with smart_open(gt_path, 'r') as f:
            gt_data = json.load(f)
        vid2gt = {}
        for item in gt_data['annotations']:
            vid2gt[item['video_id']] = item['answer']
        # update pred-gt pair
        for ith, item in enumerate(result_pred):
            try:
                result_pred[ith]['gt'] = vid2gt[item['video_id']]
            except:
                import ipdb; ipdb.set_trace()
        logging.info('Loading gt from gt file, replacing original gt in pred file')

    # gt set & gt embed
    logging.info('Building GT Embedding List')
    gt_set = set()  # use set to avoid duplicate items
    for result_item in result_pred:
        gt_set.add(result_item['gt'][0])
    gt_set = list(gt_set)
    answer2id = {ans:idx for idx, ans in enumerate(gt_set)}

    gt_embbed_list = list()
    for gt in tqdm(gt_set):
        gt = clip.tokenize(gt).cuda()
        gt_feats = model.encode_text(gt)
        gt_embbed_list.append(gt_feats)
    gt_embbed_list = torch.cat(gt_embbed_list, dim=0)

    # pred list & pred embed
    logging.info('Building Pred Embedding List')
    pred_list = []
    pred_embbed_list = []
    gt2pred = {}    # gt string to pred string
    for result_item in tqdm(result_pred):
        pred = result_item['answer'].replace('.', '')
        if len(pred) > cut_length:
            pred = pred[:cut_length]
        pred_list.append(pred)
        gt = result_item['gt'][0]
        gt2pred[gt] = pred

        pred = clip.tokenize(pred).cuda()
        pred_feats = model.encode_text(pred)
        pred_embbed_list.append(pred_feats)
    pred_embbed_list = torch.cat(pred_embbed_list, dim=0)
    pred_answer2id = {ans: idx for idx, ans in enumerate(pred_list)}

    logging.info('Getting Accuracy')
    # A. video to text retrieval
    acc_count = [0 for _ in top_k]
    for ith, result_item in tqdm(enumerate(result_pred)):
        gt = result_item['gt'][0]
        pred_feats = pred_embbed_list[ith]
        acc_list = torch.cosine_similarity(pred_feats, gt_embbed_list)
        _, c_idx = torch.topk(acc_list, max(top_k), dim=0)
        for idx, k in enumerate(top_k):
            correct_k = c_idx[:k].eq(answer2id[gt]).view(-1).float().sum().item()
            acc_count[idx] += correct_k

    final_acc = [_ / len(result_pred) * 100 for _ in acc_count]
    json_output = {}
    for idx, k in enumerate(top_k):
        logging.info("V2T: The Top-{} Accuracy of retrival is {:.2f}%, {}/{}.".format(k, final_acc[idx], int(acc_count[idx]), len(result_pred)))
        json_output['V2T_retrival_accuracy_top{}'.format(k)] = final_acc[idx]

    # B. text to video retrieval
    acc_count = [0 for _ in top_k]
    for ith, gt in tqdm(enumerate(gt_set)):
        pred = gt2pred[gt]
        pred_id = pred_answer2id[pred]
        gt_feats = gt_embbed_list[ith]
        acc_list = torch.cosine_similarity(gt_feats, pred_embbed_list)
        _, c_idx = torch.topk(acc_list, max(top_k), dim=0)
        for idx, k in enumerate(top_k):
            correct_k = c_idx[:k].eq(pred_id).view(-1).float().sum().item()
            acc_count[idx] += correct_k

    final_acc = [_ / len(result_pred) * 100 for _ in acc_count]
    for idx, k in enumerate(top_k):
        logging.info("T2V: The Top-{} Accuracy of retrival is {:.2f}%, {}/{}.".format(k, final_acc[idx], int(acc_count[idx]), len(gt_set)))
        json_output['T2V_retrival_accuracy_top{}'.format(k)] = final_acc[idx]

    # # C. save to json
    # with smart_open(output_name, 'w') as f:
    #     json.dump(json_output, f)

    return json_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--exp_root', type=str, default='/data/projects/code_yc/mmgpt/checkpoints/msvd_eval_results/')
    # parser.add_argument('--exp_name', type=str, default='webvid_0711_03ep_vtype-v2_qadata_testval_ckpt2000_gpu8')
    parser.add_argument('--pred_path', type=str, default='')
    parser.add_argument("--gtfile_path", type=str, default='', help='Optional, if given, gt will replace as pred - gt')
    parser.add_argument('--out_path', type=str, default='')
    parser.add_argument('--clip_model', type=str, default='RN50')
    args = parser.parse_args()

    # pred_file_name = os.path.join(args.exp_root, args.exp_name)
    # output_name = os.path.join(args.exp_root, args.out_name)
    pred_file_name = args.pred_path
    output_name = args.out_path
    cal_retrival_acc(pred_file_name, output_name, clip_model=args.clip_model, gt_path=args.gtfile_path)
