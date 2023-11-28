import argparse
import json
import os

import clip
import torch
from tqdm import tqdm
from refile import smart_open

import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

@torch.no_grad()
def cal_acc(file_name, output_name, clip_model='RN50', gt_path='',  cut_length=77, top_k=(1, 5)):
    logging.info('Loading CLIP')
    model, preprocess = clip.load(clip_model, device='cuda')
    with smart_open(file_name, 'r') as f:
        result_pred = json.load(f)

    # gt set process
    logging.info('Building GT Embedding List')
    gt_set = set()
    for result_item in result_pred:
        _item = result_item['gt'][0]
        if _item.lower() == 'none': # skip None item
            continue
        gt_set.add(_item)
    gt_set = list(gt_set)
    answer2id = {ans:idx for idx, ans in enumerate(gt_set)}

    gt_embbed_list = list()
    for gt in tqdm(gt_set):
        gt = clip.tokenize(gt).cuda()
        gt_feats = model.encode_text(gt)
        gt_embbed_list.append(gt_feats)
    gt_embbed_list = torch.cat(gt_embbed_list, dim=0)


    logging.info('Getting Accuracy')
    acc_count = [0 for _ in top_k]
    for result_item in tqdm(result_pred):
        gt = result_item['gt'][0]
        if gt.lower() == 'none': # skip None item
            continue
        pred = result_item['answer'].replace('.', '')
        if len(pred) > cut_length:
            pred = pred[:cut_length]
        try:
            pred = clip.tokenize(pred).cuda()
        except:
            logging.warning(f'Input {pred} cannot tokenize as normal.' )
            continue
        pred_feats = model.encode_text(pred)
        acc_list = torch.cosine_similarity(pred_feats, gt_embbed_list)
        _, c_idx = torch.topk(acc_list, max(top_k), dim=0)
        for idx, k in enumerate(top_k):
            correct_k = c_idx[:k].eq(answer2id[gt]).view(-1).float().sum().item()
            acc_count[idx] += correct_k

    final_acc = [_ / len(result_pred) for _ in acc_count]
    json_output = {}
    for idx, k in enumerate(top_k):
        logging.info("The Top-{} Accuracy is {}, {}/{}.".format(k, final_acc[idx], int(acc_count[idx]), len(result_pred)))
        json_output['accuracy_top{}'.format(k)] = final_acc[idx]
    # with open(output_name, 'w') as f:
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
    cal_acc(pred_file_name, output_name, clip_model=args.clip_model, gt_path=args.gtfile_path)

