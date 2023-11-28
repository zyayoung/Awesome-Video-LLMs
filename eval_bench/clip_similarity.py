import argparse
import json
import os

import clip
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def cal_clip_sim(file_name, output_name, clip_model='RN50', threshold=[0.5, 0.6, 0.7, 0.8], cut_length=77):
    print('Loading CLIP')
    model, preprocess = clip.load(clip_model, device='cuda')
    with open(file_name, 'r') as f:
        result_pred = json.load(f)

    acc_list = [0 for _ in threshold]
    
    for result_item in tqdm(result_pred):
        gt = result_item['gt']
        pred = result_item['answer']

        if len(pred) > cut_length:
            pred = pred[:cut_length]
        try:
            pred = clip.tokenize(pred).cuda()
        except:
            continue
        gt = [clip.tokenize(gt_ans).cuda() for gt_ans in gt]
        gt = torch.cat(gt, dim=0)

        pred_feats = model.encode_text(pred)
        gt_feats = model.encode_text(gt)
        sim = F.cosine_similarity(pred_feats, gt_feats, dim=1)

        cat_score, _ = torch.max(sim, 0)

        for idx, th in enumerate(threshold):
            if cat_score > th:
                acc_list[idx] += 1.0
    final_acc = [acc / len(result_pred) for acc in acc_list]
    for idx, th in enumerate(threshold):
        print("The Accuracy of threshold {} is {}, {}/{}.".format(th, final_acc[idx], int(acc_list[idx]), len(result_pred)))
    with open(output_name, 'w') as f:
        json.dump({th: acc for th, acc in zip(threshold, final_acc)}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_root', type=str, default='/data/projects/code_yc/mmgpt/checkpoints/webvid_eval_results/')
    parser.add_argument('--exp_name', type=str, default='webvid_0706_03ep_5frame_val_ckpt10000_gpu1')
    parser.add_argument('--clip_model', type=str, default='RN50')
    args = parser.parse_args()

    exp_folder = os.path.join(args.exp_root, args.exp_name)
    file_name = os.path.join(exp_folder, 'results_final.json')
    output_name = os.path.join(exp_folder, 'clip_sim.json')
    cal_clip_sim(file_name, output_name, clip_model=args.clip_model)
