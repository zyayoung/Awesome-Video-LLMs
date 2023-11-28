'''
Eval caption or retrieval tasks
'''
import os
import argparse
import json
from copy import deepcopy
from refile import smart_open
import logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_task", type=str, help="in list [caption, retrieval]")
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--gtfile_path", type=str, help='used for caption task')
    parser.add_argument("--out_path", type=str, default='')

    # retrieval related
    parser.add_argument("--clip_model", type=str, default='ViT-B/32')
    parser.add_argument("--num_group", type=int, default=1)
    args = parser.parse_args()
    return args

def make_caption_multiple_files(gt_path, num_group):
    # build num_group gt files to evaluate, for caption & retrieval
    random.seed(0)
    filename = gt_path.split('/')[-1].split('.json')[0]
    _dir = f'tmp/multi_gt/{filename}/'
    os.makedirs(_dir, exist_ok=True)
    # check if files exist
    dir_list = os.listdir(_dir)
    if len(dir_list) >= num_group:
        gen_filenames = [os.path.join(_dir, fname) for fname in  dir_list if fname.endswith(".json")]
        return gen_filenames

    with smart_open(gt_path, 'r') as f:
        gt_data = json.load(f)['annotations']
    cap_nums = []
    gt_ids = [[] for _ in range(num_group)]
    for item in gt_data:
        cap_num = len(item['answer'])
        cap_nums.append(cap_num)
        sel_ids = random.sample(range(cap_num), num_group)
        for jth, sel_id in enumerate(sel_ids):
            gt_ids[jth].append(sel_id)

    # generate new file
    gen_filenames = []
    for ith in range(num_group):
        filename_ = _dir + filename + f'_{ith}.json'
        gen_filenames.append(filename_)
        new_data = deepcopy(gt_data)
        for jth, item in enumerate(gt_data):
            gt_id = gt_ids[ith][jth]
            new_data[jth]['answer'] = item['answer'][gt_id: gt_id+1]
            new_data_ = {'annotations': new_data}

        with open(filename_, 'w') as f:
            json.dump(new_data_, f, indent=2)
    return gen_filenames

def metric_aggregate(eval_results_, eval_results):
    if ith == 0:
        eval_results_ = eval_results
    else:
        for key, value in eval_results.items():
            eval_results_[key] += value
    return eval_results_

def metric_average(eval_results_all):
    metric_mean = eval_results_all[0]
    num = len(eval_results_all)
    for eval_results in eval_results_all[1:]:
        for key, value in eval_results.items():
            metric_mean[key] += value
    for key, value in metric_mean.items():
        metric_mean[key] = value / num
    return metric_mean

if __name__ == "__main__":
    args = parse_args()
    gt_path = args.gtfile_path
    num_group = args.num_group
    flag_eval_use_multiple_gt = True if num_group > 1 else False
    if flag_eval_use_multiple_gt:
        gen_filenames = make_caption_multiple_files(gt_path, num_group)

    task = args.eval_task.lower()
    if task == 'caption':
        logging.info('Caption task...')
        # build different gt caption file, for cross evaluation
        from eval_bench.cococaption.eval_cap_cider import coco_caption_eval
        if flag_eval_use_multiple_gt:
            eval_results_all = []
            # aggregate multiple results
            for ith in range(num_group):
                gt_path = gen_filenames[ith]
                eval_results = coco_caption_eval(gt_path, args.pred_path, output_name=None)
                eval_results_all.append(eval_results)
            eval_results = metric_average(eval_results_all)
        else:
            eval_results = coco_caption_eval(gt_path, args.pred_path, output_name=None)
        print('Final metric_mean', eval_results)
        print("CIDEr & Bleu_4 & METEOR & ROUGE_L")
        print(f"{eval_results['CIDEr'] * 100:.1f} & {eval_results['Bleu_4'] * 100:.1f} & {eval_results['METEOR'] * 100:.1f} & {eval_results['ROUGE_L'] * 100:.1f}")

    elif task == 'retrieval':
        logging.info('Retrieval task...')
        from eval_bench.retrieval.retrival_v2t_t2v import cal_retrival_acc
        if flag_eval_use_multiple_gt:
            eval_results_all = []
            # aggregate multiple results
            for ith in range(num_group):
                gt_path = gen_filenames[ith]
                eval_results = cal_retrival_acc(args.pred_path, args.out_path, clip_model=args.clip_model, gt_path=gt_path)
                eval_results_all.append(eval_results)
            eval_results = metric_average(eval_results_all)
        else:
            eval_results = cal_retrival_acc(args.pred_path, args.out_path, clip_model=args.clip_model, gt_path=gt_path)
        print('Final metric_mean', eval_results)
        print(f"{eval_results['V2T_retrival_accuracy_top1']:.1f}, {eval_results['V2T_retrival_accuracy_top5']:.1f} & {eval_results['T2V_retrival_accuracy_top1']:.1f}, {eval_results['T2V_retrival_accuracy_top5']:.1f}")

    elif task == 'action':
        logging.info('Action Recognition task...')
        from eval_bench.retrieval.action_acc import cal_acc
        eval_results = cal_acc(args.pred_path, args.out_path, clip_model=args.clip_model)

    if args.out_path != '':
        logging.info(f"Save result to {args.out_path}")
        dir_name = os.path.dirname(args.out_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(args.out_path, 'w') as f:
            json.dump(eval_results, f, indent=2)


