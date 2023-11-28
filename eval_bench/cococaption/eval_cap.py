from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

from torchvision.datasets.utils import download_url
import json
import os
import argparse


def coco_caption_eval(gt_caption, pred_caption_file, output_name):
    coco = COCO(gt_caption)
    coco_result = coco.loadRes(pred_caption_file)
    
    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    with open(output_name, 'w') as f:
        json.dump(coco_eval.eval, f)

    return coco_eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_root", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--gtfile_path", type=str)
    args = parser.parse_args()
    exp_folder = os.path.join(args.exp_root, args.exp_name)
    file_name = os.path.join(exp_folder, 'results_final.json')
    output_name = os.path.join(exp_folder, 'cap.json')

    coco_caption_eval(args.gtfile_path, file_name, output_name=output_name)