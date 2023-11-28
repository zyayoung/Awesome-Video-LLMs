import os
import argparse
import json
import re

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
parser.add_argument("--gtfile_path", type=str, required=True)
parser.add_argument("--image_path", type=str, required=True)
parser.add_argument("--out_path", type=str, required=True)
parser.add_argument("--num-chunks", type=int, default=1)
parser.add_argument("--datatype", type=str, required=True)
parser.add_argument("--prompt", type=str, default=None)
args = parser.parse_args()

accuracy_metric_tasks = ["TextVQA", "VizWizVQA", "OKVQA", "VQAv2", "TVQA"]
ANLS_metric_tasks = ["DocVQA", "InfographicVQA", "ST-VQA"]
EM_metric_tasks = ["TallyQA", "OCR-VQA", "AI2D"]
RA_metric_tasks = ["ChartQA"]
Caption_CIDEr_metric_tasks = ["COCO-Captions", "NoCaps", "TextCaps", "VizWiz-Cap", "Screen2Words"]
OCR_metric_tasks = ["OCR"]
OCR_prompt = "what is written in the image?"

# --------------- First step: multi-GPU evaluation --------------------

# assert args.prompt is None or args.prompt == 'None', args.prompt
# os.system("python -m llava.eval_video.multi_hardware_eval" + " "
#           + "--model_name" + " " + str(args.model_name) + " "
#           + "--gtfile_path" + " " + str(args.gtfile_path) + " "
#           + "--image_path" + " " + str(args.image_path) + " "
#           + "--out_path" + " " + str(args.out_path) + " "
#           + "--num-chunks" + " " + str(args.num_chunks) + " "
#           + "--datatype" + " " + str(args.datatype))


# # --------------- Second step: merge all results of each GPU ----------
# print("Merging evaluation results of each GPU")
# os.system("python -m llava.eval_video.merge_results" + " "
#           + "--out_path" + " " + str(args.out_path))


# --------------- Third step: evaluation metrics ----------------------
print("Start Evaluating.....")

metric_mapping = {"TextVQA": "accuracy",
                "VizWizVQA": "accuracy",
                "OKVQA": "accuracy",
                "VQAv2": "accuracy",
                "DocVQA": "ANLS",
                "InfographicVQA": "ANLS",
                "ST-VQA": "ANLS",
                "ST-TallyQA": "EM",
                "OCR-VQA": "EM",
                "AI2D": "EM",
                "ChartQA": "RA",
                "COCO-Captions": "CIDEr",
                "NoCaps": "CIDEr",
                "TextCaps": "CIDEr",
                "VizWiz-Cap": "CIDEr",
                "Screen2Words": "CIDEr",
                "OCR": "accuracy",
                "TVQA": "accuracy",
                "MSVD_cap": "accuracy",
                "MSVRVTT_cap": "accuracy"
                }

if 'cap' in args.datatype or 'CAP' in args.datatype or 'Cap' in args.datatype:
    acc = 0
else:
    qas = json.load(open(args.out_path + "/results_final.json", encoding='utf-8'))
    gt_qa = {str(i): ann["gt"] for i, ann in enumerate(qas)}
    pre_qa = {str(i): ann["answer"] for i, ann in enumerate(qas)}

    if args.datatype in accuracy_metric_tasks or args.datatype in ANLS_metric_tasks:
        from .Accuracy_ANLS_Eval import VQAEval
        vqaEval = VQAEval(gt_qa, pre_qa, args.datatype, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
        acc = vqaEval.evaluate()
        
    elif args.datatype in OCR_metric_tasks:
        
        def has_word(sentence, word):
            pattern = r"\b" + re.escape(word) + r"\b"
            match = re.search(pattern, sentence)
            if match:
                return True
            else:
                return False
        def remove_special_chars(s):
            pattern = r"[^a-zA-Z0-9\s]"
            s = re.sub(pattern, "", s)
            return s

        def test_ocr(gt_qa, pre_qa):
            correct = 0
            num = 0
            image_ids = list(gt_qa.keys())
            for image_id in image_ids:
                gt_answers = gt_qa[image_id]
                answer = pre_qa[image_id]
                gt_answers = remove_special_chars(gt_answers).lower()
                answer = remove_special_chars(answer).lower()
                if has_word(answer, gt_answers):
                    correct+= 1
                num += 1
            return correct, num

        correct, num = test_ocr(gt_qa, pre_qa)
        acc = float(correct)/num
        
        print("Correct: {}, Num: {}".format(correct, acc))


print("{} evaluation {} is {}:".format(args.datatype, metric_mapping[args.datatype], acc))
