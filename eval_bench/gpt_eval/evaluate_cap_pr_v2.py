from collections import defaultdict
import traceback
import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm
import time
import random
from concurrent.futures import ThreadPoolExecutor
from refile import smart_open
from eval_bench.gpt_eval.chatgpt import set_openai, ask_gpt

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--gt_path", required=False, default='', help="The path to file containing questions.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")

    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    # parser.add_argument("--task_name", required=True, type=str, default="MSVD", help="Task type in [MSVD, MSRVTT, TGIF]")
    parser.add_argument("--kernel_size", required=False, type=int, default=10, help="Kernel size of qa.")
    parser.add_argument("--max_try_times", required=False, type=int, default=100)
    parser.add_argument("--out_path", type=str, default='')
    args = parser.parse_args()
    return args


def annotate_multi_item(prediction_set, keys, output_dir, pbar):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    keys = [key[:-5] for key in keys]    # Strip file extension
    cap_sets = [prediction_set[key] for key in keys]

    # build QA pairs
    qa_pairs = ""
    for jth, qas in enumerate(cap_sets):
        qas = cap_sets[jth]
        answer, pred = qas['a'], qas['pred']
        if len(pred) > 500:
            pred = pred[:500]
        pair = f"Correct caption {jth+1}: {answer}\nPredicted caption {jth+1}: {pred}\n\n"
        qa_pairs = qa_pairs + pair
    try:
        response = ask_gpt(qa_pairs, type="caption_pr_v2", model="gpt-3.5-turbo-1106", response_format={"type": "json_object"})
        response_dict = json.loads(response)
        for jth, qas in enumerate(cap_sets):
            result_qa_pair = [response_dict[str(jth+1)], qas]
            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{keys[jth]}.json", "w") as f:
                json.dump(result_qa_pair, f)

    except Exception as e:
        print(f"Error processing file '{keys[0]}': {e}")
        traceback.print_tb(e.__traceback__)
        time.sleep(1)
    pbar.update(len(keys))

def build_prediction_set(args, pred_contents, cut_pred_first_sentence=False):
    '''
    :param pred_contents: [{}], each sample keys: [video_id, caption, gt, | answer, question_id,]
    '''
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    # caption one pred to many gt：评测时将其拆分为 k 组结果，分别评测。即 1 pred 复制 k 份。
    # update: too much computation cost for one pred - many gt, thus reverse to 1-1 pair.
    for sample in pred_contents:
        # duplicate(flatten) the pred-gt pairs
        video_id = sample['video_id']
        for gt_ans in sample['gt'][:1]: # [:1] control gt num
            # build new name
            if video_id in video_id_counts:
                video_id_counts[video_id] += 1
            else:
                video_id_counts[video_id] = 0
            new_sample = {
                'pred': sample['caption'],
                'answer': gt_ans,
                'video_name': f"{video_id}_{video_id_counts[video_id]}",
            }
            # Create a new sample with the modified key
            new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['video_name'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]
    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of caption sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['video_name']
        answer = sample['answer']   # 'gt'
        pred = sample['pred'] # 'answer'
        if cut_pred_first_sentence:
            pred = pred.split('.')[0] + '.'
        cap_set = {"a": answer, "pred": pred}
        prediction_set[id] = cap_set
    return prediction_set, output_dir, caption_files

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    if 's3://' in args.pred_path:
        with smart_open(args.pred_path) as f:
            pred_contents = json.load(f)
    else:
        with open(args.pred_path) as f:
            pred_contents = json.load(f)

    # build_prediction_set for each task
    prediction_set, output_dir, caption_files = build_prediction_set(args, pred_contents)
    num_tasks = args.num_tasks
    kernel_size = args.kernel_size
    set_openai()

    # While loop to ensure that all captions are processed.
    for _ in range(args.max_try_times):
        # Files that have not been processed yet.
        completed_files = os.listdir(output_dir)
        print(f"completed_files: {len(completed_files)}")
        if len(completed_files) == len(caption_files):
            print(f"incomplete_files: 0")
            break #TODO
            # pass
        else:
            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

        # Break the loop when there are no incomplete files
        if len(incomplete_files) == 0:
            break

        random.shuffle(incomplete_files)
        with ThreadPoolExecutor(num_tasks) as p:
            pbar = tqdm(total=len(incomplete_files))
            while incomplete_files:
                p.submit(annotate_multi_item, prediction_set, incomplete_files[:kernel_size], output_dir, pbar)
                incomplete_files = incomplete_files[kernel_size:]
        time.sleep(1)

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json
    # Iterate through json files
    bad_count = 0
    for file_name in tqdm(os.listdir(output_dir)):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            try:
                with open(file_path, "r") as json_file:
                    content = json.load(json_file)
                    combined_contents[file_name[:-5]] = content
            except:
                bad_count+=1
    print('result bad count', bad_count)
    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    keys = ["precision", "coverage"]
    count = defaultdict(list)
    abnormal_count = defaultdict(int)
    for result, _ in combined_contents.values():
        for key, value in result.items():
            if key not in keys or not range(1, 6):
                abnormal_count[key] += 1
            else:
                count[key].append(value)

    eval_results = {k: sum(v) / len(v) for k, v in count.items()}
    print('abnormal_count:', abnormal_count)
    print('eval_results:', eval_results)
    for key, value in eval_results.items():
        print(f"{key}:", f"{value:.2f}")

    if args.out_path != '':
        print(f"Save result to {args.out_path}")
        dir_name = os.path.dirname(args.out_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(args.out_path, 'w') as f:
            json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()

