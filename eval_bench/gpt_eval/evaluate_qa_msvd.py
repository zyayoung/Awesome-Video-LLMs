import traceback
import openai
import os
import argparse
import json
import ast
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import random
from refile import smart_open
from eval_bench.gpt_eval.chatgpt import set_openai, ask_gpt

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--question_path", required=False, default='', help="The path to file containing questions.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", required=False, help="OpenAI API key.")

    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    parser.add_argument("--task_name", required=True, type=str, default="MSVD", help="Task type in [MSVD, MSRVTT, TGIF]")
    parser.add_argument("--kernel_size", required=False, type=int, default=10, help="Kernel size of qa.")
    parser.add_argument("--cut_pred_first_sentence", required=False, type=bool, default=0, help="")
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
    qa_sets = [prediction_set[key] for key in keys]

    # build QA pairs
    qa_pairs = ""
    for jth, qas in enumerate(qa_sets):
        question, answer, pred = qas['q'], qas['a'], qas['pred']
        if len(pred) > 500:
            pred = pred[:500]
        answer = answer.strip()
        if answer[-1] != '.':
            answer = answer + '.'
        pair = f"Question {jth+1}: {question}\nCorrect Answer {jth+1}: {answer}\nPredicted Answer {jth+1}: {pred}\n\n"
        qa_pairs = qa_pairs + pair
    try:
        # Compute the correctness score
        response = ask_gpt(qa_pairs, type="qa", model="gpt-3.5-turbo")
        response_dict = ast.literal_eval(response)  # [{...}, {...}]
        assert isinstance(response_dict, list)
        assert len(response_dict) == len(keys), f"expected {len(keys)} answers but got {len(response_dict)}"
        # build pairs
        for jth, qas in enumerate(qa_sets):
            result_qa_pair = [response_dict[jth], qas]
            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{keys[jth]}.json", "w") as f:
                json.dump(result_qa_pair, f)
    except Exception as e:
        print(f"Error processing file '{keys[0]}': {e}")
        traceback.print_tb(e.__traceback__)
        time.sleep(1)
    pbar.update(len(keys))

def build_prediction_set_MSVD(args, pred_contents, cut_pred_first_sentence=False):
    # default: 'gt', 'answer'
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # add question path for msvd
    question_dict = {}
    if args.question_path != '':
        with smart_open(args.question_path, 'r') as f:
            q_data = json.load(f)
            for it in q_data:
                qid, question = it['question_id'], it['question']
                question_dict[qid] = question

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        # video_id = sample['video_name']
        if 'question_id' in sample.keys():
            video_id = sample['question_id']    # 'id'
        else:
            video_id = sample['id']  # 'id'

        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        if 'question' not in new_sample.keys():
            new_sample['question'] = question_dict[video_id]
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['video_name'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['video_name']
        # question = sample['Q']
        # answer = sample['A']
        # pred = sample['pred']
        question = sample['question']
        if 'gt' in sample.keys():
            answer = sample['gt'][0]  # 'answer'
            pred = sample['answer']  # 'pred'
        else:
            answer = sample['answer']
            pred = sample['pred']
        if cut_pred_first_sentence:
            pred = pred.split('.')[0] + '.'

        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set
    return prediction_set, output_dir, caption_files

def build_prediction_set_TGIF(args, pred_contents, cut_pred_first_sentence=False):
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['id'] # 0?
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['video_name'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['video_name']
        # question = sample['Q']
        # answer = sample['A']
        # pred = sample['pred']
        question = sample['question']
        answer = sample['answer']   # 'gt'
        pred = sample['pred'] # 'answer'
        if cut_pred_first_sentence:
            pred = pred.split('.')[0] + '.'
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set
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
    if args.task_name in ["MSVD", "MSRVTT"]:  # default
        prediction_set, output_dir, caption_files = build_prediction_set_MSVD(args, pred_contents, args.cut_pred_first_sentence)
    elif args.task_name in ["TGIF"]:
        prediction_set, output_dir, caption_files = build_prediction_set_TGIF(args, pred_contents, args.cut_pred_first_sentence)

    # Set the OpenAI API key.
    set_openai()
    num_tasks = args.num_tasks
    kernel_size = args.kernel_size

    # While loop to ensure that all captions are processed.
    for _ in range(args.max_try_times):
        # Files that have not been processed yet.
        completed_files = set(os.listdir(output_dir))
        print(f"completed_files: {len(completed_files)}")
        if len(completed_files) == len(caption_files):
            print(f"incomplete_files: 0")
            break
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
            except Exception:
                bad_count+=1
    print('result bad count', bad_count)
    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    abnormal_count = {'score': 0, 'correct': 0}
    for key, result in combined_contents.items():   # {'correct': 'yes', 'score': 4.8}
        # Computing score
        count += 1
        try:
            if type(result[0]) is str:
                result[0] = ast.literal_eval(result[0])
            score_match = result[0]['score']    # original
        except:
            score_match = -1
            for score in range(6):
                if score in result[0] or str(score) in result[0]:
                    score_match = score
                    break
            if score_match == -1:
                score_match = 3
                abnormal_count['score'] += 1
        score = int(score_match)
        score_sum += score

        # Computing accuracy
        correct = result[0]['correct']  # original
        if type(correct) is str:
            if "yes" in correct.lower():
                yes_count += 1
            elif "no" in correct.lower():
                no_count += 1
            else:
                abnormal_count["correct"] += 1
    print('Abnormal count', abnormal_count)

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


    eval_results = {"Accuracy": accuracy, "Average score": average_score}
    if args.out_path != '':
        print(f"Save result to {args.out_path}")
        dir_name = os.path.dirname(args.out_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(args.out_path, 'w') as f:
            json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    main()

