import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
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
    parser.add_argument("--max_eval_num", required=False, type=int, default=0) # if max_eval_num>0, eval max num
    parser.add_argument("--out_path", type=str, default='')
    args = parser.parse_args()
    return args

def annotate_multi_item(prediction_set, caption_files, output_dir, kernel_size=10):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for start_ith in tqdm(range(0, len(caption_files), kernel_size)):
        if start_ith + kernel_size <= len(caption_files):
            end_ith = start_ith + kernel_size
        else:
            end_ith = len(caption_files)
        length = end_ith - start_ith
        keys = caption_files[start_ith: end_ith]
        keys = [key[:-5] for key in keys]    # Strip file extension
        qa_sets = [prediction_set[key] for key in keys]
        questions = [qa_set['q'] for qa_set in qa_sets]
        answers = [qa_set['a'] for qa_set in qa_sets]
        preds = [qa_set['pred'] for qa_set in qa_sets]

        # build QA pairs
        qa_pairs = ""
        for jth in range(length):
            qas = qa_sets[jth]
            question, answer, pred = qas['q'], qas['a'], qas['pred']
            if len(pred) > 500:
                pred = pred[:500]
            pair = f"Correct Answer {jth+1}: {answer}\nPredicted Answer {jth+1}: {pred}\n\n"
            qa_pairs = qa_pairs + pair
        try:
            # Compute the correctness score
            response = ask_gpt(qa_pairs, type="action", model="gpt-3.5-turbo")
            response_dict = ast.literal_eval(response)  # [{...}, {...}]
            # build pairs
            for jth in range(length):
                response, qa_set = response_dict[jth], qa_sets[jth]
                result_qa_pair = [response, qa_set]
                # Save the question-answer pairs to a json file.
                with open(f"{output_dir}/{keys[jth]}.json", "w") as f:
                    json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{keys[0]}': {e}")
            time.sleep(1)

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
        video_id = sample['question_id']    # 'id'
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        if ('question' not in new_sample.keys()) and (len(question_dict)>0):
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
        question = "" # sample['question']
        answer = sample['gt'][0]   # 'answer'
        pred = sample['answer'] # 'pred'
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
        answer = sample['answer'][0]   # 'gt'
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

    if args.max_eval_num > 0:
        pred_contents = pred_contents[:args.max_eval_num]
    # build_prediction_set for each task
    if args.task_name in ["MSVD"]:  # default
        prediction_set, output_dir, caption_files = build_prediction_set_MSVD(args, pred_contents, args.cut_pred_first_sentence)
    elif args.task_name in ["MSRVTT", "TGIF"]:
        prediction_set, output_dir, caption_files = build_prediction_set_TGIF(args, pred_contents, args.cut_pred_first_sentence)

    # Set the OpenAI API key.
    set_openai()
    num_tasks = args.num_tasks
    kernel_size = int(args.kernel_size)
    max_try_times = args.max_try_times

    # While loop to ensure that all captions are processed.
    while True:
        max_try_times -= 1
        if max_try_times < 0:
            break
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
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
                # pass
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # TODO rollback
            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks   # if set num=1, the result will be only one process
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            # with Pool() as pool:
                # pool.starmap(annotate, task_args)
            # annotate(prediction_set, all_parts[0], output_dir)
            # annotate_multi_item(prediction_set, all_parts[0], output_dir)

            with Pool(int(num_tasks)) as p:
                for i in range(int(num_tasks)):
                    # chunk_id = i
                    # print('Run', chunk_id)
                    # p.apply_async(annotate, (prediction_set, all_parts[i], output_dir))
                    p.apply_async(annotate_multi_item, (prediction_set, all_parts[i], output_dir, kernel_size))
                p.close()
                p.join()
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            combined_contents = json.load(f)
    else:
        # Iterate through json files
        bad_count = 0
        for file_name in os.listdir(output_dir):
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
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    abnormal_count = {'score': 0, 'pred': 0}
    for key, result in combined_contents.items():   # {'pred': 'yes', 'score': 4.8}
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
        try:
            pred = result[0]['pred']  # original
        except:
            if 'yes' in str(result[0]):
                pred = 'yes'
            elif 'no' in str(result[0]):
                pred = 'no'
            else:
                pred = random.choice(['yes', 'no'])
                abnormal_count['pred'] += 1
        if type(pred) is str:
            if "yes" in pred.lower():
                yes_count += 1
            elif "no" in pred.lower():
                no_count += 1
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

