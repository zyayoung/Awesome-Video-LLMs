import argparse
import json
import math
import os

import torch
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          CLIPImageProcessor, CLIPVisionModel,
                          StoppingCriteria)

from llava.data.loading import load_video
from llava.data.base_dataset import get_multimodal_template
from llava.model import LlavaLlamaForCausalLMVideo
from llava.constants import (DEFAULT_IM_END_TOKEN,
                                   DEFAULT_IM_START_TOKEN,
                                   DEFAULT_IMAGE_PATCH_TOKEN,
                                   DEFAULT_IMAGE_TOKEN,
                                   NUM_FRAME, VIDEO_BRANCH_TYPE)
from llava.conversation import SeparatorStyle, conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import KeywordsStoppingCriteria


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    # print(n,k)
    chunks = split_list(lst, n)
    return chunks[k]

output_list = []
gt_list = []

def eval_model(args):
    num_frames = NUM_FRAME
    video_branch_type = VIDEO_BRANCH_TYPE
    max_new_tokens = 4096

    # gts_path = args.gtfile_path
    # gts = json.load(open(gts_path, encoding='utf-8'))
    # print(gts)
    # gts = get_chunk(gts, args.num_chunks, args.chunk_idx)

    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = LlavaLlamaForCausalLMVideo.from_pretrained(model_name, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).cuda()
    print(model.config)
    print(model)

    mm_use_im_start_end = getattr(model.config, "use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16)
    assert vision_tower.device.type == 'cuda', "Vision tower must be on cuda"
    vision_tower.to(device='cuda', dtype=torch.float16)
    
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    assert mm_use_im_start_end, "Only support use_im_start_end=True for now"
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2 

    gts_path = args.gtfile_path
    gts = json.load(open(gts_path, encoding='utf-8'))
    ############################################################
    if 'cap' in args.datatype or 'CAP' in args.datatype or 'Cap' in args.datatype:
        eval_data_dict = {}
        for item in tqdm(gts['annotations']):
            image_name = item['video_id']
            if image_name in eval_data_dict.keys():
                eval_data_dict[image_name]["answer"].append(item["answer"][0])
            else:
                eval_data_dict[image_name] = item
        eval_data_list = []
        for c_k in eval_data_dict.keys():
            # eval_data_dict[c_k]['gt'] = eval_data_dict[c_k]['answer']
            eval_data_list.append(eval_data_dict[c_k])
        gts = eval_data_list
    ############################################################
    
    gts = get_chunk(gts, args.num_chunks, args.chunk_idx)

    failed_num = 0    
    for ann in tqdm(gts):
        try:
            output_json = {}
            
            qs_id = ann["question_id"]
            qs = ann["question"]
            image_file = ann["image"]

            image_file_path = os.path.join(args.image_path, image_file)

            image_qs = get_multimodal_template(video_branch_type, image_token_len, num_frames)
            qs = image_qs + '\n' + qs

            if "v1" in model_name.lower():
                conv_mode = "llava_v1"
            elif "mpt" in model_name.lower():
                conv_mode = "multimodal"
            else:
                conv_mode = "multimodal"

            if args.conv_mode is not None and conv_mode != args.conv_mode:
                print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
            else:
                args.conv_mode = conv_mode

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            inputs = tokenizer([prompt])
            
            video_path = image_file_path
            images = load_video(video_path, num_frames=num_frames)
            image_list = [image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0] for image in images]
            input_ids = torch.as_tensor(inputs.input_ids).cuda()
            
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    input_ids,
                    images=[torch.stack(image_list,dim=0).half().cuda()],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()

            print(f"Video: {image_file}. Question: {ann['question']}\n" \
                f"Prediction: {outputs}, +++++++++++++++++     GT: {ann['answer']}")
            if 'cap' in args.datatype or 'CAP' in args.datatype or 'Cap' in args.datatype:
                # output_json['image_id'] = image_file
                # output_json['image_id'] = ann['image_id']
                output_json['video_id'] = ann['video_id']
                output_json['caption'] = outputs
                output_json['answer'] = outputs
                output_json['question_id'] = qs_id
                output_json['gt'] = ann['answer']
                output_list.append(output_json)
                # gt_json = {}
                # gt_json['image_id'] = image_file
                # gt_json['caption'] = ann['answer']
                # gt_list.append(gt_json)
            else:
                output_json['question_id'] = qs_id
                output_json['answer'] = outputs
                output_json['gt'] = ann['answer']
            
                output_list.append(output_json)
        except:
            failed_num += 1
            print(f'evaluation failed, {ann["image"]}')
    if failed_num > 0:
        print(f'evalutaion failed: {failed_num} in {args.chunk_idx}')
    filename = args.out_path + "/results_" + str(args.chunk_idx) + ".json"
    with open(filename, 'w', encoding="utf-8") as file_obj:
        json.dump(output_list, file_obj, ensure_ascii=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/opt-350m")
    parser.add_argument("--gtfile_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--datatype", type=str, required=True)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)

    args = parser.parse_args()

    # print(args.num_chunks, args.chunk_idx)
    # exit()
    eval_model(args)
