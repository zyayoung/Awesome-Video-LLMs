import os
import json
import pandas as pd
import random
from tqdm import tqdm
# eval_path = "/data/dataset/TextVQA/TextVQA_0.5.1_val.json"
# eval_path = "/data/dataset/InfographicsVQA/infographicVQA_val_v1.0.json"
# eval_path = "/data/dataset/VizWiz-VQA/val.json"
# eval_path = "/data/dataset/OKVQA/OpenEnded_mscoco_val2014_questions.json"; eval_path1 = "/data/dataset/OKVQA/mscoco_val2014_annotations.json"
# eval_path = "/data/dataset/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json"; eval_path1 = "/data/dataset/VQAv2/v2_mscoco_val2014_annotations.json"
# anns = json.load(open(os.path.join(eval_path1), "r"))
# q_anns = json.load(open(os.path.join(eval_path), "r"))
# eval_path = "/data/dataset/DocVQA/val_v1.0.json"
# eval_path = "/data/dataset/OCR/IC13_857_ocr.json"
# eval_path = "/data/dataset/TallyQA/test.json"

# eval_path = "/data/dataset/TallyQA/test.json"

# anns = json.load(open(os.path.join(eval_path), "r"))
q_list = ["Describe the video concisely.",
          "Provide a brief description of the given video clip.",
          "Offer a succinct explanation of the video presented.",
          "Summarize the visual content of the video.",
          "Give a short and clear explanation of the subsequent video.",
          "Share a concise interpretation of the footage provided.",
          "Present a compact description of the video’s key features.",
          "Relay a brief, clear account of the video shown.",
          "Render a clear and concise summary of the video clip.",
          "Write a terse but informative summary of the video.",
          "Create a compact narrative representing the video presented."]

anction_q_list = ["Describe the main action in the video concisely.",
          "Provide a brief description of the action in the given video clip.",
          "Offer an action category of the video presented.",
          "Summarize the temporal action of the video.",
          "Give a short and clear explanation of the action.",
          "Describe the temporal action in a word of the footage provided.",
          "Present a compact description of the video’s action.",
          "Relay a brief, clear account of the action shown in the video.",
          "Render a clear and concise action classification of the video clip.",
          "Classify the action of the video.",
          "Create a compact narrative representing the action presented."]

##################################################################
############# TVQA
# def series_name(vid_name):
#     folder_name = vid_name.split('_')[0]
#     if folder_name in ['castle', 'friends', 'grey', 'house', 'met']:
#         return folder_name + '_frames'
#     else:
#         return 'bbt_frames'
        
# data_path = '/data/workspace/data/dataset/TVQA'
# json_path = os.path.join(data_path, f"tvqa_val.jsonl")
# with open(json_path, "r") as f:
#     qa_json_list = list(f)
# eval_data = []
# for vid_name in qa_json_list:
#     ann = json.loads(vid_name)
#     vid_name = ann['vid_name']
#     folder_name = series_name(vid_name)
#     # video_path = os.path.join(data_path, 'frames/frames_hq', folder_name, vid_name)
#     video_path = os.path.join('frames/frames_hq', folder_name, vid_name)
#     # video_lists = sorted(os.listdir(video_path))
#     ann['video_path'] = video_path
#     # import ipdb; ipdb.set_trace()
    
#     answer_list = list()
#     for a_i in range(10):
#         an_name = 'a{}'.format(a_i)
#         if an_name in ann.keys():
#             answer_list.append(ann[an_name])
#         else:
#             break
#     # for a_i in range(ann['answer_idx']):
#     #     answer_list.append(ann['a{}'.format(a_i)])
#     # if len(answer_list) < 1:
#     #     for a_i in range(10):
#     #        an_name = 'a{}'.format(a_i)
#     #        if an_name in ann.keys():
#     #            answer_list.append(ann[an_name])
#     #        else:
#     #            import ipdb; ipdb.set_trace()
#     #            break
#     data = {
#         "image": ann['video_path'],
#         "question": ann['q'],
#         "answer": answer_list,
#         "question_id": ann['qid'],
#         "ts": ann['ts']
#     }
#     eval_data.append(data)
# with open(os.path.join("/data/workspace/data/dataset/TVQA/MMGPT_format_tvqa_val.json"), "w") as f:
#     json.dump(eval_data, f, indent=2)
##################################################################
############# WebVid
# list_data_dict = pd.read_csv("/data/workspace/data/dataset/WebVid10M/csv_list/results_10M_train.csv", header=0, delimiter=',')
# list_data_dict = pd.read_csv("/data/workspace/data/dataset/WebVid10M/csv_list/results_10M_val.csv", header=0, delimiter=',')
# oss_list = []
# # f = open("/data/projects/code_yc/mmgpt/available_list2/filepath_0.txt", 'r')
# f = open("/data/projects/code_yc/mmgpt/available_val_list/filepath_0.txt", 'r')
# for line in f.readlines():
#     oss_list.append(int(line.strip().split(' ')[0]))
# videoids = list(list_data_dict.values[:, 0])
# page_dirs = list(list_data_dict.values[:, 3])
# descrips = list(list_data_dict.values[:, 4])
# # import ipdb; ipdb.set_trace()
# eval_data = []
# for i in range(len(oss_list)):
#     qa = dict()
#     qa['page_dir'] = page_dirs[oss_list[i]]
#     qa['videoid'] = videoids[oss_list[i]]
#     qa['answer'] = [descrips[oss_list[i]]]

#     qa['question'] = q_list[random.randint(0,10)]
#     image_name = os.path.join(str(qa['page_dir']), str(qa['videoid'])+'.mp4')
#     # import ipdb; ipdb.set_trace()
#     data = {
#         "image": image_name,
#         "question": qa['question'],
#         "answer": qa['answer'],
#         "question_id": int(0),
#     }
#     eval_data.append(data)
# # with open(os.path.join("/data/workspace/data/dataset/WebVid10M/csv_list/MMGPT_format_webvid_file0.json"), "w") as f:
# with open(os.path.join("/data/workspace/data/dataset/WebVid10M/csv_list/MMGPT_format_webvid_val.json"), "w") as f:
#     json.dump(eval_data, f, indent=2)
##################################################################
##################################################################
############# MSVD
# def read_mapping(file_name):
#     with open(file_name, 'r') as f:
#         data = f.readlines()
    
#     id2name = dict()

#     for line in data:
#         try:
#             item = line.strip().split(' ')
#             assert(len(item)) == 2
#             video_name = item[0] + '.avi'
#             video_id = int(item[1][3:])
#             id2name[video_id] = video_name
#         except:
#             print(f'broken data: {line}')
#     return id2name

# data_root = '/data/datasets/MSVD/'
# split = "test"
# name_mapping_file = os.path.join(data_root, 'youtube_mapping.txt')
# label_list = os.path.join(data_root, f'{split}_qa.json')

# print('load label..')
# id2name = read_mapping(name_mapping_file)
# with open(label_list, 'r') as f:
#     data_list = json.load(f)

# # import ipdb; ipdb.set_trace()
# eval_data = []
# for i in tqdm(range(len(data_list))):
#     item = data_list[i]
#     answer = item['answer']
#     question = item['question']
#     videoid = item['video_id']
#     image_name = id2name[videoid]
#     # import ipdb; ipdb.set_trace()

#     # import ipdb; ipdb.set_trace()
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": [answer],
#         "question_id": int(item['id']),
#     }
#     eval_data.append(data)
# # with open(os.path.join("/data/workspace/data/dataset/WebVid10M/csv_list/MMGPT_format_webvid_file0.json"), "w") as f:
# with open(os.path.join(data_root, "MMGPT_evalformat_MSVD_{}.json".format(split)), "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(os.path.join(data_root, "MMGPT_evalformat_MSVD_{}.json".format(split)))
##################################################################
##################################################################
############# MSVTT
# data_root = '/data/datasets/MSR-VTT/'
# split = "val"

# label_list = os.path.join(data_root, f'msrvtt-qa/{split}.csv')
# print('load label..')
# data_list = pd.read_csv(label_list, header=0, delimiter=',')

# eval_data = []
# for i in tqdm(range(len(data_list))):
#     answer = data_list['answer'][i]
#     question = data_list['question'][i]
#     image_name = "video{}.mp4".format(data_list['video'][i])
#     question_id = data_list['qid'][i]
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": [answer],
#         "question_id": int(question_id),
#     }
#     eval_data.append(data)

# with open(os.path.join(data_root, "MMGPT_evalformat_MSRVTT_{}.json".format(split)), "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(os.path.join(data_root, "MMGPT_evalformat_MSRVTT_{}.json".format(split)))
##################################################################
############# TGIF
# data_root = '/data/datasets/tgif/tgif-qa/dataset'
# split = "Test"

# data_list = pd.read_csv(os.path.join(data_root,f'{split}_frameqa_question.csv'), sep='\t')
# # gif_name = list_data_dict['gif_name']
# # questions = list_data_dict['question']
# # answers = list_data_dict['answer']
# # vid_id = list_data_dict['vid_id']
# # key = list_data_dict['key']
# print('load label..')

# eval_data = []
# for i in tqdm(range(len(data_list))):
#     # import ipdb; ipdb.set_trace()
#     answer = data_list['answer'][i]
#     question = data_list['question'][i]
#     image_name = data_list['gif_name'][i]+'.gif'
#     question_id = 0
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": [answer],
#         "question_id": int(question_id),
#     }
#     eval_data.append(data)
# save_path = os.path.join(data_root, "MMGPT_evalformat_tgif_frameqa_{}.json".format(split))
# with open(save_path, "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(save_path)
##################################################################
##################################################################
############# TGIF
# def series_name(vid_name):
#     folder_name = vid_name.split('_')[0]
#     if folder_name in ['castle', 'friends', 'grey', 'house', 'met']:
#         return folder_name + '_frames'
#     else:
#         return 'bbt_frames'
    
# data_root = '/data/workspace/data/dataset/TVQA'
# # split = "test_public"
# split = "val"

# label_list = os.path.join(data_root, f'tvqa_{split}.jsonl')
# print('load label..')
# with open(label_list, 'r') as f:
#     data_list = list(f)

# eval_data = []
# for i in tqdm(range(len(data_list))):
#     item = json.loads(data_list[i])
#     answer_list = list()
#     for idx in range(5):
#         answer = item[f'a{idx}']
#         answer_list.append(answer)
#     question = item['q']
#     image_name = os.path.join(series_name(item["vid_name"]), item["vid_name"]+'.mp4')
#     question_id = item['qid']
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": answer_list,
#         "question_id": int(question_id),
#     }
#     eval_data.append(data)
# save_path = os.path.join(data_root, "MMGPT_evalformat_tvqa_{}.json".format(split))
# with open(save_path, "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(save_path)
##################################################################
##################################################################
############# MSVTT Caption
# data_root = '/data/datasets/MSR-VTT/'
# split = "test"

# label_list = os.path.join(data_root, f'{split}_videodatainfo.json')
# with open(label_list, 'r') as f:
#     data_list = json.load(f)
# print('load label..')

# eval_data_dict = {}
# eval_data_list = []
# for item in tqdm(data_list['sentences']):
#     answer = item['caption']+'.'
#     question = q_list[random.randint(0,10)]
#     image_name = 'TestVideo/{}.mp4'.format(item['video_id'])
#     question_id = 0
#     videoid = item['video_id'][5:]
#     # import ipdb; ipdb.set_trace()
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": [answer],
#         "question_id": int(question_id),
#         # 'image_id': int(videoid),
#         'video_id': int(videoid),
#         'caption': answer,
#         'id': int(videoid)
#     }
#     eval_data_list.append(data)
#     if image_name in eval_data_dict.keys():
#         eval_data_dict[image_name]["answer"].append(answer)
#     else:
#         eval_data_dict[image_name] = data


# images = list()
# for c_k in eval_data_dict.keys():
#     images.append({'id':eval_data_dict[c_k]['video_id']})

# eval_data = dict()
# eval_data['annotations'] = eval_data_list
# eval_data['images'] = images

# with open(os.path.join(data_root, "MMGPT_evalformat_MSRVTT_Caption_{}.json".format(split)), "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(os.path.join(data_root, "MMGPT_evalformat_MSRVTT_Caption_{}.json".format(split)))
##################################################################
##################################################################
############# MSVD Caption
# def read_mapping(file_name):
#     with open(file_name, 'r') as f:
#         data = f.readlines()
    
#     id2name = dict()

#     for line in data:
#         try:
#             item = line.strip().split(' ')
#             assert(len(item)) == 2
#             video_name = item[0] + '.avi'
#             video_id = int(item[1][3:])
#             id2name[video_id] = video_name
#         except:
#             print(f'broken data: {line}')
#     return id2name

# data_root = '/data/datasets/MSVD/captions'
# split = "test"

# name_mapping_file = os.path.join(data_root, 'youtube_mapping.txt')
# label_list = os.path.join(data_root, f'sents_{split}_lc_nopunc.txt')

# print('load label..')
# id2name = read_mapping(name_mapping_file)

# with open(label_list, 'r') as f:
#     data_list = f.readlines()

# print('read data..')

# ann_id = 0
# eval_data_dict = {}
# eval_data_list = []
# for item in tqdm(data_list):
#     item = item.strip().split('\t')
#     answer = item[1]+'.'
#     question = q_list[random.randint(0,10)]
#     videoid = item[0][3:]
#     image_name = id2name[int(videoid)]
#     # import ipdb; ipdb.set_trace()
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": [answer],
#         "question_id": int(0),
#         # 'image_id': int(videoid),
#         'video_id': int(videoid),
#         'caption': answer,
#         'id': int(videoid)
#     }
#     eval_data_list.append(data)
#     if image_name in eval_data_dict.keys():
#         # eval_data_dict[image_name]["caption"].append(answer)
#         pass
#     else:
#         eval_data_dict[image_name] = data
#         ann_id = ann_id+1

# # eval_data_list = []
# # for c_k in eval_data_dict.keys():
# #     eval_data_list.append(eval_data_dict[c_k])

# images = list()
# for c_k in eval_data_dict.keys():
#     images.append({'id':eval_data_dict[c_k]['video_id']})

# eval_data = dict()
# eval_data['annotations'] = eval_data_list
# eval_data['images'] = images
# # with open(os.path.join("/data/workspace/data/dataset/WebVid10M/csv_list/MMGPT_format_webvid_file0.json"), "w") as f:
# with open(os.path.join(data_root, "MMGPT_evalformat_MSVD_Caption_{}.json".format(split)), "w") as f:
# # with open(os.path.join(data_root, "MMGPT_evalformat_MSVD_Caption_mini100_{}.json".format(split)), "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(os.path.join(data_root, "MMGPT_evalformat_MSVD_Caption_{}.json".format(split)))
##################################################################
##################################################################
############# ActivityNet200
# def read_mapping(file_name):
#     with open(file_name, 'r') as f:
#         data = json.load(f)

#     id2name = {int(key):data[key] for key in data.keys()}
#     return id2name

# # data_root = '/data/workspace/data/dataset/ActivityNet/'
# data_root = '/data/VideoDataset/NIU/'
# split = "val"
# name_mapping_file = os.path.join(data_root, 'anet/anet1.3_label2name.json')
# label_list = os.path.join(data_root, f'anet/anet_{split}_video_fps1.txt')

# print('load label..')
# id2name = read_mapping(name_mapping_file)
# with open(label_list, 'r') as f:
#     data_list = f.readlines()
    
# eval_data = []
# for line in tqdm(data_list):
#     item = line.strip().split(' ')
#     # import ipdb; ipdb.set_trace()
#     v_name = item[0]
#     v_id = item[-1]
#     mp4name = f"v_{v_name}.mp4"
#     mkvname = f"v_{v_name}.mkv"
#     if os.path.exists(os.path.join(data_root,'activitynet_videos',mp4name)) or \
#         os.path.exists(os.path.join(data_root,'activitynet_videos',mkvname)):
#         if os.path.exists(os.path.join(data_root,'activitynet_videos',mp4name)):
#             image_name = mp4name
#         else:
#             image_name = mkvname
#         answer = id2name[int(v_id)]
#         question = anction_q_list[random.randint(0,10)]
#         data = {
#             "image": image_name,
#             "question": question,
#             "answer": [answer],
#             "question_id": int(0),
#         }
#         eval_data.append(data)
# # with open(os.path.join("/data/workspace/data/dataset/WebVid10M/csv_list/MMGPT_format_webvid_file0.json"), "w") as f:
# with open(os.path.join(data_root, "MMGPT_evalformat_ActivityNet_{}.json".format(split)), "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(os.path.join(data_root, "MMGPT_evalformat_ActivityNet_{}.json".format(split)))
##################################################################
# ##################################################################
# ############# HMDB51
# def read_mapping(file_name):
#     with open(file_name, 'r') as f:
#         data = json.load(f)

#     id2name = {int(key):data[key] for key in data.keys()}
#     return id2name

# # data_root = '/data/workspace/data/dataset/ActivityNet/'
# data_root = '/data/workspace/data/dataset/HMDB-51'
# split = "val"
# name_mapping_file = os.path.join(data_root, 'id2class_mapping.json')
# label_list = os.path.join(data_root, f'{split}.json')

# print('load label..')
# id2name = read_mapping(name_mapping_file)
# with open(label_list, 'r') as f:
#     data_list = json.load(f)
    
# eval_data = []
# for item in tqdm(data_list):
#     # item = line.strip().split(' ')
#     # import ipdb; ipdb.set_trace()
#     v_name = str(list(item.keys())[0])
#     v_id = int(item[v_name])
#     image_name = v_name
#     answer = id2name[int(v_id)]
#     # if '_' in id2name[int(v_id)]:
#     answer = answer.replace('_',' ')
#     question = anction_q_list[random.randint(0,10)]
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": [answer],
#         "question_id": int(0),
#     }
#     eval_data.append(data)
# file_path = os.path.join(data_root, "MMGPT_evalformat_HMDB51_{}.json".format(split))
# with open(file_path, "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(file_path)
# ##################################################################
##################################################################
############# UCF101
# def read_mapping(file_name):
#     with open(file_name, 'r') as f:
#         data = f.readlines()

#     id2sentence = {}
#     name2id = {}
#     for line in data:
#         id, cla_name = line.strip().split(" ")
#         up_list = []
#         for c_i in range(len(cla_name)):
#             if cla_name[c_i] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
#                 up_list.append(c_i)
#         split_words = ""
#         for c_i in range(len(up_list)):
#             if c_i == 0:
#                 if c_i != len(up_list)-1:
#                     split_words = split_words+cla_name[up_list[c_i]:up_list[c_i+1]]+" "
#                 else:
#                     split_words = split_words+cla_name[up_list[c_i]:].lower()
#             elif c_i == len(up_list)-1:
#                 split_words = split_words+cla_name[up_list[c_i]:].lower()
#             else:
#                 split_words = split_words+cla_name[up_list[c_i]:up_list[c_i+1]].lower()+" "
#         id2sentence[int(id)] = split_words
#         name2id[cla_name] = int(id)
#     return id2sentence, name2id

# def load_smaples(data_root, txt_list):
#     smaples = list()
#     for file_name in txt_list:
#         f = open(os.path.join(data_root, file_name), 'r')
#         for i_f in f.readlines():
#             cat_name, f_name = i_f.strip().split('/')
#             # import ipdb; ipdb.set_trace()
#             i_sample = (cat_name, i_f.strip())
#             smaples.append(i_sample)
#     return smaples

# data_root = '/data/workspace/data/dataset/UCF101/'
# split = "test"
# name_mapping_file = os.path.join(data_root, 'ucfTrainTestlist/classInd.txt')

# label_lists = ['ucfTrainTestlist/{}list0{}.txt'.format(split,i) for i in range(1,4)]

# print('load label..')
# id2sentence, name2id = read_mapping(name_mapping_file)
# data_list = load_smaples(data_root, label_lists)

# eval_data = []
# for item in tqdm(data_list):
#     # item = line.strip().split(' ')
#     # import ipdb; ipdb.set_trace()
    
#     answer = id2sentence[int(name2id[item[0]])]
#     image_name = item[1]
#     question = anction_q_list[random.randint(0,10)]
#     data = {
#         "image": image_name,
#         "question": question,
#         "answer": [answer],
#         "question_id": int(0),
#     }
#     eval_data.append(data)
# file_path = os.path.join(data_root, "MMGPT_evalformat_UCF101_{}.json".format(split))
# with open(file_path, "w") as f:
#     json.dump(eval_data, f, indent=2)
# print(file_path)
##################################################################