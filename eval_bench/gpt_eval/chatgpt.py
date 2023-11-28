'''
build chatgpt class to easily use it
'''

import openai
import json
import os
import hashlib


def get_messages(infos, type='caption'):
    if type == 'qa':
        # format
        # pair = f"Question {jth + 1}: {question}\nCorrect Answer {jth + 1}: {answer}\nPredicted Answer {jth + 1}: {pred}\n\n"
        # qa_pairs = qa_pairs + pair
        qa_pairs = infos
    elif type == 'caption':
        # format: f"Correct Caption: {}\n Predicted Caption: {}\n\n ..."
        caption_pairs = infos

    messages = {
        'qa': [
                    {
                        "role": "system",
                        "content":
                            "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                            "To evaluate the correctness of generative outputs for question-answer pairs, your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task: "
                            "I will provide a list of content to evaluate, please output them one by one in order.\n"
                            "------\n"
                            "##INSTRUCTIONS:\n"
                            "- Accurate and concise predictions are much better than lengthy ones, and vague predictions should be regarded as invalid.\n"
                            "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the correctness of the prediction compared to the answer.\n"
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pairs:\n\n"
                            f"{infos}"
                            "For each pair, provide your evaluation only as a yes/no and match score where the score is an integer value between 1 and 5, with 5 indicating the predicted answer is equivalent to the correct answer.\n"
                            "Please generate each response in the form of a Python dictionary string with keys 'score' and 'correct', where value of 'correct' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING. Then return a Python list by combining these responses.\n"
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string.\n"
                            "For example, your each response should look like this: {'score': 3, 'correct': 'yes'}."
                    }
                ],
        'caption_pr_v2': [
                    {
                        "role": "system",
                        "content":
                            "You are a video captions assessment tool."
                            "------\n"
                            "##INSTRUCTIONS:\n"
                            "- Synonyms and paraphrases are valid matches.\n"
                            "- Evaluate the precision and coverage of the prediction compared to the correct answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "You will be rating captions on two scales: precision and coverage. "
                            "These scales range from 1 to 5, with a higher score indicating a better match.\n"
                            "Precision: Accuracy and conciseness of the predicted caption. 5: a high level of accuracy with no extra details. 3: a partial match with some additional details that couldn't be verified.\n"
                            "Coverage: Comprehensiveness of the predicted caption. 5: the predicted caption includes all elements of the correct caption. 3: covers some parts of the correct caption.\n"
                            "Lengthy responses can have higher coverage but lower precision, vise versa.\n\n"
                            "Please return a JSON object for all questions above, where keys are question number starting from 1. Each value is an object with keys precision and coverage where values are integers from 1 to 5.\n"
                            "Example:\n\n"
                            "Correct caption a: there is a woman is talking to a person.\n"
                            "Predicted caption a: The video shows a man and a woman sitting in a car, talking to each other.\n\n"
                            "Correct caption b: a woman adds orange paste to flour.\n"
                            "Predicted caption b: a woman is making a dish.\n\n"
                            "Your response should be: {\"a\": {\"precision\": 4, \"coverage\": 5}, \"b\": {\"precision\": 5, \"coverage\": 3}}\n\n"
                            "Please evaluate the following video captions:\n\n"
                            f"{infos}"
                    }
                ]
    }
    return messages[type]


def set_openai():
    if 'OPENAI_API_BASE' in os.environ:
        openai.api_base = os.environ['OPENAI_API_BASE']
    openai.api_key = os.environ['OPENAI_API_KEY']


def ask_gpt(infos, type="caption", model="gpt-3.5-turbo-16k", **kwargs):
    messages = get_messages(infos, type=type)
    rsp = openai.ChatCompletion.create(
      model=model,
      messages=messages,
      temperature=0.2,  # avoid diverse answer
      timeout=10,
      request_timeout=10,
      **kwargs,
    )
    os.makedirs('results/gpt_raw', exist_ok=True)
    with open(f"results/gpt_raw/{hashlib.md5(json.dumps(messages).encode()).hexdigest()}.json", 'w') as f:
        json.dump([messages, rsp], f)
    return rsp.get("choices")[0]["message"]["content"]

def main():
    '''
    import ast
    cap1, cap2 = "there are two dogs and one cat in the beach.", "three dogs and two cat in the sky."
    cap5, cap6 = "there are two dogs and one cat in the beach.", "dogs and cat in the sky."
    cap3, cap4 = "the boy said: I love you.", "the boy said, I like you."
    infos = f"Correct Caption: {cap1}\n Predicted Caption: {cap2}\n\n " \
            f"Correct Caption: {cap5}\n Predicted Caption: {cap6}\n\n " \
            f"Correct Caption: {cap3}\n Predicted Caption: {cap4}\n\n "
    response = ask_gpt(infos, type="caption")
    response = ast.literal_eval(response)
    print(response)
    '''

    set_openai()
    # sys_content = 'You are an intelligent chatbot to generate prompt, you should generate as detail as possible.'
    sys_content = 'You are ChatGPT, a large language model trained by OpenAI. \n Knowledge cutoff: 2021-09 \n Current date: 2023-09-18. Please generate a prompt, you should generate as detail as possible.'
    # message = 'Given a list of question and answers, please help me to generate the prompt to evaluate the relevance of question and answer. If the answer is too long and most content are irrelevant to question, the score should be low. If the answer match to the question, the score should be high. No need to evaluate the correctness of answer, just evaluate the relevance score'
    message = 'Generate a prompt to evaluate correctness by gpt for action recognition, .'
    messages = [{"role": "system", "content": sys_content},
                {"role": "user", "content": message }]
    rsp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,  # avoid diverse answer
    )
    answer = rsp.get("choices")[0]["message"]["content"]
    print(answer)



if __name__ == '__main__':
    main()