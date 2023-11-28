from glob import glob
import json

prompt_tokens = 0
completion_tokens = 0
for log in glob("results/gpt_raw/*.json"):
    with open(log) as f:
        usage = json.load(f)[-1]['usage']
        prompt_tokens += usage['prompt_tokens']
        completion_tokens += usage['completion_tokens']

print("prompt_tokens:", prompt_tokens)
print("completion_tokens:", completion_tokens)
