from datasets import load_dataset
from transformers import GPT2Tokenizer
import pickle
from tqdm import trange
 
ds = load_dataset("alespalla/chatbot_instruction_prompts")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
 
data = []
token_len = []
block_size = 1024
for i in trange(ds['train'].__len__()):
    text = f"### Prompt: {ds['train'][i]['prompt'].strip()}\n### Response: {ds['train'][i]['response'].strip()}<|endoftext|>"
    ids = tokenizer(text)['input_ids']
    mask_ids = tokenizer(f"### Prompt: {ds['train'][i]['prompt'].strip()}")['input_ids']
    len_mask = len(mask_ids)
    targets = [-1 for _ in range(len_mask)]
    if len(ids)>block_size:
        targets.extend(ids[len_mask+1:block_size+1])
        data.append((ids[:block_size], targets))
        token_len.append(block_size)
    else:
        targets.extend(ids[len_mask+1:])
        data.append((ids[:-1], targets))
        token_len.append(len(ids[:-1]))
 
with open('chatdata.pkl', 'wb') as f:
    pickle.dump(data, f)