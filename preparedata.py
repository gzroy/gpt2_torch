from datasets import load_dataset
from transformers import GPT2Tokenizer

dataset = load_dataset("bookcorpusopen", split="train")

block_size=513
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def tokenize_function(examples):
    token_ids = [tokenizer(text) for text in examples["text"]]
    total_length = [len(t["input_ids"]) for t in token_ids]
    total_length = [(l//(block_size+1))*(block_size+1) for l in total_length]
    result = []
    label = []
 
    for i in range(len(total_length)):
        result.extend([token_ids[i]["input_ids"][j:j+block_size+1] for j in range(0, total_length[i], block_size+1)])
    return {"token_ids": result}
 
ds_test = ds['train'].select(range(10000))
 
tokenized_datasets = ds_test.map(
    tokenize_function, batched=True, num_proc=8, remove_columns=["title", "text"], batch_size=100
)
 
tokenized_datasets.save_to_disk("boocorpusopen_10000_512tokens")