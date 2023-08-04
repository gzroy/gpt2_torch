from transformers import GPT2Tokenizer
from model_v1 import GPT2
import torch
import argparse
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt2 prompt fine tunning')
    parser.add_argument('--checkpointfile', type=str, default='checkpoints_prompt/model_1.pt')
    parser.add_argument('--question', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--maxtoken', type=int, default=0)
    parser.add_argument('--answer_num', type=int, default=1)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer.get_vocab())

    checkpoint = torch.load(args.checkpointfile)
    config = checkpoint['config']
    model = GPT2(**config)
    model.to(args.device)
    model = torch.compile(model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    prompt = f"### Prompt: {args.question}\n"
    prompt_id = tokenizer.encode(prompt)
    input_data = torch.reshape(torch.tensor(prompt_id, device=args.device), [1,-1])

    maxtoken = args.maxtoken
    if maxtoken==0:
        maxtoken = config['block_size'] - input_data.shape[1]
    
    for i in range(args.answer_num):
        predicted = model.generate(input_data, maxtoken, 1.0, args.topk, config['block_size'])
        predicted_data = predicted.cpu().numpy()[0]
        index = np.argwhere(predicted_data==(vocab_size-1))
        predicted_data = predicted_data[:index[0][0]]
        print(tokenizer.decode(predicted_data))