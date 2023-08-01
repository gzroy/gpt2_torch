from transformers import GPT2Tokenizer
from model import GPT2
import torch
from torch.nn import functional as F
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt2 predict')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/')
    parser.add_argument('--checkpoint_name', type=str, default='')
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--dff', type=int, default=768*4)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--decoder_layers', type=int, default=6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input', type=str)
    parser.add_argument('--generate_len', type=int, default=100)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer.get_vocab())
    model = GPT2(vocab_size, args.d_model, args.block_size, 0, args.heads, args.dff, 0, 0, 0, args.decoder_layers)
    model.to(args.device)
    model = torch.compile(model)
    checkpoint = torch.load(args.checkpoint_path+args.checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    token_id = tokenizer.encode(args.input)
    input_data = torch.reshape(torch.tensor(token_id, device=args.device), [1,-1])
    predicted = model.generate(input_data, args.generate_len, 1.0, args.topk, args.block_size)
    print("Generated text:\n-------------------")
    print(tokenizer.decode(predicted.cpu().numpy()[0]))
