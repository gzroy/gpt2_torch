from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_from_disk
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import argparse
from model_v1 import GPT2
#from model import GPT2
import time
import math
from chatdata import ChatDataset

def load_basemodel(model_name, config):
    model_hf = GPT2Model.from_pretrained(model_name)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = sd_hf.keys()
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
    sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
    transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

    #model = GPT2(vocab_size, 768, 1024, 0, 12, 768*4, 0, 0, 0, 12)
    model = GPT2(**config)
    sd = model.state_dict()
    sd_keys = sd.keys()
    sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
    sd_keys = [k for k in sd_keys if not k.endswith('lm_head.weight')]

    for k in sd_keys_hf:
        if any(k.endswith(w) for w in transposed):
            # special treatment for the Conv1D weights we need to transpose
            assert sd_hf[k].shape[::-1] == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k].t())
        else:
            # vanilla copy over the other parameters
            assert sd_hf[k].shape == sd[k].shape
            with torch.no_grad():
                sd[k].copy_(sd_hf[k])
    
    return model, config

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters):
    min_lr = learning_rate/10
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

@torch.no_grad()
def accuracy(logits, targets):
    prediction = F.softmax(logits, dim=2)
    prediction = torch.argmax(prediction, dim=2)
    compare = torch.eq(prediction, targets).float()
    mask = 1 - torch.eq(targets, -1).float()
    accuracy = torch.sum(compare*mask)/torch.sum(mask)
    return accuracy.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt2 prompt fine tunning')
    parser.add_argument('--block_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--no_mixed', action='store_true', help='Specify this to not use mixed precesion to train')
    parser.add_argument('--dataset', type=str, default='chatbot_1.pkl', help='The dataset path')
    parser.add_argument('--learning_rate', type=float, default=0.00006)
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--start_epoch', type=int, default=0, help='If resume training then specify the epoch to continue')
    parser.add_argument('--steps_epoch', type=int, default=4000, help='Training batch size, original model use 64')
    parser.add_argument('--num_epoch', type=int, default=1, help='Specify the number of epochs to train')
    parser.add_argument('--total_epochs', type=int, default=10, help='Specify the total target epochs to train')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_prompt/')
    parser.add_argument('--logfile', type=str, default='train_result_gpt2_prompt.txt')
    parser.add_argument('--gptmodelname', type=str, default='gpt2')
    args = parser.parse_args()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer.get_vocab())

    mixed = False
    dtype = 'float32'
    if not args.no_mixed:
        mixed = True
        dtype = 'float16'

    start_epoch = args.start_epoch

    if args.resume!="":
        checkpoint = torch.load(args.checkpoint_path+args.resume)
        config = checkpoint['config']
        model = GPT2(**config)
        model.to(args.device)
        model = torch.compile(model)
        optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Checkpoint '+args.resume+' loaded! Resume from epoch '+str(start_epoch))
    else:
        config = {'vocab_size':vocab_size, 'd_model': 768, 'block_size': 1024, 'embed_pdrop': 0.1, 'num_heads': 12, 
                  'dff': 768*4, 'attn_pdrop': 0.1, 'resid_pdrop': 0.1, 'dropout': 0.1, 'num_layer': 12}
        model = load_basemodel(args.gptmodelname, config)
        model.to(args.device)
        model = torch.compile(model)
        optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), args.device)
    
    
    model.train()

    dataset = ChatDataset(args.dataset, 1024, 12)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    total_loss = 0
    total_accuracy = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    for epoch in range(start_epoch, start_epoch+args.num_epoch):
        start = time.time()
        for batch, (x,y) in enumerate(dataloader):
            optimizer.zero_grad()
            lr = get_lr(batch+epoch*args.steps_epoch, args.warmup_steps, args.learning_rate, args.steps_epoch*args.total_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            x = x.to(args.device)
            y = y.to(args.device)

            if mixed:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(x, y)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            total_accuracy += accuracy(logits, y)
            if batch%100 == 0 and batch>0:
                line = f'Batch: {batch+epoch*args.steps_epoch}, Loss: {total_loss/100:.4f}, Accuracy: {total_accuracy/100:.4f}, Learning_rate: {lr:.5f}'
                with open(args.logfile, 'a') as logfile:
                    logfile.write(line+'\n')
                print(line)
                total_loss = 0
                total_accuracy = 0
                if batch%args.steps_epoch == 0:
                    break

        with open(args.logfile, 'a') as logfile:
            line = f'Saving checkpoint for epoch {epoch+1} in {args.checkpoint_path}'
            logfile.write(line+'\n')
            print(line)
            line = f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n'
            logfile.write(line+'\n')
            print(line)
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config
            }, args.checkpoint_path+'model_'+str(epoch)+'.pt')
    

