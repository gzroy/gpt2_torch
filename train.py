import argparse
from datasets import load_from_disk
from transformers import GPT2Tokenizer
from torch.utils.data import DataLoader
from model import GPT2
import torch
from torch.nn import functional as F
import math
import time

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
    accuracy = torch.mean(compare).item()
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gpt1')
    parser.add_argument('--resume', type=str, default='', help='Specify the CKPT name for resume training')
    parser.add_argument('--start_epoch', type=int, default=0, help='If resume training then specify the epoch to continue')
    parser.add_argument('--num_epoch', type=int, default=1, help='Specify the number of epochs to train')
    parser.add_argument('--steps_epoch', type=int, default=5000, help='Specify the steps of epoch')
    parser.add_argument('--total_epochs', type=int, default=120, help='Specify the total target epochs to train')
    parser.add_argument('--no_mixed', action='store_true', help='Specify this to not use mixed precesion to train')
    parser.add_argument('--dataset', type=str, default='boocorpusopen_10000_513tokens_gpt2/', help='The dataset path')
    parser.add_argument('--block_size', type=int, default=512, help='The sequence lenght of the tokens for trianing')
    parser.add_argument('--decoder_layers', type=int, default=6, help='Decoder layers, orginial gpt1 model contains 12 layers')
    parser.add_argument('--heads', type=int, default=12, help='Multi attention heads per decoder layer')
    parser.add_argument('--d_model', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--dff', type=int, default=3072, help='Feed forward layer feature dimension')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size, original model use 64')
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--embed_pdrop', type=float, default=0.1)
    parser.add_argument('--ff_pdrop', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--learning_rate', type=float, default=0.0006, help='Original gpt1 use 0.00025')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    parser.add_argument('--logfile', type=str, default='train_result_gpt2.txt')
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset)
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    vocab_size = len(tokenizer.get_vocab())

    mixed = False
    dtype = 'float32'
    if not args.no_mixed:
        mixed = True
        dtype = 'float16'
    
    model = GPT2(
        vocab_size = vocab_size, 
        d_model = args.d_model, 
        block_size = args.block_size, 
        embed_pdrop = args.embed_pdrop,
        num_heads = args.heads,
        dff = args.dff,
        attn_pdrop = args.attn_pdrop,
        resid_pdrop = args.resid_pdrop,
        dropout = args.ff_pdrop,
        num_layer = args.decoder_layers)
    
    model.to(args.device)
    model = torch.compile(model)

    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (0.9, 0.95), args.device)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    start_epoch = args.start_epoch
    if args.resume != '':
        checkpoint = torch.load(args.checkpoint_path+args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Checkpoint '+args.resume+' loaded! Resume from epoch '+str(start_epoch))

    model.train()
    total_loss = 0
    total_accuracy = 0
    for epoch in range(start_epoch, start_epoch+args.num_epoch):
        start = time.time()
        for batch, data in enumerate(dataloader):
            optimizer.zero_grad()
            lr = get_lr(batch+epoch*args.steps_epoch, args.warmup_steps, args.learning_rate, args.steps_epoch*args.total_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            x = data['token_ids'][...,:-1].contiguous().to(args.device)
            y = data['token_ids'][...,1:].contiguous().to(args.device)
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
            }, args.checkpoint_path+'model_'+str(epoch)+'.pt')


