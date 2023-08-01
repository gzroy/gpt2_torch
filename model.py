import torch
from torch import nn
from torch.nn import functional as F
import math
import inspect

class MHA(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop, resid_pdrop):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.ln = nn.Linear(d_model, d_model*3)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.size()
        x_qkv = self.ln(x)
        q, k, v = x_qkv.split(self.d_model, dim=2)
        q = q.view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C//self.num_heads).transpose(1, 2)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_pdrop if self.training else 0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y

class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout):
        super().__init__()
        self.ln1 = nn.Linear(d_model, dff)
        self.ln2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        x = self.dropout(x)
        return x
    
class Block(nn.Module):
    def __init__(self, d_model, num_heads, dff, attn_pdrop, resid_pdrop, dropout):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, num_heads, attn_pdrop, resid_pdrop)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dff, dropout)

    def forward(self, x):
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ff(self.layernorm2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, vocab_size, d_model, block_size, embed_pdrop, num_heads, dff, attn_pdrop, resid_pdrop, dropout, num_layer):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, sparse=False)
        self.pos_embed = nn.Embedding(block_size, d_model, sparse=False)
        self.dropout_embed = nn.Dropout(embed_pdrop)
        #self.blocks = [Block(d_model, num_heads, dff, attn_pdrop, resid_pdrop, dropout) for _ in range(num_layer)]
        self.blocks = nn.ModuleList([Block(d_model, num_heads, dff, attn_pdrop, resid_pdrop, dropout) for _ in range(num_layer)])
        self.num_layer = num_layer
        self.block_size = block_size
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embed.weight = self.lm_head.weight
        self.layernorm = nn.LayerNorm(d_model)

        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) 
        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.dropout_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.layernorm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, -1, :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, block_size=512):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx