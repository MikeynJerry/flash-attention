import torch
from torch.backends.cuda import enable_mem_efficient_sdp
from torch.cuda.amp.autocast_mode import autocast
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import flash_attn_func
import math


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., sr_size=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.sr_size = sr_size
        if sr_size is not None and sr_size != 1:
            self.sr_k = DownsampleAvgPool(sr_size)
            self.sr_v = DownsampleAvgPool(sr_size)

    def forward(self, x, context=None, mask=None):
        context = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        if self.sr_size is not None and self.sr_size != 1:
            k = self.sr_k(k.permute(0, 2, 1).contiguous())
            v = self.sr_v(v.permute(0, 2, 1).contiguous())
            v = v.permute(0, 2, 1).contiguous()
            sim = torch.matmul(q, k) * self.scale
        else:
            sim = (q @ k.transpose(1, 2) * self.scale)

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)
            sim.masked_fill_(~mask, max_neg_value)

        sim = sim.softmax(dim=-1)

        out = sim @ v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class NativeCrossAttention(CrossAttention):
    def forward(self, x, context=None, mask=None):
        assert self.sr_size is None or self.sr_size == 1, "sr_size not implemented for native cross attention."
        context = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if mask is not None:
            mask = mask[:, None, None, :].bool().expand(-1, -1, q.size(2), -1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TorchAttention(CrossAttention):
    def forward(self, x, context=None, mask=None, is_causal=False, dropout_p=0.):
        assert self.sr_size is None or self.sr_size == 1, "sr_size not implemented for native cross attention."
        context = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if mask is not None:
            mask = mask[:, None, None, :].bool().expand(-1, -1, q.size(2), -1)

        L = q.size(-2)
        S = k.size(-2)

        attn_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0) if is_causal else mask
        attn_mask = attn_mask.to(dtype=torch.bfloat16).masked_fill(~attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
        attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        out = attn_weight @ v
        out = rearrange(out, 'b h n d -> b n (h d)')
        print("torch", out.flatten()[:10])
        return self.to_out(out)


class MyAttention(CrossAttention):
    def forward(self, x, context=None, mask=None):
        assert self.sr_size is None or self.sr_size == 1, "sr_size not implemented for native cross attention."
        context = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        if mask is not None:
            mask = mask[:, None, None, :].bool().expand(-1, -1, q.size(2), -1)

        print(q.size(), k.transpose(-2, -1).size(), v.size(), mask.size())
        print(mask)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class FlashAttention2(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., sr_size=None):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

        self.sr_size = sr_size
        if sr_size is not None and sr_size != 1:
            self.sr_k = DownsampleAvgPool(sr_size)
            self.sr_v = DownsampleAvgPool(sr_size)

    def forward(self, x, context=None, mask=None):
        assert self.sr_size is None or self.sr_size == 1, "sr_size not implemented for native cross attention."
        context = context if context is not None else x

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        print("flash", q.size(), k.size(), v.size())
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), (q, k, v))
        print("flash", q.size(), k.size(), v.size())

        if mask is not None:
            mask = mask[:, None, None, :].bool().expand(-1, -1, q.size(1), -1)
        print(mask.size())

        out = flash_attn_func(q, k, v, mask=mask)
        print("flash", out.flatten()[:10])
        out = rearrange(out, 'b n h d -> b n (h d)')
        return self.to_out(out)


device = torch.device("cuda:0")
torch.backends.cudnn.deterministic = True
torch.random.manual_seed(0)
torch.cuda.random.manual_seed_all(0)

native_attn = NativeCrossAttention(512, 4096).to(device, dtype=torch.bfloat16)
torch_attn = TorchAttention(512, 4096).to(device, dtype=torch.bfloat16)
my_attn = MyAttention(512, 4096).to(device, dtype=torch.bfloat16)
flash_attn = FlashAttention2(512, 4096).to(device, dtype=torch.bfloat16)
sd = native_attn.state_dict()
torch_attn.load_state_dict(sd)
my_attn.load_state_dict(sd)
flash_attn.load_state_dict(sd)

for np, tp in zip(native_attn.parameters(), torch_attn.parameters()):
    assert torch.equal(np, tp)

for np, mp in zip(native_attn.parameters(), my_attn.parameters()):
    assert torch.equal(np, mp)

for np, fp in zip(native_attn.parameters(), flash_attn.parameters()):
    assert torch.equal(np, fp)


x = torch.randn((8, 4096, 512), dtype=torch.bfloat16, device=device)
y = x.clone().detach()
context = torch.randn((8, 77, 4096), dtype=torch.bfloat16, device=device)
mask = torch.ones((8, 77), device=device, dtype=torch.int)
for i in range(mask.size(0)):
    idx = torch.randint(0, mask.size(1) - 1, (1,))
    mask[i, idx:mask.size(1)] = 0

with autocast(dtype=torch.bfloat16), torch.backends.cuda.sdp_kernel(
    enable_flash=False, enable_math=True, enable_mem_efficient=False
):
    torch.random.manual_seed(0)
    torch.cuda.random.manual_seed_all(0)
    native_out = native_attn(x, context=context, mask=mask)

# torch.random.manual_seed(0)
# torch.cuda.random.manual_seed_all(0)
# torch_out = torch_attn(x, context=context, mask=mask)
# 
# torch.random.manual_seed(0)
# torch.cuda.random.manual_seed_all(0)
# my_out = my_attn(x, context=context, mask=mask)

torch.random.manual_seed(0)
torch.cuda.random.manual_seed_all(0)
flash_out = flash_attn(y, context=context, mask=mask)


# print(native_out.size(), torch_out.size(), my_out.size())
# 
# torch.set_printoptions(precision=8)
print("native", native_out.flatten()[:10])
# print("torch", torch_out.flatten()[:10])
# print("mine", my_out.flatten()[:10])
print("flash", flash_out.flatten()[:10])
# print("native vs torch", torch.allclose(native_out, torch_out))
# print("mine vs torch", torch.allclose(my_out, torch_out))
#print("diff", torch.abs(torch_out-native_out))

"""
NativeCrossAttention(
  (to_q): Linear(in_features=512, out_features=512, bias=False)
  (to_k): Linear(in_features=4096, out_features=512, bias=False)
  (to_v): Linear(in_features=4096, out_features=512, bias=False)
  (to_out): Sequential(
    (0): Linear(in_features=512, out_features=512, bias=True)
    (1): Dropout(p=0.0, inplace=False)
  )
) torch.Size([8, 3840, 512]) torch.Size([8, 77, 4096]) torch.Size([8, 77])
NativeCrossAttention(
  (to_q): Linear(in_features=512, out_features=512, bias=False)
  (to_k): Linear(in_features=4096, out_features=512, bias=False)
  (to_v): Linear(in_features=4096, out_features=512, bias=False)
  (to_out): Sequential(
    (0): Linear(in_features=512, out_features=512, bias=True)
    (1): Dropout(p=0.0, inplace=False)
  )
) torch.Size([8, 3840, 512]) torch.Size([8, 77, 4096]) torch.Size([8, 77])
"""