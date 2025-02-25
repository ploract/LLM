import math

from torchaudio.models.wav2vec2.components import FeedForward

from Config import LLMConfig
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(dim))              # nn.Parameter即模型可学习参数
    def forward(self, x):
        return self.weights * (x.float() * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)).type_as(x)
# x是bf16
# torch.Tensor（PyTorch 库中的张量对象）有 float() 方法，用于将张量的数据类型转换为浮点类型
# keepdim 保证得到的张量与原来的张量尺寸一致，即(bsz, seq_len, 1)

def precompute_pos_cis(dim: int, theta: float = 1e6, end: int = int(32 * 1024)):        # end代表可以支持多远的位置编码，一般就是seq_len，这里的dim实际上是head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(0, end)
    freqs = torch.outer(t, freqs).float()                 # 计算t和freqs的外积，生成一个(end, head_dim / 2)大小的张量
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # torch.ones_like 生成一个和freqs尺寸相同的全1张量，torch.polar生成cos(m * theta) + i * sin(m * theta)的复数
    return pos_cis                                        # pos_cis 大小为 (seq_len, head_dim / 2)


def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]    # shape = (1, seq_len, 1, head_dim / 2)
        return pos_cis.view(*shape)                                                     # 改变 pos_cis 的尺寸

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))              # 先把xq变成尺寸为(batch_size, seq_len, n_heads, head_dim / 2, 2)的张量，再转化成(batch_size, seq_len, n_heads, head_dim / 2)的复数形式
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))              # (batch_size, seq_len, n_heads, head_dim / 2)
    pos_cis = unite_shape(pos_cis, xq_)

    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)                               # xq_ * pos_cis 得到 尺寸为 (batch_size, seq_len, n_heads, head_dim / 2) 的张量结果，
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)                               # 再经过torch.view_as_real得到 (batch_size, seq_len, n_heads, head_dim / 2, 2)，然后flatten把维度为3以及之后的铺平
    return xq_out.type_as(xq), xk_out.type_as(xk)                                       # 代码中的变量经过float()是fp32格式，复数运算是complex64格式，因此需要转换成bf16格式


def repeat_kv(x: torch.Tensor, n_rep:int):
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)                              # 把x从(bs, seq_len, n_kv_heads, head_dim)扩展成(bs, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim)                            # 原来的最后一个维度的向量扩展n_rep倍，例如(1, 2, 3, 4) -> (1, 2, 3, 4, 1, 2, 3, 4) //n_rep = 2, head_dim = 4
    )                                                                                  # 返回的尺寸为 (batch_size, seq_len, n_heads, head_dim)  n_heads = n_rep * n_kv_heads


class Attention(nn.Module):
    def __init__(self, args:LLMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # 这种情况下就是一对一的
        assert args.n_heads % self.n_kv_heads == 0                                      # 要保证query能平均分到k和v的组里
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads                                    # 把Q分成的组数，即每一组K，V头和 n_rep 组query的头做运算
        self.head_dim = args.dim // self.n_heads
        self.wq = nn.Linear(args.dim, self.n_heads * self.head_dim, bias = False)       # y = x @ W.T + b, bias = False 也相当于 b = 0
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias = False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float('-inf'))
        mask = torch.triu(mask, diagonal = 1)                                          # torch.triu(mask, diagonal=1) 会修改 mask，只保留上三角矩阵部分，并将对角线及其下方的元素置为零，其他部分保留为 -inf
        self.register_buffer('mask', mask, persistent = False)                         # 注册为不参与梯度运算的参数，persistent = False表明训练出来的模型不需要保存这个参数

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)      # Q 的尺寸为 (bsz, seq_len, n_heads, head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # K 的尺寸为 (bsz, seq_len, n_kv_heads, head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)   # V 的尺寸为 (bsz, seq_len, n_kv_heads, head_dim)
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)                   # 添加了 RoPE位置编码 的 Q 和 K
        if past_key_value is not None:                               # 使用到kv_cache的时候，seq_len = 1
            xk = torch.cat([past_key_value[0], xk], dim = 1)     # (bsz, 1 + past_seq_len, n_kv_heads, head_dim)
            xv = torch.cat([past_key_value[1], xv], dim = 1)     # (bsz, 1 + past_seq_len, n_kv_heads, head_dim)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (
            xq.transpose(1, 2),                              # (bsz, n_heads, seq_len, head_dim)
            repeat_kv(xk, self.n_rep).transpose(1, 2),       # (bsz, n_heads, new_seq_len, head_dim)
            repeat_kv(xv, self.n_rep).transpose(1, 2)        # (bsz, n_heads, new_seq_len, head_dim)
        )                                                               # 实际上，这里的 transpose 不会改变最后一个维度，例如：[1, 2, 3, 4]还是[1, 2, 3, 4]
        scores = (xq @ xk.transpose(-2, -1)) / (self.head_dim ** 0.5)   # 计算注意力分数，训练时 (bsz, n_heads, seq_len, seq_len) ，推理时且不是首个输入时 (bsz, n_heads, 1, new_seq_len)
        scores += self.mask[:, :, :seq_len, :seq_len]                   # 当使用了kv_chache时，seq_len = 1，会自动广播
        scores = F.softmax(scores.float(), dim = -1).type_as(scores)    # (bsz, n_heads, seq_len, seq_len)
        scores = self.attn_dropout(scores)                              # 注意力分数先经过一次dropout
        output = scores @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)       # (bsz, seq_len, dim)
        output = self.resid_dropout(self.wo(output))                    # output再经过一次dropout (bsz, seq_len, dim)
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)  # 三分之八的维度缩放

            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)  # 向上取整，保证hidden_dim能被64整除
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias = False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias = False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias = False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # x经过w1升维再经过激活函数得到的矩阵 与 x经过w3升维得到的矩阵 做哈达玛积 得到一个新的矩阵，最后再过w2降维并dropout
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))      # F.silu() 即为Swish 激活函数


class ploractBlock(nn.Module):
    def __init__(self, layer_id: int, config: LLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = self.dim // self.n_heads
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim, eps = config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps = config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(self, x, pos_cis, past_key_value = None, use_cache = False):
        h_attn, past_kv = self.attention(self.attention_norm(x), pos_cis, past_key_value = past_key_value, use_cache = use_cache)
        h = x + h_attn                                                    # x 经过 注意力机制 得到 残差            （第一次残差连接）
        out = h + self.feed_forward(self.ffn_norm(h))                     # x 先 norm ,再经过 ffn ,然后得到 残差   （第二次残差连接）
        return out, past_kv

class ploract(PreTrainedModel):
    config_class = LLMConfig
    def __init__(self, params: LLMConfig = None):
        self.params = params or LLMConfig()  # 如果没有提供 params，则默认使用 LLMConfig() 创建一个默认配置。
        super().__init__(self.params)        # PreTrainedModel 在构造时，主要的职责是：接收一个 config 对象并存储它。
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([ploractBlock(l, params) for l in range(self.n_layers)]) # n_layers层
        self.norm = RMSNorm(params.dim, eps = params.norm_eps)
        self.output = nn.Linear(self.params.dim, self.vocab_size, bias = False)
        self.token_embedding.weight = self.output.weight
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim = self.params.dim // self.params.n_heads, theta = params.rope_theta), persistent = False) # 保存模型时，该缓冲区会被忽略
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,             # input的尺寸是 (bsz, seq_len)，每个元素是一个 0 - vocabsize-1 的整数
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,                              # 默认false，推理阶段打开
                **args
                ):
        past_key_values = past_key_values or [None] * (self.n_layers)
        start_pos = args.get('start_pos', 0)                              # 如果args有就取，否则取0
        h = self.dropout(self.token_embedding(input_ids))                 # 经过embedding后得到词向量，然后再经过dropout -> (bsz, seq_len, dim)，推理情况下(bsz, 1, dim)
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]   # 取出从start_pos到start_pos + seq_length的元素
        past_kvs = []

        for l, layer in enumerate(self.layers):
            h, past_kv = layer(h, pos_cis, past_key_value = past_key_values[l], use_cache = use_cache) # past_key_values 来自上一次推理（即得到此次的输入的推理过程）
            past_kvs.append(past_kv)                                  #如果使用kv cache的话，每次past_kv的尺寸都会从(bsz, seq_len, dim) -> (bsz, seq_len + 1, dim)
        logits = self.output(self.norm(h))
        self.OUT.__setitem__('logits', logits)                   # 字典形式存储，(bsz, seq_len, vocab_size)
        self.OUT.__setitem__('past_key_values', past_kvs)        # 字典形式存储
        return self.OUT

    @torch.inference_mode()
    # 生成函数：支持流式生成与一次性生成
    def generate(self, input_ids, eos_token_id = 2, max_new_tokens = 1024, temperature = 0.75, top_p = 0.90,    # temperature
                 stream=False, rp=1., use_cache=True, pad_token_id=0, **args):
        return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

    # 内部流式生成函数
    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            # 首次调用或未使用缓存时，传入整个序列
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values = past_kvs, use_cache = use_cache, **args), False
            else:
                # 仅传入最后一个token，同时更新start_pos
                out = self(input_ids[:, -1:], past_key_values = past_kvs, use_cache = use_cache, start_pos = input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values  # 取出最后一步的logits及更新后的KV缓存，取出的logits的尺寸为 (bsz, vocab_size)

            logits[:, list(set(input_ids.tolist()[0]))] /= rp  # 对已经生成的token进行惩罚，防止重复生成

            logits /= (temperature + 1e-9) # 温度缩放, temperature越小，越严谨，越大，越有创造力

            if top_p is not None and top_p < 1.0: # 如果设置了top_p采样，则进行核采样处理
                sorted_logits, sorted_indices = torch.sort(logits, descending = True, dim = -1)  # 从大到小排序logits，同时得到indices的顺序
                sorted_probs = F.softmax(sorted_logits, dim=-1)                                  # 把 logits 过 softmax
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)                            # 计算前缀和
                sorted_indices_to_remove = cumulative_probs > top_p                              # 得到 bool 数组
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()       # 把 bool 数组 向右移动一格
                sorted_indices_to_remove[:, 0] = False                                           # 第一格补0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)   # 在dim = 1上，把sorted_indices_to_remove在索引sorted_indices上的元素放在新的矩阵上
                logits[indices_to_remove] = -float('Inf')                                        # 把sorted_indices_to_remove上为True的位置改为为-inf
            input_ids_next = torch.multinomial(F.softmax(logits, dim = -1), num_samples = 1)     # 根据采样后的概率分布选取下一个token, (bsz, 1)
            input_ids = torch.cat((input_ids, input_ids_next), dim = 1)                   # 将新token拼接到已有序列上

            yield input_ids[:, start:]                # 生成器返回新生成部分

            if input_ids_next.item() == eos_token_id: # 若生成的token为结束符，则停止生成
                break
