from transformers import PretrainedConfig
class LLMConfig(PretrainedConfig):
    model_type = 'ploract'
    #参数声明了类型，自动检测是否为int或float
    def __init__ (self,
                  dim: int = 512,
                  n_layers: int = 8,
                  n_heads: int = 8,
                  n_kv_heads: int = 8,      # 可以取2，就采用了多头注意力
                  vocab_size: int = 6400,   # 取决于训练好的tokenizer
                  hidden_dim: int = None,   # 通过dim计算 三分之八的dim
                  multiple_of: int = 64,    # 保证是64的整数倍
                  norm_eps: float = 1e-5,
                  max_seq_len: int = 1024,
                  rope_theta: int = 1e6,    # RoPE位置编码的base
                  dropout: float = 0.0      # 默认没有dropout
                  ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.dropout = dropout

