import os
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
from contextlib import nullcontext
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from model import ploract
from Config import LLMConfig
from dataset import PretrainDataset


def get_lr(current_step, total_steps, lr):                                            # 手动实现cos decay
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))  # 学习率逐渐从最大值变为最小值

def init_model(llm_config):
    tokenizer = AutoTokenizer.from_pretrained('./ploract_tokenizer')  # 加载分词器
    model = ploract(llm_config).to(args.device)                         # 初始化模型并移动到GPU上
    print(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')  # 打印模型参数量
    return model, tokenizer

def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction = "none")           # reduction = "none" 意思为算完loss后，不要那么快算平均，而是mask掉之后，这个时候有多少个token，就算多少个loss
    start_time = time.time()                                     # reduction 默认为 "mean"， 因此会自动求平均
    for step, (X, Y, loss_mask) in enumerate(train_loader):      # (X, Y, loss_mask)和PretrainDataset中的__getitem__有关
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)   # step 从 (0, iter_per_epoch - 1) 总步数为 args.epochs * iter_per_epoch
        for param_group in optimizer.param_groups:               # optimizer.param_groups 包含了各种层(token_embedding,attention中的各种层,norm,ffn等等)
            param_group['lr'] = lr                               # 动态改变学习率

        with ctx:                                         # 混合精度训练
            res = model(X)                                # res.logits的形状为 (batch_size, seq_len, vocab_size)
            loss = loss_fct(                              # Y 的形状为 (batch_size, seq_len)
                res.logits.view(-1, res.logits.size(-1)), # 变成一个二维矩阵，把batch_size和seq_len融入一维
                Y.view(-1)                                # 因为train_loader会自动增加一维batch_size，因此需要展平
            ).view(Y.size())                              # 交叉熵损失函数中，res.logits表示每个位置的预测分布，Y表示每个位置下一个token的索引
            loss = (loss * loss_mask).sum() / loss_mask.sum()  # loss_mask.sum()就是矩阵中1的个数，也就是参与了loss的计算的元素个数
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()                     # 反向传播
                                                          # 防止loss溢出，使用梯度缩放器，先scale，再unscale(具体做法为，先把位于BF16容易出现下溢为0的部分扩大一定倍数，计算完之后再缩小回去)
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)                    # unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) # 梯度裁剪
            scaler.step(optimizer)                        # 相当于 optimizer.step，更新模型参数
            scaler.update()                               # 每次参数更新后，都会调整用于梯度缩放的因子
            optimizer.zero_grad(set_to_none = True)       # 梯度清零，set_to_none表示不占显存，直接置为none

        if (step + 1) % args.log_step == 0:
            spend_time = time.time() - start_time
            print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,  # 当前epoch
                    args.epochs,  # 总的epoch数
                    step,  # 当前步数
                    iter_per_epoch,  # 每个epoch的迭代次数
                    loss.item() * args.accumulation_steps,  # 损失值，乘以累积步数
                    optimizer.param_groups[-1]['lr'],  # 当前学习率
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60  # 估计剩余时间
                )
            )
            if (wandb is not None):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 每隔一定步数保存模型
        if (step + 1) % args.save_step == 0:
            model.eval()  # 设置模型为评估模式
            ckp = f'{args.save_dir}/pretrain.pth'             # 保存模型的路径，这里采用的是一直把模型保存到同一个文件中，而不是保存很多个模型
            state_dict = model.state_dict()                   # 把model的参数取成字典的形式，然后保存到ckp文件里
            torch.save(state_dict, ckp)                       # 保存模型
            model.train()                                     # 恢复为训练模式




if __name__ == '__main__':    # 只有当前文件执行，才会执行，而不是被import后就执行
    parser = argparse.ArgumentParser()   # 创建命令行的参数解析器parser，可以直接在命令行中 pretrain.py --<参数名> 来添加参数

    # 添加各类参数，用于配置训练过程
    parser.add_argument("--save_dir", type=str, default="results")     # 保存结果的目录
    parser.add_argument("--epochs", type=int, default=2)               # 训练的轮数
    parser.add_argument("--batch_size", type=int, default=80)          # 每批次的样本数量
    parser.add_argument("--learning_rate", type=float, default=5e-4)   # 学习率
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")   # 设备类型，支持cuda或cpu
    parser.add_argument("--use_wandb", type = bool, default = True)            # 是否使用wandb进行日志记录
    parser.add_argument("--dtype", type=str, default="bfloat16")       # 数据类型，默认使用bfloat16
    parser.add_argument("--wandb_project", type=str, default="Ploract-Pretrain")  # wandb项目名称
    parser.add_argument("--num_workers", type=int, default=1)          # 数据加载时的CPU数
    parser.add_argument("--accumulation_steps", type=int, default=2)   # 梯度累积步数，累积多少步更新一次，变相扩大了batch_size，更新梯度时，为batch_size * K个样本的梯度的均值
    parser.add_argument("--grad_clip", type=float, default=1.0)        # 梯度裁剪阈值，防止梯度过大，提高训练稳定性（超过1.0的直接设置为1.0）
    parser.add_argument("--warmup_iters", type=int, default=0)         # warmup，即初始时，前多少步学习率缓慢线性增长
    parser.add_argument("--log_step", type=int, default=10)            # 每多少步记录一次日志
    parser.add_argument("--save_step", type=int, default=1000)         # 每多少步保存一次模型
    parser.add_argument('--max_seq_len', default=512, type=int)        # 输入的最大序列长度
    parser.add_argument("--data_path", type=str, default="pretrain.jsonl")  # 训练数据的路径

    args = parser.parse_args() # 解析命令行参数，后续需要调用参数只需要 args.<参数>
    lm_config = LLMConfig(max_seq_len = args.max_seq_len)

    args.save_dir = os.path.join(args.save_dir)                  # 使用路径对象保存
    os.makedirs(args.save_dir, exist_ok=True)                    # exist_ok 表示如果有的话就不需要新建文件夹了

    tokens_per_iter = args.batch_size * lm_config.max_seq_len    # 每次迭代处理的token数量
    torch.manual_seed(1337)                                      # 设置随机种子，保证训练可复现
    device_type = args.device

    # 设置wandb运行名称
    args.wandb_run_name = f"Ploract-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()   # torch.cuda.amp.autocast() torch自己封装的混合精度训练的编辑器
                                                                                 # 如果是CPU的话，就使用上下文编辑器，FP32
    if args.use_wandb:
        import wandb
 
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)  # 初始化wandb项目
    else:
        wandb = None  # 如果不使用wandb，设置为None

    model, tokenizer = init_model(lm_config)   # 初始化模型和分词器
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length = lm_config.max_seq_len)  # 加载训练数据集

    # 创建数据加载器
    train_loader = DataLoader(
        train_ds,
        batch_size = args.batch_size,
        pin_memory = True,  # 是否将数据复制到CUDA内存
        drop_last = False,  # 如果最后的数据凑不成一个batch，仍然保留数据（即不丢弃最后一批数据）
        shuffle = False,    # 不对数据进行乱序，因为如果发现Loss突出变化很快的话，容易定位出来数据
        num_workers = args.num_workers,  # 数据加载时使用的子线程数
    )

    #梯度缩放器的作用是在低精度（如 float16 或 bfloat16）训练时，动态调整梯度大小，避免因数值范围不足导致的训练不稳定问题
    scaler = torch.cuda.amp.GradScaler(enabled = (args.dtype in ['float16', 'bfloat16'])) # 初始化梯度缩放器（用于混合精度训练）
    optimizer = optim.AdamW(model.parameters(), lr = args.learning_rate)                  # 设置优化器
    iter_per_epoch = len(train_loader)     # 运行一个epoch需要的迭代次数，换句话说就是 样本总量/batch_size(除不尽则向上取整)


    for epoch in range(args.epochs):       # 开始训练
        train_epoch(epoch, wandb)          # 调用训练函数开始训练每个epoch