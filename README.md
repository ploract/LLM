# 基于LLaMA3大模型构建项目

从零构建千万参数规模的中文大语言模型，支持预训练、指令微调和推理蒸馏全流程。基于类LLaMA3架构设计，专注中文场景优化。

## 模型特点
🦙 **轻量高效** - 千万参数规模实现流畅问答  
🧠 **思维链能力** - 通过R1蒸馏实现慢思考机制  
🇨🇳 **中文优化** - 专为中文设计的BBPE分词器  
⚡ **高效训练** - 混合精度+梯度累积优化

## 模型架构
### 核心技术
- **基础组件**：从零实现 RMSNorm/SwiGLU/RoPE
- **注意力机制**：分组多头注意力（Grouped Query Attention）
- **位置编码**：旋转位置编码（RoPE）
- **生成策略**：温度采样+Top-p采样


## 训练流程
1. **预训练阶段**（Pretrain）
   - 数据：匠数大模型数据集（过滤优化）
   - 目标：语言建模（CLM）
   - 技术：混合精度训练，梯度累积策略

2. **指令微调**（SFT）
   - 数据：匠数SFT数据集（512/1024双阶段）
   - 特性：指令loss掩码机制
   - 规模：10B token训练量

3. **R1推理蒸馏** (distill)
   - 数据：Deepseek-R1中文蒸馏数据
   - 方法：黑盒蒸馏策略
   - 效果：增强推理链生成能力

  ## 代码结构
```text
├── train_tokenizer.py      # 分词器训练
├── Config.py               # 模型超参数配置
├── model.py                # 核心模型架构实现
├── dataset.py              # 训练数据加载
├── pretrain.py             # 预训练主程序
├── SFT.py                  # 基础指令微调（512 tokens上下文）
├── SFT_long.py             # 长文本指令微调（1024 tokens上下文）
├── distill.py              # R1推理蒸馏实现
├── model_eval.py           # 模型交互式测试脚本
├── pretrain.log            # 预训练日志
├── SFT.log                 # SFT日志
├── SFT_long.log            # SFT日志
├── distill.log             # R1蒸馏日志
└── README.md
```

## 实验记录
- 环境：
  - python版本：3.12
  - torch版本：2.5.1+cu124
  - transformers版本：4.49.0
  - 单机1卡4090/24G
### Pretrain
4090/24G，跑2个epoch，每个epoch约80min，总时长<3h

  - 小实验：上述都刚刚跑到总step数的10%左右就停掉了
    - 深紫色：bs=80，梯度累积=2，wramup=False，lr=5e-4
    - 橘黄色：bs=80，梯度累积=4，warmup=True（ratio=0.03），lr=5e-4
    - 青色：bs=80，梯度累积=4，warmup=True（ratio=0.1），lr=5e-4
    - 浅紫色：bs=80，梯度累积=4，warmup=True（ratio=0.1），lr=5e-5
![image](https://github.com/user-attachments/assets/22a36386-b696-4293-858d-4df8efc37545)

  - 蓝色：bs=80，梯度累积=4
  - 绿色：bs=84，梯度累积=8
![image](https://github.com/user-attachments/assets/a03383f7-5681-4d92-9655-841266dbeccb)


  - 绿色：学习率0.004
  - 橙色：学习率0.001
![image](https://github.com/user-attachments/assets/04dbead1-f0e2-4ff4-b8fc-5ddab05a8679)

从收敛情况来看，等效batch_size=160(batch_size * gradient_accumlation)左右是个不错的实践.
- 最终选择：epochs=2 ,batch_size=84, 梯度累积=2 ，lr=5e-4, warmup=None
  - 显存峰值：23G/24G（利用率还是比较高的）
![W B Chart 2025_2_25 20_53_20](https://github.com/user-attachments/assets/1d4fc24c-860b-4dcb-b131-398bcbc18edc)
![W B Chart 2025_2_25 20_54_42](https://github.com/user-attachments/assets/0b0d8479-005f-4c87-ad0c-4590e1c3c149)



### SFT
SFT的代码大量继承了Pretrain的代码，仅仅数据加载做了改变，SFT类数据集定义参考dataset.py文件

- SFT数据
  - sft512.jsonl(7.1G)，由匠数科技的SFT数据(24G)清洗而成，筛选出了总长度小于512的部分

  - 数据格式为：
```text
 {
    "conversations": [
        {"role": "user", "content": "你是谁"},
        {"role": "assistant", "content": "我是SbongeBob"},
        {"role": "user", "content": "再见"},
        {"role": "assistant", "content": "再见！"}
    ]
}
```
- 使用sft_512.jsonl数据跑，单个epoch时间约为6.7h，epochs=1 ,batch_size=84, 梯度累积=2 ，lr=5e-4, warmup=None
![W B Chart 2025_2_25 21_00_21](https://github.com/user-attachments/assets/69a64dc1-74f2-48e8-a7e8-8ddb32a1b167)
![W B Chart 2025_2_25 21_00_43](https://github.com/user-attachments/assets/8d2f54b3-1338-4f51-a8bb-b703fbbb7d35)


- 使用sft_1024.jsonl 训练，epochs=1,batch_size=28，lr=5e-4 ,梯度累积=8，max_seq_len=1024, warmup=None
![W B Chart 2025_2_25 21_07_44](https://github.com/user-attachments/assets/9c29c2d0-713c-469f-b3c7-bab5c49ba23f)
![W B Chart 2025_2_25 21_07_48](https://github.com/user-attachments/assets/b232908c-16cb-452b-91f9-db46f7b5566f)

### Distill

这里我们采用黑盒蒸馏方法对拥有长文能力的model进行蒸馏，后面会对比长文能力的有无对蒸馏效果的影响。
- R1的输出格式一般采用在<think> 思考内容 </think> <answer> 回答</answer>的格式来产生慢思考过程。
- 结合开源社区的结论：R1的纯基于强化学习无法在<的3b模型上复现，因此我们优先采用蒸馏方法。
- 蒸馏数据使用r1_1024.jsonl(360M)，格式和SFT数据相似，如图：
```text
{"conversations":
[{"role": "user",
 "content": "请帮我写一个古风诗句，表达一种淡淡的哀愁和思念。"}, 
 {"role": "assistant", 
"content": "<think>\n好吧，用户让我帮他写一个古风诗句，主题是淡淡的哀愁和思念。首先，我需要理解他的需求。他可能希望这句诗能传达一种难以言喻的悲伤，但又不失古风的意境，这可能用于文学创作或者表达情感。\n\n我应该考虑古风诗的结构和常用词汇。古风诗常常使用自然意象，比如山水、 Places、四季等，来营造氛围。哀愁和思念可能涉及离别、思念、痛苦等情感。\n\n接着，我需要选择合适的词语来表达这些情感。比如“无尽有”可以表达无法言喻的悲伤，“空字”作为思念的代名词，这样既简洁又富有情感。如果季节有变化，换个季节会更有变化。\n\n然后是描绘场景，比如“天涯冷月”强调southenly的寒冷，增添夜晚的氛围，增加诗意。再加“孤影凄凄照”来表达内心的思念，最后一句“苦耐心犹未去”直接点明哀愁。\n\n最后，检查句子的对仗和节奏，确保流畅自然，符合古风的韵律。这样组合起来，应该能够满足用户的需求。\n</think>\n
<answer>\n无尽有，空字，若无云处。天涯冷月，孤影凄凄照，苦耐心犹未去。\n</answer>"}]}
```
- 鉴于我们的tokenizer对<think></think>编码效率低，需要4个token，因此模型对学习这种范式会略显困难，为了优先学习这种范式，我们会手动加大这些token的损失惩罚。

使用r1_1024.jsonl数据集进行训练，使用不同模型基座进行了两个版本的训练
- 使用SFT_1024.pth作为基座训练（经过1024数据训练）epochs=3，batch_size=30, lr=1e-6, 梯度累积=8，max_seq_len=1024, warmup=None
![W B Chart 2025_2_25 21_25_08](https://github.com/user-attachments/assets/c866e5ff-6df4-4899-aed0-f9fcdf0369c9)
![W B Chart 2025_2_25 21_25_13](https://github.com/user-attachments/assets/dc453c77-e3dd-4fc5-8bf2-fdc48aa8fd82)



## 推理
- 直接通过修改eval_model.py加载相应模型
- python eval_model.py --model_mode 2
  
![image](https://github.com/user-attachments/assets/c6ee32bf-33c2-46b5-9caa-436a1c61c2ab)

