import json
from torch.utils.data import Dataset
import torch

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length = 512):  # data_path路径存放了json文件
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(data_path)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding = 'utf-8') as f:
            for line in f:
                data = json.loads(line.strip())     # 每一行是一个json对象
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        #构建输入文本
        text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"      # 构建输入文本，添加<BOS>和<EOS>的token
        encoding = self.tokenizer(                  # 转换为分好词后的序列
            text,
            max_length = self.max_length,
            padding = "max_length",  # 填充到最大长度
            truncation = True,       # 超过最大长度的部分会被截断
            return_tensors = 'pt'    # 返回PyTorch的Tensor格式
        )
        input_ids = encoding.input_ids.squeeze()                    # encoding.input_ids一般是(1, seq_len)，tokenizer 默认处理单个文本样本，因此返回的 input_ids 是一个二维张量，第一维是批次大小（batch size），因为这里只处理单个文本，批次大小为 1
        loss_mask = (input_ids != self.tokenizer.pad_token_id)      # 防止padding参与损失计算，创建一个loss_mask，标记pad位置为0，非pad位置为1
        X = torch.tensor(input_ids[:-1], dtype = torch.long)        # 不包括结尾的 token，输入的是整数，因此用torch.long
        Y = torch.tensor(input_ids[1:], dtype = torch.long)         # 不包括首部的 <bos> token
        loss_mask = torch.tensor(loss_mask[1:], dtype = torch.long) # 计算loss时，pad位置不参与计算
        return X, Y, loss_mask

class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<s>assistant\n', add_special_tokens=False).input_ids  # .input_ids 提取'<s>assistant\n'的token id 列表
        self.eos_id = tokenizer('</s>\n', add_special_tokens=False).input_ids          # add_special_tokens=False 防止tokenizer自动添加特殊token，比如<s>和</s>

    def __len__(self):
        # 返回样本总数
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding = 'utf-8') as f:
            for line in f:
                data = json.loads(line.strip())     # strip()移除两端的空格，中间的不移除
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            # 偶数轮为用户，奇数轮为助手
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        # 使用分词器提供的模板方法构建对话提示
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,                                  # tokenize=False 作用: 控制是否将格式化后的文本转换为 Token ID 序列。
            add_generation_prompt=False                      # add_generation_prompt=False 作用: 控制是否在对话末尾添加生成提示符（即提示模型开始生成回复的标记）。
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):                                  # 遍历整个输入序列
            if input_ids[i: i + len(self.bos_id)] == self.bos_id:   # 检查当前位置是否匹配'<s>assistant\n'
                start = i + len(self.bos_id)                       # 跳过'<s>assistant\n'的部分
                end = start                                        # 从开始位置向后查找结束标记</s>
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1

                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)): # 将开始标记之后到结束标记位置之间的 token 标记为 1（参与损失计算）
                    loss_mask[j] = 1

                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)       # 更新索引：跳过整个对话部分（包括结束标记），开始寻找下一段需要参与loss计算的token
            else:
                i += 1
        return loss_mask


    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = self._create_chat_prompt(sample['conversations'])     # 利用对话轮次构建对话提示
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length] # 对对话提示进行编码，并限制最大长度
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids)) # 若不足最大长度则补齐 pad_token_id
        loss_mask = self._generate_loss_mask(input_ids)                # 根据输入 token IDs 生成动态的损失掩码

        # 构建训练数据：X 为输入序列（去掉最后一个 token），Y 为目标序列（去掉第一个 token）
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置
        return X, Y, loss_mask


