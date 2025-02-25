import math
from Config import LLMConfig
from model import ploract
import argparse
import torch
from transformers import AutoTokenizer

def init_model(args):
     tokenizer = AutoTokenizer.from_pretrained('./ploract_tokenizer')
     if args.model_mode == 0:
          ckp = './results/pretrain.pth'
     elif args.model_mode == 1:
          ckp = './results/SFT.pth'
     else:
          ckp = './results/distill.pth'
     model = ploract(LLMConfig(max_seq_len = args.max_seq_len))
     state_dict = torch.load(ckp, map_location = args.device)
     model.load_state_dict({k:v for k,v in state_dict.items() if 'mask' not in k}, strict = False)         # mask不需要保存在model中
     print(f'模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
     return model.eval().to(args.device), tokenizer

def main():
     parser = argparse.ArgumentParser()
     parser.add_argument('--save_dir', default='results', type=str)
     parser.add_argument('--temperature', default=0.85, type=float)
     parser.add_argument('--top_p', default=0.85, type=float)
     parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
     parser.add_argument('--max_seq_len', default=8192, type=int)
     parser.add_argument('--history_cnt', default=0, type=int)
     parser.add_argument('--stream', default=True, type=bool)
     parser.add_argument('--model_mode', default=2, type=int, help="0: 预训练模 , 1: SFT-Chat 模型 , 2: Distill 模型")
     args = parser.parse_args()
     model, tokenizer = init_model(args)
     messages = []
     while True:
          # 获取用户输入
          prompt = input('👶: ')  # 手动输入对话内容

          messages = messages[-args.history_cnt:] if args.history_cnt else []
          messages.append({"role": "user", "content": prompt})

          new_prompt = tokenizer.apply_chat_template(
               messages,
               tokenize=False,
               add_generation_prompt=True
          )[-args.max_seq_len + 1:] if args.model_mode != 0 else (tokenizer.bos_token + prompt)
          print('new_prompt:', new_prompt)

          with torch.no_grad():
               x = torch.tensor(tokenizer(new_prompt)['input_ids'], device=args.device).unsqueeze(0)
               outputs = model.generate(
                    x,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=args.max_seq_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stream=True,
                    pad_token_id=tokenizer.pad_token_id,
                    rp=1.3
               )

               print('🤖️: ', end='')
               try:
                    history_idx = 0
                    for y in outputs:
                         answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                         if (answer and answer[-1] == '�') or not answer:
                              continue
                         print(answer[history_idx:], end='', flush=True)
                         history_idx = len(answer)
               except StopIteration:
                    print("No answer")
               print('\n')

          messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
     main()
