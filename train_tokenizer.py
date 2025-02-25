import random
import json
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer
)

import os

def train_tokenizer():
    def read_texts_from_jsonl(file_path):
        with open(file_path, "r", encoding = "utf-8") as f:
            for line in f:
                data = json.loads(line)
                yield data["text"]
        data_path = "pretrain.jsonl"

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space = False)

        special_tokens = {"<ukn>", "<s>", "</s>"}

        trainer = trainers.BpeTrainer(
            vocab_size = 6400,
            special_tokens = special_tokens,
            show_progress = True,
            initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
        )
        texts = read_texts_from_jsonl(data_path)

        tokenizer.train_from_iterator(texts, trainer = trainer)

        tokenizer.decoder = decoders.ByteLevel()
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2

        tokenizer_dir = "./ploract_tokenizer"
        os.makedirs(tokenizer_dir, exist_ok = True)
        tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
        tokenizer.model.save("./ploract_tokenizer")

        config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": False,
            "added_tokens_decoder":{
                "0":{
                    "content": "<unk>",
                    "lstrip": False,
                    "normalized": False,
                    "rstrip": False,
                    "single_word": False,
                    "special": True
                }
            }
        }



        with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), ) as f:
            json.dump()



