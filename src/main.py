from gpt import GPT, GPTRewardModel, HFGPTRewardModel, get_configs
from llama import LLaMA, ModelArgs
from dataset import AnthropicHHRLHFDataset, DahoasRMStaticDataset, DahoasSFTStaticDataset, EYLSFTStaticDataset
import torch
import tiktoken
import click
from tokenizer import LLaMATokenizer
from trainers import RewardModelTrainer, SFTTrainer
import json


def prepare_gpt2_input(prompt, device):
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
    return x, decode


def generate_gpt2(model, prompt, device):
    model.eval()
    model.to(device)
    max_new_tokens = 30
    num_samples = 2
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)

    for k in range(num_samples):
        y = model.generate(x,
                           max_new_tokens,
                           temperature=temperature,
                           top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')


@click.command()
@click.option('--task', '-t')
def main(task):
    device = 'cuda'

    if task == 'gpt':
        prompt = """Human: Hello, my name is Kate. What is your name?
Assitant:"""
        model = GPT.from_pretrained("gpt2-medium")
        generate_gpt2(model, prompt, device)
    elif task == "unwrap_gpt":
        prompt = """Human: Hello, my name is Kate. What is your name?
Assitant:"""
        ckpt_file = "sft_1678085469_step60000.pt"
        new_file = "original_" + ckpt_file
        ckpt_path = "./runs/sft_1678085469/"
        model = GPT.from_checkpoint("gpt2-medium",
                                    ckpt_path + ckpt_file,
                                    compile=True)
        generate_gpt2(model, prompt, device)
        mods = model.modules()
        next(mods)
        inner_model = next(mods)
        checkpoint = torch.load(ckpt_path + ckpt_file, map_location="cpu")
        torch.save(
            {
                'step': checkpoint['step'],
                'model_state_dict': inner_model.state_dict(),
                'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            }, ckpt_path + new_file)
    elif task == "gpt_sft":
        prompt = """Human: Hello, my name is Kate. What is your name?
Assitant:"""

        model = GPT.from_checkpoint(
            "gpt2-medium",
            "./runs/sft_1678085469/original_sft_1678085469_step100000.pt")
        generate_gpt2(model, prompt, device)
    elif task == 'llama':
        num_samples = 3
        max_new_tokens = 300
        prompt = """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:"""
        ckpt_path = '../models/7B/consolidated.00.pth'
        params_path = '../models/7B/params.json'
        tokenizer_path = '../models/tokenizer.model'
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        tokenizer = LLaMATokenizer(model_path=tokenizer_path)
        with open(params_path) as fp:
            params = json.loads(fp.read())
        model_args: ModelArgs = ModelArgs(max_seq_len=1024,
                                          max_batch_size=32,
                                          **params)
        model_args.vocab_size = tokenizer.n_words

        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = LLaMA(model_args)
        # model.to(device)
        # model.eval()
        torch.set_default_tensor_type(torch.FloatTensor)
        model.load_state_dict(checkpoint, strict=False)

        with torch.inference_mode():
            for k in range(num_samples):
                x = torch.tensor(tokenizer.encode(prompt, bos=True, eos=False),
                                 dtype=torch.long,
                                 device=device)[None, ...]
                y = model.generate(x, max_new_tokens)
                print(tokenizer.decode(y[0].tolist()))
                print('---------------')
    elif task == "reward":
        prompt = """Human: Hello, my name is Kate. What is your name?
Assitant:"""
        cfg = get_configs("gpt2-medium/dropout")
        x, _ = prepare_gpt2_input(prompt, device)
        rm = GPTRewardModel.from_pretrained(cfg)
        rm.eval()
        rm.to(device)
        score = rm(x)
        print(score)
    elif task == "reward_sft":
        prompt = """Human: Hello, my name is Kate. What is your name?
Assitant:"""
        cfg = get_configs("gpt2-medium")
        x, _ = prepare_gpt2_input(prompt, device)
        rm = GPTRewardModel.from_backbone_checkpoint(
            cfg, "./runs/sft_1678085469/original_sft_1678085469_step100000.pt")
        rm.eval()
        rm.to(device)
        score = rm(x)
        print(score)
    elif task == "dataset":
        from datasets import load_dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split='train')
        print(dataset[0])
    elif task == "sft":
        cfg = get_configs("gpt2-medium/dropout")
        model = GPT.from_pretrained(cfg)
        train_ds = EYLSFTStaticDataset(block_size=1024,
                                       split='train',
                                       max_examples=None,
                                       tokenizer_name="tiktoken/gpt2")
        test_ds = EYLSFTStaticDataset(block_size=1024,
                                      split='test',
                                      max_examples=None,
                                      tokenizer_name="tiktoken/gpt2")
        trainer = SFTTrainer(device,
                             model,
                             train_ds,
                             test_ds,
                             batch_size=2,
                             max_steps=120000,
                             cfg=cfg,
                             finetune_method=False)
        trainer.fit()
    elif task == "train_rm_sft":
        cfg = get_configs("gpt2-medium/lora")
        rm = GPTRewardModel.from_backbone_checkpoint(
            cfg, "./runs/sft_1678085469/original_sft_1678085469_step100000.pt")
        train_ds = DahoasRMStaticDataset(block_size=1024,
                                         split='train',
                                         max_examples=None,
                                         tokenizer_name="tiktoken/gpt2")
        test_ds = DahoasRMStaticDataset(block_size=1024,
                                        split='test',
                                        max_examples=None,
                                        tokenizer_name="tiktoken/gpt2")
        trainer = RewardModelTrainer(device,
                                     rm,
                                     train_ds,
                                     test_ds,
                                     batch_size=2,
                                     total_epochs=1,
                                     cfg=cfg,
                                     finetune_method="lora")
        trainer.fit()
    elif task == "train_rm":
        cfg = get_configs("gpt2-medium/lora")
        rm = GPTRewardModel.from_pretrained(cfg)
        train_ds = DahoasRMStaticDataset(block_size=1024,
                                         split='train',
                                         max_examples=None,
                                         tokenizer_name="tiktoken/gpt2")
        test_ds = DahoasRMStaticDataset(block_size=1024,
                                        split='test',
                                        max_examples=None,
                                        tokenizer_name="tiktoken/gpt2")
        trainer = RewardModelTrainer(device,
                                     rm,
                                     train_ds,
                                     test_ds,
                                     batch_size=1,
                                     total_epochs=1,
                                     cfg=cfg,
                                     finetune_method="lora")
        trainer.fit()
    elif task == "test_loss":
        from loss import KPairwiseLoss
        loss_func = KPairwiseLoss()
        scores = torch.tensor([[0.8, 0.4], [0.5, 0.6]])
        loss = loss_func(scores)
        print(loss)
    elif task == "test_tokenizer":
        from dataset import TiktokenTokenizer
        from transformers import GPT2Tokenizer, GPT2TokenizerFast
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token)
        print(
            tokenizer("How are you?<|endoftext|>",
                      max_length=20,
                      padding="max_length",
                      truncation=True,
                      return_tensors="pt"))
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print(tokenizer.pad_token)
        print(
            tokenizer("How are you?",
                      max_length=20,
                      padding="max_length",
                      truncation=True,
                      return_tensors="pt"))

        tokenizer = TiktokenTokenizer('gpt2')
        print(tokenizer.pad_token)
        print(
            tokenizer("How are you?",
                      max_length=20,
                      padding="max_length",
                      truncation=True,
                      return_tensors="pt"))


if __name__ == "__main__":
    main()
