from gpt import GPT, GPTRewardModel, HFGPTRewardModel
from configs import get_configs
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


def generate_gpt2(model, prompt, device, samples=2):
    model.eval()
    model.to(device)
    max_new_tokens = 50
    temperature = 0.9
    top_k = 200
    x, decode = prepare_gpt2_input(prompt, device)

    for k in range(samples):
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
        cfg = get_configs("gpt2-xl")
        model = GPT.from_pretrained(cfg)
        generate_gpt2(model, prompt, device, samples=10)
    elif task == "unwrap_gpt":
        prompt = """Human: Hello, my name is Kate. What is your name?
Assitant:"""
        ckpt_file = "1678083261_step40000.pt"
        new_file = "original_sft_" + ckpt_file
        ckpt_path = "./runs/sft_1678083261/"
        cfg = get_configs("gpt2-xl")

        model = GPT.from_checkpoint(cfg, ckpt_path + ckpt_file, compile=True)
        mods = model.modules()
        next(mods)
        inner_model = next(mods)

        generate_gpt2(inner_model, prompt, device)
        checkpoint = torch.load(ckpt_path + ckpt_file, map_location="cpu")
        torch.save(
            {
                'step': checkpoint['step'],
                'model_state_dict': inner_model.state_dict(),
                'optimizer_state_dict': checkpoint['optimizer_state_dict'],
            }, ckpt_path + new_file)
    elif task == "gpt_sft":
        prompt = """Human: You are an asshole! You are an idiot!
Assitant:"""
        cfg = get_configs("gpt2-medium")

        model = GPT.from_checkpoint(
            cfg,
            "./runs/ppo_gpt2medium-batch1-fp16_202303170754/ppo_gpt2medium-batch1-fp16_202303170754_actor_step50.pt")
        # model = GPT.from_checkpoint(
        #     cfg, "./runs/sft_1678085469/original_sft_1678085469_step100000.pt")
        generate_gpt2(model, prompt, device, samples=10)
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
    elif task == "test_loss":
        from loss import KPairwiseLoss
        loss_func = KPairwiseLoss()
        scores = torch.tensor([[0.8, 0.4], [0.5, 0.6]])
        loss = loss_func(scores)
        print(loss)
    elif task == "load_fsdp":
        cfg = get_configs("gpt2-medium")
        model = GPT.from_checkpoint(
            cfg, "/home/yanjia/Code/minChatGPT/src/rm_1678263092_final.pt")
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
