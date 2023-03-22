# minChatGPT

This is a custom project from Stanford CS224N Winter 2023 class. The goal of this project is to answer this question
> Will alignment from human feedback can also help small language models such as GPT-2?

And the answer is YES! With RLHF, evaluation shows that ChatGPT prefers the aligned GPT-2 outputs for 88% of times over the vanilla GPT-2 outputs. 
Please see the technical [report](./report.pdf) for more details.

Also, you can test minChatGPT in [Colab Notebook](https://colab.research.google.com/drive/1LR1sbWTyaNAmTZ1g1M2tpmU_pFw1lyEX?usp=sharing)

**Disclaimer**: 
1. This model has not been tested or evalauted against its safety. It may generate harmful or toxic content.
2. The demo is only meant to show how to improve small models with RLHF. The performance is not comparable with any conversation systems that are backed by large language models.
3. This is not an error free codebase! In fact there may be bugs here and there. Please make an issue if you have any questions.

# Poster
![alt text](Poster.png)

# Directory
```bash
src
  |_train_ppo.py # training script for PPO 
  |_train_rm.py # trianing script for Reward Model
  |_train_sft.py # training script for SFT model
  |_tariners.py # the actual training loops and other trainer utilities, such as saving states
  |_loss.py # loss functions used in different training
  |_main.py # some scratch code to quickly test something
  |_gpt.py # GPT-2 implementation with LoRA
  |_evaluate.py # evaluate the generation with ChatGPT
  |_dataset.py # multiple datasets definition
  |_tokenizer.py # tokenizers in a unified class
  |_llama.py # wish I could have more time to test with LLaMA
init_debian.sh # in case you need to initialize a debian system from scratch
requirements.txt # dependencies without PyTorch! Install your own pytorch 2.0 nightly.
```

# Acknowledgement
This project wouldn't been done without the help from:
1. [Stanford CS224N](https://web.stanford.edu/class/cs224n/), Professor Manning and the TAs
2. [nanoGPT](https://github.com/karpathy/nanoGPT)
3. [ColossalAI](https://github.com/hpcaitech/ColossalAI)
4. [OpenAI Baselines](https://github.com/openai/baselines)
5. [OpenAssistant](https://github.com/LAION-AI/Open-Assistant)
6. [Anthropic HH RLHF](https://github.com/anthropics/hh-rlhf)
7. And my project mentor Jesse Mu!
