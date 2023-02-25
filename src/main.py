from models import GPT
import torch
import tiktoken

device = 'cuda'
max_new_tokens = 20
num_samples = 8
temperature = 0.9
top_k = 200
prompt = "Hello, my name is"

model = GPT.from_pretrained()
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

indices = encode(prompt)
x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])
for k in range(num_samples):
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    print(decode(y[0].tolist()))
    print('---------------')
