  # minChatGPT

Plan

- Reward Model Training w/ GPT2
  - The larger the model is, the higher test acc is
  - LoRA is better than fine-tuning last block, 68% vs 64%
  - LoRA is even better than fine-tuning all layers
  - LoRA rank 1 seems enough
  - Bigger batch size does help
  - Fine-tuning all layers is not stable
- SFT w/ GPT2
  - Overfits quickly
  - Generation are more like Human/Assitant dialouge 
  - Generation diverisity is worse after overfitting
- Fine-tune reward model with SFT model
  - Fine-tuned with LoRA
  - It helps a lot for GPT2 medium, 61%->67%
  - But it doesn't help for GPT2 xl, 68->68%
 
TODO:
- Implement PPO Trainer
- Need to figure out FSDP to fit multiple models into VRAM
- Compare PPO vs SFT. How to evaluate?
