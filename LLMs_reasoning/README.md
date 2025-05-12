# Improving-Reasoning-Capability-of-LLMs-with-Reinforcement-Learning

This project investigates how to enhance large language models’ (LLMs) reasoning capabilities via Reinforcement Learning (RL). We build on the **DeepSeek** framework and introduce **Group Relative Policy Optimization (GRPO)**, a novel RL algorithm tailored for instruction-following and chain‐of‐thought tasks.

A typical LLMs training pipeline nowadays:
![image](https://github.com/user-attachments/assets/eabb91a3-526e-4694-bd25-c48c1c854275)
---

## 🚀 Features

- **GRPO**: A variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO.
- **QLoRA Integration**: Combines quantized LoRA fine-tuning for parameter-efficient adaptation
- **Chain-of-Thought (CoT)**: Leverages CoT prompts to decompose complex reasoning
- **Benchmarks**: Evaluation on the vi_gsm8k dataset (100 sampled problems)

## 📊 Benchmark Results

### Table I: Perplexity, SacreBLEU and Exact Match on 100 sampled vi_gsm8k problems

| Model                                  | Perplexity | SacreBLEU | Exact Match (Accuracy) |
|----------------------------------------|-----------:|----------:|------------:|
| mT5-small                              |     3.9578 |   20.1683 |       0.050 |
| Encoder–Decoder (scratch)              |    11.7095 |   39.9151 |       0.020 |
| GPT2-medium                            |     5.1044 |   30.1349 |       0.030 |
| Decoder-only (scratch)                 |    19.5144 |   27.5626 |       0.010 |
| Qwen 2.5-3B (Instruction + CoT, QLoRA) |     1.7495 |   21.1412 |       0.470 |
| **Qwen 2.5-3B (GRPO, QLoRA)**           |      N/A   |   31.5785 |       **0.670** |

### Table II: ROUGE-1/2/L F1 scores on the same 100 problems

| Model                                  | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 |
|----------------------------------------|-----------:|-----------:|-----------:|
| mT5-small                              |      0.4530 |      0.2061 |      0.3332 |
| Encoder–Decoder (scratch)              |      0.4548 |      0.1955 |      0.3222 |
| GPT2-medium                            |      0.4689 |      0.2245 |      0.3478 |
| Decoder-only (scratch)                 |      0.4649 |      0.1932 |      0.3241 |
| Qwen 2.5-3B (Instruction + CoT, QLoRA) |      0.5971 |      0.3259 |      0.4308 |
| **Qwen 2.5-3B (GRPO, QLoRA)**           |      0.5475 |      0.2972 |      0.3795 |


---

