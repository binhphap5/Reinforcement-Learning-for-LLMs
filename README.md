# 🤖 Reinforcement Learning for Large Language Models (LLMs)

This repository combines two independent research directions that use **Reinforcement Learning (RL)** to enhance the capabilities and alignment of Large Language Models (LLMs).

---

## 🧠 Structure

```
.
├── LLMs_reasoning/        # Enhancing LLM reasoning with GRPO
│   └── README.md          # Detailed explanation of GRPO-based approach
│
├── LLMs_preference/       # Aligning LLMs to user preferences with PPO & DPO
│   └── README.md          # Detailed explanation of PPO and DPO training
│
└── README.md              # (You are here) High-level summary of the entire project
```

---

## 1. 🧮 Reasoning with GRPO

Located in: `LLMs_reasoning/`

- Implements **Group Relative Policy Optimization (GRPO)**, a variant of PPO from DeepSeek.
- Targeted at improving LLM performance on **math reasoning tasks** using **Chain-of-Thought** prompting.
- Finetunes **Qwen 2.5B** with **QLoRA** for efficiency.
- Evaluation done on a Vietnamese subset of GSM8K (`vi_gsm8k`).

📈 Highlights:
- Achieves **+20% accuracy** improvement over standard instruction tuning on math problems.

➡️ See [`LLMs_reasoning/README.md`](./LLMs_reasoning/README.md) for full details.

---

## 2. 🗣️ Preference Alignment with PPO & DPO

Located in: `LLMs_preference/`

- Uses **PPO** to fine-tune GPT-2 for generating *negative* student feedback using a reward model (PhoBERT).
- Uses **DPO** to fine-tune Qwen 2.5B directly from user preference pairs without a reward model.
- Demonstrates alignment of model outputs to human-intended choices.

📌 Use Cases:
- Automatically bias outputs toward desired sentiment (e.g., negative feedback).
- Train models to prefer "chosen" over "rejected" completions from instruction datasets.

➡️ See [`LLMs_preference/README.md`](./LLMs_preference/README.md) for full details.

---

## 🛠️ Tech Stack

- PyTorch, Hugging Face Transformers
- PEFT (QLoRA) for memory-efficient finetuning
- RL Algorithms: GRPO, PPO, DPO

---

## 🚀 Goals

- Understand and implement RL algorithms tailored for LLMs.
- Improve **reasoning** and **alignment** without relying on fully supervised fine-tuning (which always require many high quality instruction datasets).
- Provide reproducible baselines for RL in Vietnamese-language LLMs.

---


