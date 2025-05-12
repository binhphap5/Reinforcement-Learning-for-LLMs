# Preference-Tuning-with-LLMs

This project describes two approaches for aligning large language models to human preferences: **Proximal Policy Optimization (PPO)** and **Direct Preference Optimization (DPO)**.

---

## 1. Proximal Policy Optimization (PPO)

- **Base model**: GPT-2  
- **Reward model**: PhoBERT-based classifier fine-tuned to predict “negative” vs. “neutral” and "positive" student feedback.  
- **Goal**: Encourage GPT-2 to generate *negative student feedback* (e.g. "môn này khó quá", "phòng học nóng nực", ...)  
- **Pipeline**:  
  1. **Rollout**: Sample candidate feedback responses from GPT-2.  
  2. **Reward scoring**: Pass each response through the PhoBERT classifier to get a scalar reward:
     - `r(response) = +z (z is the logit value after the classifier head)` if classified as “negative”  
     - `r(response) = 0` otherwise  
  3. **PPO update**:  
     ```python
     for batch in feedback_prompts:
         # 1) Sample actions and compute log-probs under current policy π_θ
         responses, logp_old = policy.sample(batch)
         # 2) Compute rewards via reward_model
         rewards = reward_model(responses)
         # 3) Compute PPO loss with clipped objective
         loss = ppo_loss(logp, logp_old, rewards, clip=0.2)
         # 4) Backpropagate & update GPT-2 policy parameters θ
         loss.backward()
         optimizer.step()
     ```
  4. **Outcome**: GPT-2 becomes specialized at generate **Negative** feedback rather than neutral or overly general comments.

---

## 2. Direct Preference Optimization (DPO)

- **Base model**: Qwen 2.5-3B-Instruct  
- **Data**: `vi-alpaca-preference` preference pairs  
- **Goal**: Increase the probability of producing the “chosen” response over the “rejected” response directly, without a separate reward model.  
- **Training loop**:  
  ```python
  for x, y_chosen, y_rejected in preference_dataset:
      score_ch = model.logprob(y_chosen, x)
      score_rj = model.logprob(y_rejected, x)
      """ In Pytorch, we are using Gradient Descent instead of Gradient Ascent.
      That means we have to minimize the "-log" instead of maximize "log". """
      loss = -torch.log(torch.sigmoid(score_ch - score_rj))
      loss.backward()
      optimizer.step()
- Example preference pair: 
  "question": "Xác định và sửa lỗi ngữ pháp.\n\nTôi đã đi đến cửa hàng.",
  "chosen":  "Không có lỗi ngữ pháp. Câu này đã chính xác.",
  "rejected": "Câu này không có lỗi ngữ pháp."

