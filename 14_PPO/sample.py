import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional

class PPOTrainer:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        optimizer: optim.Optimizer,
        clip_epsilon: float = 0.2,
        entropy_beta: float = 0.01,
        value_clip: float = 0.4,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.clip_epsilon = clip_epsilon
        self.entropy_beta = entropy_beta
        self.value_clip = value_clip
        self.gamma = gamma
        self.lam = lam
        
        # 确保模型在同一设备上
        self.device = next(model.parameters()).device
        self.ref_model.to(self.device)

    def generate(self, prompt: str, max_length: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用模型生成回复并获取logits和values"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            logits = outputs.logits
            values = outputs.hidden_states[-1].mean(dim=1)  # 简化的价值估计
            
        # 采样生成回复
        generated_ids = torch.softmax(logits[:, -1, :], dim=-1).multinomial(1)
        generated_text = self.tokenizer.decode(generated_ids[0])
        
        return inputs.input_ids, generated_ids, logits, values

    def compute_rewards(
        self, 
        rewards: List[float], 
        values: torch.Tensor, 
        action_logprobs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算优势和回报"""
        rewards = torch.tensor(rewards, device=self.device)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_advantage = 0
        last_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * last_value - values[t]
            advantages[t] = delta + self.gamma * self.lam * last_advantage
            last_advantage = advantages[t]
            last_value = values[t]
            
        returns = advantages + values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns

    def train_step(
        self, 
        queries: List[torch.Tensor], 
        responses: List[torch.Tensor], 
        rewards: List[float]
    ) -> Dict[str, float]:
        """执行一个训练步骤"""
        # 获取参考模型的logits
        ref_logprobs = []
        for query, response in zip(queries, responses):
            input_ids = torch.cat([query, response], dim=1)
            with torch.no_grad():
                ref_outputs = self.ref_model(input_ids=input_ids)
                ref_logits = ref_outputs.logits
                ref_logprobs.append(self._get_logprobs(ref_logits, response))
        
        # 获取当前模型的logits和values
        logprobs = []
        values = []
        entropies = []
        
        for query, response in zip(queries, responses):
            input_ids = torch.cat([query, response], dim=1)
            outputs = self.model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            logits = outputs.logits
            hidden_states = outputs.hidden_states
            
            # 提取logprobs和entropy
            batch_logprobs = self._get_logprobs(logits, response)
            entropy = -torch.sum(torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1), dim=-1)
            
            # 简化的价值估计
            value = hidden_states[-1].mean(dim=1)
            
            logprobs.append(batch_logprobs)
            values.append(value)
            entropies.append(entropy.mean())
        
        # 计算优势和回报
        advantages = []
        returns = []
        
        for value, logprob, ref_logprob, reward in zip(values, logprobs, ref_logprobs, rewards):
            adv, ret = self.compute_rewards([reward], value, logprob - ref_logprob)
            advantages.append(adv)
            returns.append(ret)
        
        # 计算损失
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for logprob, ref_logprob, advantage, return_, value, entropy in zip(
            logprobs, ref_logprobs, advantages, returns, values, entropies
        ):
            # 计算策略概率比
            ratio = torch.exp(logprob - ref_logprob)
            
            # 策略损失
            pg_loss1 = -advantage * ratio
            pg_loss2 = -advantage * torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            policy_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # 值损失
            value_clipped = value + torch.clamp(return_ - value, -self.value_clip, self.value_clip)
            v_loss1 = (return_ - value) ** 2
            v_loss2 = (return_ - value_clipped) ** 2
            value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
            
            # 熵损失
            entropy_loss = -entropy * self.entropy_beta
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropy_losses.append(entropy_loss)
        
        # 总损失
        total_loss = (
            torch.stack(policy_losses).mean() + 
            torch.stack(value_losses).mean() + 
            torch.stack(entropy_losses).mean()
        )
        
        # 优化步骤
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 返回训练统计信息
        return {
            "policy_loss": policy_losses[-1].item(),
            "value_loss": value_losses[-1].item(),
            "entropy_loss": entropy_losses[-1].item(),
            "total_loss": total_loss.item(),
            "reward": rewards[-1],
            "kl_divergence": (logprobs[-1] - ref_logprobs[-1]).mean().item()
        }
    
    def _get_logprobs(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算给定标签的log概率"""
        logprobs = torch.log_softmax(logits, dim=-1)
        logprobs_labels = torch.gather(logprobs, 2, labels.unsqueeze(2)).squeeze(-1)
        return logprobs_labels

# 示例使用
def main():
    # 加载模型和分词器
    model_name = "gpt2"  # 示例模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # 初始化PPO训练器
    ppo_trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer
    )
    
    # 示例训练循环
    prompts = [
        "请介绍一下人工智能",
        "解释一下量子计算的基本概念",
        "什么是自然语言处理?"
    ]
    
    for epoch in range(3):
        for prompt in prompts:
            # 生成回复
            query, response, _, _ = ppo_trainer.generate(prompt)
            
            # 这里需要一个奖励函数，实际应用中可以是人类反馈或自动评估
            # 简化示例，使用随机奖励
            reward = np.random.uniform(1, 5)
            
            # 训练一步
            stats = ppo_trainer.train_step([query], [response], [reward])
            
            # 打印训练统计
            print(f"Epoch {epoch+1} | Reward: {stats['reward']:.2f} | Loss: {stats['total_loss']:.4f}")
    
    # 保存微调后的模型
    model.save_pretrained("ppo_finetuned_model")
    tokenizer.save_pretrained("ppo_finetuned_model")

if __name__ == "__main__":
    main()    