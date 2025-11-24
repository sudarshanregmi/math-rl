# file name: train_math_grpo.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Import our custom verifiers
from math_rewards import correctness_reward_func, format_reward_func, xml_count_reward_func

# --- Config ---
# Using Qwen 2.5 Math as base allows faster convergence, but you can use Llama/Mistral
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 
OUTPUT_DIR = "output/Qwen-Reasoning-GRPO"

# GRPO specific hyperparameters
NUM_GENERATIONS = 8     # G: Number of rollouts per prompt (The "Group" in GRPO)
MAX_COMPLETION_LEN = 768 # Reasoning traces can be long

def get_gsm8k_dataset():
    """Loads and formats the GSM8K dataset for reasoning."""
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    
    # We add a system prompt to force the model into "Reasoning Mode"
    system_prompt = """You are a mathematical reasoning bot. 
    You must wrap your thinking process between <think> and </think> tags.
    You must put your final answer inside \\boxed{}."""
    
    def format_data(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["question"]}
            ],
            "solution": example["answer"]
        }
    
    return dataset.map(format_data, remove_columns=dataset.column_names)

def main():
    # 1. Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (using bfloat16 for Ampere GPUs, use float16 otherwise)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    # 2. LoRA Config
    # GRPO is memory heavy because it generates G outputs per step. 
    # PEFT/LoRA is standard to fit this on consumer GPUs.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
    )

    # 3. Training Arguments
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,           # GRPO usually prefers lower LR than SFT
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1, # Keep small, num_generations increases actual memory usage
        gradient_accumulation_steps=4,
        num_generations=NUM_GENERATIONS, # The core GRPO param.
        max_prompt_length=256,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=1,
        save_steps=100,
        report_to="none", # Set to 'wandb' for tracking
        use_vllm=False,   # Enable if vLLM is installed for 3x faster generation
    )

    # 4. Dataset
    dataset = get_gsm8k_dataset()

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            xml_count_reward_func,  # Reward structure (<think> tags)
            format_reward_func,     # Reward format (think + boxed)
            correctness_reward_func # Reward Accuracy (The Verifier)
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train
    print("Starting GRPO on Math Dataset...")
    trainer.train()
    
    # 7. Save
    trainer.save_model(OUTPUT_DIR)
    print("Training complete.")

if __name__ == "__main__":
    main()
