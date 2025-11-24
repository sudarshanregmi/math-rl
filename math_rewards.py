# file name: math_rewards.py
import re

def get_content(completion):
    """Helper to extract text from TRL's conversational completion format."""
    # If it's a list (conversational format), grab the content of the first message
    if isinstance(completion, list):
        return completion[0]["content"]
    # If it's already a string (standard format), return it directly
    return completion

def correctness_reward_func(prompts, completions, solution, **kwargs):
    """
    The Automatic Verifier.
    Checks if the extracted answer from the model matches the ground truth solution.
    """
    rewards = []
    for completion, sol in zip(completions, solution):
        content = get_content(completion)
        
        # 1. Extract the model's answer. 
        matches = re.findall(r"\\boxed\{(.*?)\}", content)
        if matches:
            pred = matches[-1].strip()
        else:
            # Fallback: try to find the last number
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            pred = numbers[-1] if numbers else ""

        # 2. Clean the solution (Ground Truth)
        clean_sol = re.sub(r"<<.*?>>", "", sol) 
        clean_sol = clean_sol.split("####")[-1].strip() if "####" in clean_sol else sol.strip()
        
        # 3. Compare
        try:
            # Float comparison
            if abs(float(pred) - float(clean_sol)) < 1e-5:
                rewards.append(1.0)
                continue
        except ValueError:
            pass
            
        # String comparison
        if pred == clean_sol:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

def format_reward_func(completions, **kwargs):
    """
    Encourages the model to follow the Chain-of-Thought format:
    <think> ... </think> \boxed{answer}
    """
    pattern = r"^<think>.*?</think>\s*.*$"
    rewards = []
    for completion in completions:
        content = get_content(completion)
        match = re.match(pattern, content, re.DOTALL | re.MULTILINE)
        rewards.append(1.0 if match else 0.0)
    return rewards

def xml_count_reward_func(completions, **kwargs):
    """
    Penalty if the model hallucinates multiple <think> tags or forgets them.
    """
    rewards = []
    for completion in completions:
        content = get_content(completion)
        count = content.count("<think>") + content.count("</think>")
        if count == 2:
            rewards.append(0.5) 
        else:
            rewards.append(0.0)
    return rewards
