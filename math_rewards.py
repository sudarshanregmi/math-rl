# file name: math_rewards.py
import re
import math

def correctness_reward_func(prompts, completions, solution, **kwargs):
    """
    The Automatic Verifier.
    Checks if the extracted answer from the model matches the ground truth solution.
    """
    rewards = []
    for content, sol in zip(completions, solution):
        # 1. Extract the model's answer. 
        # We expect the model to output \boxed{answer} or just the number at the end.
        # This regex looks for the last occurrence of \boxed{...}
        matches = re.findall(r"\\boxed\{(.*?)\}", content)
        if matches:
            pred = matches[-1].strip()
        else:
            # Fallback: try to find the last number in the text if no boxed format
            # (Strict reasoning models usually enforce boxed, but this is a safety net)
            numbers = re.findall(r"[-+]?\d*\.\d+|\d+", content)
            pred = numbers[-1] if numbers else ""

        # 2. Clean the solution (Ground Truth)
        # Datasets often have extra formatting in the solution column
        clean_sol = re.sub(r"<<.*?>>", "", sol) # Remove intermediate steps in some datasets
        clean_sol = clean_sol.split("####")[-1].strip() if "####" in clean_sol else sol.strip()
        
        # 3. Compare
        # For strict math, we usually verify exact string match or float equality
        try:
            # Float comparison for numbers
            if abs(float(pred) - float(clean_sol)) < 1e-5:
                rewards.append(1.0)
                continue
        except ValueError:
            pass
            
        # String comparison for non-numbers
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
    # DOTALL allows . to match newlines (multiline thinking)
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    return [1.0 if match else 0.0 for match in matches]

def xml_count_reward_func(completions, **kwargs):
    """
    Penalty if the model hallucinates multiple <think> tags or forgets them.
    """
    rewards = []
    for content in completions:
        count = content.count("<think>") + content.count("</think>")
        # We want exactly one opening and one closing tag
        if count == 2:
            rewards.append(0.5) # Small bonus for perfect XML structure
        else:
            rewards.append(0.0)
    return rewards
