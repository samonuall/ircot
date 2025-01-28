import datasets

def create_prompts_for_math(instances):
    pass

if __name__ == "__main__":
    # features: ['query', 'reasoning', 'id', 'excluded_ids', 'gold_ids_long', 'gold_ids', 'gold_answer']
    bright_questions = datasets.load_dataset("xlangai/BRIGHT", "examples", split="theoremqa_theorems")
    
    # features: ['id', 'content']
    bright_theorems = datasets.load_dataset("xlangai/BRIGHT", "documents", split="theoremqa_theorems")
    
    
    
    