import os
import json
from collections import Counter
from typing import List, Dict

from tqdm import tqdm
from datasets import load_dataset


def create_prompts_for_math(instances: List[Dict]):
    pass



def write_math_instances_to_filepath(instances: List[Dict], full_filepath: str):
    """
    Process and write MATH dataset instances to a JSONL file.
    The MATH dataset contains mathematical problems and their solutions.
    """
    print(f"Writing in: {full_filepath}")
    problem_types = Counter()
    
    with open(full_filepath, "w") as full_file:
        for raw_instance in tqdm(instances):
            # Generic Format similar to other datasets
            processed_instance = {}
            processed_instance["dataset"] = "math"
            processed_instance["question_id"] = raw_instance["unique_id"]
            processed_instance["question_text"] = raw_instance["problem"]
            processed_instance["level"] = raw_instance["level"]            
            # Track problem level distribution
            problem_types[raw_instance["level"]] += 1
            
            # Store the solution
            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],  # Store the final answer
            }
            processed_instance["answers_objects"] = [answers_object]
            
            # Use full solution as context
            # TODO: actually don't do this
            processed_instance["contexts"] = [
                {
                    "idx": 0,
                    "title": f"Problem {raw_instance['unique_id']}",
                    "paragraph_text": raw_instance["solution"].strip(),
                    "is_supporting": True,
                }
            ]
            
            # Write the processed instance
            full_file.write(json.dumps(processed_instance) + "\n")
    
    print(f"Problem types distribution: {str(problem_types)}")


if __name__ == "__main__":
    # Load the MATH dataset
    dataset = load_dataset("nlile/hendrycks-MATH-benchmark")
    dev_dataset = dataset["train"].select(range(3000))
    train_dataset = dataset["train"].select(range(3000, len(dataset["train"])))
    test_dataset = dataset["test"]

    print(dataset["test"])
    print(dev_dataset)
    
    # Create the output directory
    directory = os.path.join("processed_data", "math")
    os.makedirs(directory, exist_ok=True)
    
    # Process and write train split
    processed_full_filepath = os.path.join(directory, "train.jsonl")
    write_math_instances_to_filepath(train_dataset, processed_full_filepath)
    
    # Process and write validation split
    processed_full_filepath = os.path.join(directory, "dev.jsonl")
    write_math_instances_to_filepath(dev_dataset, processed_full_filepath)
    
    # Process and write test split
    processed_full_filepath = os.path.join(directory, "test.jsonl")
    write_math_instances_to_filepath(test_dataset, processed_full_filepath) 