import time, re, json
from typing import List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_error_json():
    with open("evaluation_samples_false.json", "r") as f:
        error_json = json.load(f)
    return error_json

def get_true_json():
    with open("evaluation_samples_true.json", "r") as f:
        true_json = json.load(f)
    return true_json    

def extract_keywords(json_str: str) -> List[str]:
    
    pattern = r'\b\w+\b'
    keywords = re.findall(pattern, json_str.lower())
    return list(set([w for w in keywords if len(w) > 1]))  # Return unique keywords


def evaluate(samples: List[dict], references_true: bool) -> dict:
    
    y_true, y_pred = [], []
    context_hits = []
    total_time = 0
    for sample in samples:
        start_time = time.time()
        ref = extract_keywords(sample["reference_answer"])
        pred = extract_keywords(sample["generated_answer"])

        matched = set(ref) & set(pred)
        hit_rate = len(matched) / max(len(ref), 1)  # Avoid division by zero

        if references_true:
            y_true.append(1)
        else:            
            y_true.append(0)

        y_pred.append(1 if hit_rate > 0.7 else 0)  # Predict True if hit_rate > 0.7, else False
        
        ctx_tokens = set()
        for ctx in sample["context"]:
            ctx_tokens |= set(extract_keywords(ctx))

        coverage = len(ctx_tokens & set(ref)) / max(len(ref), 1)  # Coverage of reference keywords in context
        context_hits.append(round(coverage, 2))
        total_time += time.time() - start_time  
        

    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    avg_coverage = sum(context_hits) / len(context_hits) if context_hits else 0
    avg_latency = round(total_time / len(samples), 3) if samples else 0


    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_coverage": avg_coverage,
        "avg_latency": avg_latency
    }

# Example usage
if __name__ == "__main__":
    
    samples_true = get_true_json()
    samples_false = get_error_json()

    evaluation_true = evaluate(samples_true, True)
    evaluation_false = evaluate(samples_false, False)

    print("Evaluation Results (True Samples):")
    for metric, value in evaluation_true.items():
        print(f"{metric.capitalize()}: {value:.2f}")

    print("\nEvaluation Results (False Samples):")
    for metric, value in evaluation_false.items():
        print(f"{metric.capitalize()}: {value:.2f}")


