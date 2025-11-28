import pandas as pd
import numpy as np

df = pd.read_csv('C:\\College Study\\Semester 5\\IE643\\Project\\results\\gpu_training\\model_validation_streamed_gsm8k_validation.csv')

def compute_gsm8k_score(df):
    # Extract relevant columns
    predictions = df['true_answer'].apply(lambda x: x.split('####')[-1].strip()).astype(str)
    ground_truths = df['model_answer'].apply(lambda x: x.split('####')[-1].strip()).astype(str)

    correct_count = 0
    total_count = len(ground_truths)

    for pred, gt in zip(predictions, ground_truths):
        try:
            # Evaluate the predicted answer
            pred_value = eval(pred)
            gt_value = eval(gt)

            # Compare the evaluated values
            if pred_value == gt_value:
                correct_count += 1
        except:
            # If evaluation fails, consider it incorrect
            continue

    score = correct_count / total_count if total_count > 0 else 0.0

    df["exact_true_answer"] = predictions
    df["exact_model_answer"] = ground_truths

    return score, df

score, df = compute_gsm8k_score(df)

df.to_csv('gsm8k_evaluation_results.csv', index=False)

print(f'GSM8K Evaluation Score: {score:.4f} * 100 = {score * 100:.2f}%')