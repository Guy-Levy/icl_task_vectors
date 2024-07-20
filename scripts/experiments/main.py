# This must be first
from dotenv import load_dotenv
load_dotenv(".env")

import sys
import os
import pickle
import time
from typing import Optional, Dict, Tuple, List
import torch
from scipy.stats import pearsonr
import gc
import shutil
import itertools

from transformers import PreTrainedModel, PreTrainedTokenizer

from scripts.utils import MAIN_RESULTS_DIR, main_experiment_results_dir

from core.data.task_helpers import get_all_tasks, get_task_by_name, COMPLEX_TASKS
from core.models.llm_loading import load_model_and_tokenizer
from core.models.utils.inference import hidden_to_logits
from core.analysis.utils import logits_top_tokens
from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.task_vectors import run_icl, run_task_vector, analyze_tv_distance_accuracy_correlation, modulated_generate
from core.utils.misc import limit_gpus, seed_everything
from core.experiments_config import MODELS_TO_EVALUATE, TASKS_TO_EVALUATE, COMPLEX_TASKS_TO_EVALUATE

# plotting & stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.optimize import minimize

# Helper functions

def get_results_file_path(model_type: str, model_variant: str, experiment_id: str = "") -> str:
    return os.path.join(main_experiment_results_dir(experiment_id), f"{model_type}_{model_variant}.pkl")

def get_new_experiment_id() -> str:
    return str(max([int(results_dir) for results_dir in os.listdir(MAIN_RESULTS_DIR) if results_dir.isdigit()] + [0]) + 1)

def load_or_create_results(results_file: str) -> Dict:
    if os.path.exists(results_file):
        with open(results_file, "rb") as f:
            return pickle.load(f)
    return {}

def save_results(results: Dict, results_file: str):
    with open(results_file, "wb") as f:
        pickle.dump(results, f)

def calculate_correlation(x: List[int], y: List[float]) -> Tuple[float, float]:
    correlation, p_value = pearsonr(x, y)
    return correlation, p_value

# Main experiment functions

def evaluate_task(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task_name: str,
    num_examples_range: range,
    print_icl_samples: bool = False
) -> Tuple[Dict, Dict, Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    seed_everything(41)
    accuracies = {}
    task_hiddens = {}
    correctness_mask = {}

    task = get_task_by_name(tokenizer=tokenizer, task_name=task_name)

    # Evaluate baseline
    baseline_datasets = task.create_datasets(num_datasets=100, num_examples=0)
    predictions = run_icl(model, tokenizer, task, baseline_datasets, include_train=False)
    accuracies["baseline"] = calculate_accuracy_on_datasets(task, predictions, baseline_datasets)

    # Evaluate ICL and Task Vector
    # TODO: Change back to 400, 100
    # num_test_datasets, num_dev_datasets = 400, 100
    num_test_datasets, num_dev_datasets = 100, 50 # best for now.

    accuracies["tv_dev_by_layer"] = {}
    accuracies["icl"] = {}
    accuracies["tv"] = {}
    tv_ordered_tokens_by_layer = {}

    for num_examples in num_examples_range:
        torch.cuda.empty_cache()
        try:
            test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=num_examples)
            dev_datasets = task.create_datasets(num_datasets=num_dev_datasets, num_examples=num_examples)
            icl_predictions = run_icl(model, tokenizer, task, test_datasets, print_samples=print_icl_samples)
            tv_predictions, tv_dev_accuracy_by_layer, current_task_hiddens = run_task_vector(
                model,
                tokenizer,
                task,
                test_datasets,
                dev_datasets,
                layers_to_test=range(10,20)
            )
            accuracies["tv_dev_by_layer"][num_examples] = tv_dev_accuracy_by_layer
            accuracies["icl"][num_examples] = calculate_accuracy_on_datasets(task, icl_predictions, test_datasets)
            accuracies["tv"][num_examples], tv_correctness_mask = calculate_accuracy_on_datasets(task, tv_predictions, test_datasets, True)

            task_hiddens[num_examples] = current_task_hiddens
            correctness_mask[num_examples] = torch.from_numpy(tv_correctness_mask)

            tv_ordered_tokens_by_layer[num_examples] = {}
            try:
                for layer_num in tv_dev_accuracy_by_layer.keys():
                    task_hidden = current_task_hiddens.mean(axis=0)[layer_num]
                    logits = hidden_to_logits(model, task_hidden)
                    tv_ordered_tokens_by_layer[num_examples][layer_num] = logits_top_tokens(logits, tokenizer, k=100)
            except Exception as e:
                print("Error:", e)

        except torch.cuda.OutOfMemoryError:
            print(f"During num_examples: {num_examples} - CUDA out of memory error occurred")

    return accuracies, tv_ordered_tokens_by_layer, task_hiddens, correctness_mask


def run_main_experiment(
    model_type: str,
    model_variant: str,
    num_examples_range: range,
    experiment_id: str = "",
    model: Optional[PreTrainedModel] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    print_icl_samples: bool = False,
) -> Tuple[Dict[str, Dict[int, int]], Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, torch.Tensor]]]:
    print(f"Evaluating model: {model_type} {model_variant}")

    results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results = load_or_create_results(results_file)

    limit_gpus(range(0, 8))

    if model is None or tokenizer is None:
        print("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
        print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)
    print(f"The range of examples to be tested: {num_examples_range}")
    best_layers, task_hiddens_dict, correctness_mask_dict = {}, {}, {}

    for i, task_name in enumerate(TASKS_TO_EVALUATE):
        if task_name in results:
            print(f"Skipping task {i+1}/{len(TASKS_TO_EVALUATE)}: {task_name}")
            continue

        print(f"\n{'='*50}\nRunning task {i+1}/{len(TASKS_TO_EVALUATE)}: {task_name}")
        task = tasks[task_name]
        
        tic = time.time()
        accuracies, tv_ordered_tokens_by_layer, task_hiddens, correctness_mask = evaluate_task(
            model, tokenizer, task_name, num_examples_range, print_icl_samples=print_icl_samples
        )
        actual_num_examples_run = list(accuracies['icl'].keys())

        print(f"Baseline Accuracy: {accuracies['baseline']:.2f}")
        print(f"ICL Accuracy: {', '.join(f'{num:.2f}' for num in accuracies['icl'].values())}")
        print(f"Task Vector Accuracy: {', '.join(f'{num:.2f}' for num in accuracies['tv'].values())}")
        print(f"Time: {time.time() - tic:.2f} seconds")

        best_layer = {i: max(accuracies["tv_dev_by_layer"][i], key=accuracies["tv_dev_by_layer"][i].get) for i in actual_num_examples_run}
        best_layers[task_name] = best_layer

        results[task_name] = {
            "baseline_accuracy": accuracies["baseline"],
            "num_examples": list(num_examples_range),
            "actual_examples_run": actual_num_examples_run,
            "icl_accuracy": accuracies["icl"],
            "tv_accuracy": accuracies["tv"],
            "tv_dev_accruacy_by_layer": accuracies["tv_dev_by_layer"],
            "tv_ordered_tokens_by_layer": tv_ordered_tokens_by_layer,
            "best_layer": best_layer,
        }

        task_hiddens_dict[task_name] = task_hiddens
        correctness_mask_dict[task_name] = correctness_mask

        if len(actual_num_examples_run) > 1:
            for acc_type in ['icl', 'tv']:
                correlation, p_value = calculate_correlation(actual_num_examples_run, list(accuracies[f'{acc_type}'].values()))
                results[task_name][f"num_examples_{acc_type}_accuracy_correlation"] = {
                    "correlation": correlation,
                    "p_value": p_value
                }
                print(f"Correlation between number of examples and {acc_type.upper()} accuracy: {correlation:.4f} (p-value: {p_value:.4f})")

        save_results(results, results_file)

    return best_layers, task_hiddens_dict, correctness_mask_dict

def run_tv_distance_accuracy_correlation(
    model_type: str,
    model_variant: str,
    task_hiddens_dict: Dict[str, Dict[int, torch.Tensor]],
    correctness_mask_dict: Dict[str, Dict[int, torch.Tensor]],
    experiment_id: str = ""
) -> None:
    print(f"Running TV distance-accuracy correlation analysis for: {model_type} {model_variant}")

    default_results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    with open(default_results_file, "rb") as f:
        main_results = pickle.load(f)

    results_file = default_results_file.replace('.pkl', '_tv_correlation.pkl')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results = load_or_create_results(results_file)

    for task_name in TASKS_TO_EVALUATE:
        if task_name in results:
            print(f"Skipping task: {task_name} (already processed)")
            continue

        print(f"\nAnalyzing task: {task_name}")

        tv_acc = main_results[task_name]["tv_accuracy"]
        if not tv_acc:
            continue
        best_num_examples = max(tv_acc, key=tv_acc.get)
        best_layer = main_results[task_name]["best_layer"][best_num_examples]
        print(f"Best number of examples: {best_num_examples}, Best layer: {best_layer}")

        task_vectors = task_hiddens_dict[task_name][best_num_examples][:, best_layer]
        correctness_mask = correctness_mask_dict[task_name][best_num_examples]

        mean_tv = task_vectors.mean(dim=0)
        distances = torch.norm(task_vectors - mean_tv.unsqueeze(0), dim=1)
        correlation, p_value = pearsonr(distances.numpy(), correctness_mask.numpy())
        print(f"Correlation between accuracy and distance from mean TV: {correlation:.4f}, p-value: {p_value:.4f}")

        mean_tv_pos = task_vectors[correctness_mask].mean(dim=0)
        distances_pos = torch.norm(task_vectors - mean_tv_pos.unsqueeze(0), dim=1)
        correlation_pos, p_value_pos = pearsonr(distances_pos.numpy(), correctness_mask.numpy())
        print(f"Correlation between accuracy and distance from mean TV of successful examples: {correlation_pos:.4f}, p-value: {p_value_pos:.4f}")

        results[task_name] = {
            "best_num_examples": best_num_examples,
            "best_layer": best_layer,
            "correlation": correlation,
            "p_value": p_value,
            "correlation_pos": correlation_pos,
            "p_value_pos": p_value_pos,
        }

        save_results(results, results_file)

    print("TV distance-accuracy correlation analysis completed.")

def test_mean_tv_injection(
    model_type: str,
    model_variant: str,
    task_hiddens_dict: Dict[str, Dict[int, torch.Tensor]],
    correctness_mask_dict: Dict[str, Dict[int, torch.Tensor]],
    experiment_id: str = "",
    print_debug: bool = False
) -> None:
    print(f"Testing mean TV injection for: {model_type} {model_variant}")

    default_results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    with open(default_results_file, "rb") as f:
        main_results = pickle.load(f)

    results_file = default_results_file.replace('.pkl', '_mean_tv_injection.pkl')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results = load_or_create_results(results_file)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    for task_name in TASKS_TO_EVALUATE:
        if task_name in results:
            print(f"Skipping task: {task_name} (already processed)")
            continue

        print(f"\nAnalyzing task: {task_name}")
        torch.cuda.empty_cache()
        try:
            task = tasks[task_name]
            tic = time.time()

            tv_acc = main_results[task_name]["tv_accuracy"]
            if not tv_acc:
                continue
            best_num_examples = max(tv_acc, key=tv_acc.get)
            best_layer = main_results[task_name]["best_layer"][best_num_examples]
            print(f"Best number of examples: {best_num_examples}, Best layer: {best_layer}")

            task_vectors = task_hiddens_dict[task_name][best_num_examples]
            correctness_mask = correctness_mask_dict[task_name][best_num_examples]

            if print_debug:
                print_debug_info(task_vectors, correctness_mask)

            mean_tv = task_vectors[:, best_layer].mean(dim=0)
            mean_tv_pos = task_vectors[:, best_layer][correctness_mask].mean(dim=0)

            if print_debug:
                print_more_debug_info(mean_tv, mean_tv_pos)

            num_test_datasets = 100
            test_datasets = task.create_datasets(num_datasets=num_test_datasets, num_examples=best_num_examples)

            original_accuracy = test_accuracy(model, tokenizer, task, test_datasets, task_vectors, best_layer)
            mean_tv_accuracy = test_accuracy(model, tokenizer, task, test_datasets, mean_tv, best_layer, num_test_datasets)
            mean_tv_pos_accuracy = test_accuracy(model, tokenizer, task, test_datasets, mean_tv_pos, best_layer, num_test_datasets)

            print(f"Original TV accuracy: {original_accuracy:.4f}")
            print(f"Mean TV accuracy: {mean_tv_accuracy:.4f}")
            print(f"Mean TV (positive) accuracy: {mean_tv_pos_accuracy:.4f}")

            results[task_name] = {
                "best_num_examples": best_num_examples,
                "best_layer": best_layer,
                "original_accuracy": original_accuracy,
                "mean_tv_accuracy": mean_tv_accuracy,
                "mean_tv_pos_accuracy": mean_tv_pos_accuracy,
            }

            print(f"Time taken for task: {time.time() - tic:.2f} seconds")

            save_results(results, results_file)

        except torch.cuda.OutOfMemoryError:
            print("CUDA out of memory error occurred")

    print("Mean TV injection testing completed.")

# Helper functions for test_mean_tv_injection

def print_debug_info(task_vectors, correctness_mask):
    print(f"Shape of task_vectors: {task_vectors.shape}")
    print(f"Shape of correctness_mask: {correctness_mask.shape}")
    print(f"Number of correct examples: {correctness_mask.sum()}")
    print(f"Total number of examples: {len(correctness_mask)}")

def print_more_debug_info(mean_tv, mean_tv_pos):
    print(f"Mean TV norm: {torch.norm(mean_tv)}")
    print(f"Mean TV (positive) norm: {torch.norm(mean_tv_pos)}")
    print(f"Difference between mean TVs: {torch.norm(mean_tv - mean_tv_pos)}")

def test_accuracy(model, tokenizer, task, test_datasets, task_hiddens, best_layer, num_repeat=None):
    if num_repeat:
        task_hiddens = task_hiddens.unsqueeze(0).repeat(num_repeat, task_hiddens.shape[0], 1)
    predictions = modulated_generate(
        model, tokenizer, task, test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_layer
    )
    return calculate_accuracy_on_datasets(task, predictions, test_datasets)

def run_complex_task_experiment(
    model_type: str,
    model_variant: str,
    task_hiddens_dict: Dict[str, Dict[int, torch.Tensor]],
    correctness_mask_dict: Dict[str, Dict[int, torch.Tensor]],
    experiment_id: str = ""
):
    print(f"Running complex task experiment for: {model_type} {model_variant}")

    default_results_file = get_results_file_path(model_type, model_variant, experiment_id=experiment_id)
    with open(default_results_file, "rb") as f:
        main_results = pickle.load(f)

    results_file = default_results_file.replace('.pkl', '_complex_tasks.pkl')
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    results = load_or_create_results(results_file)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_type, model_variant)
    print("Loaded model and tokenizer.")

    tasks = get_all_tasks(tokenizer=tokenizer)

    for complex_task_name in COMPLEX_TASKS_TO_EVALUATE:
        if complex_task_name not in TASKS_TO_EVALUATE:
            continue

        if complex_task_name in results:
            print(f"Skipping task: {complex_task_name} (already processed)")
            continue

        print(f"\nAnalyzing complex task: {complex_task_name}")
        complex_task = tasks[complex_task_name]
        sub_tasks = COMPLEX_TASKS[complex_task_name]["sub_tasks"]

        for sub_task in sub_tasks:
            if sub_task not in TASKS_TO_EVALUATE:
                continue

        # Determine best_num_examples and best_layer
        tv_acc = main_results[complex_task_name]["tv_accuracy"]
        if not tv_acc:
            continue
        best_num_examples = max(tv_acc, key=tv_acc.get)
        best_layer = main_results[complex_task_name]["best_layer"][best_num_examples]
        print(f"Best number of examples: {best_num_examples}, Best layer: {best_layer}")

        torch.cuda.empty_cache()
        try:
            tic = time.time()

            # Get task vectors and correctness masks for sub-tasks
            sub_task_vectors = {}
            sub_task_correctness_masks = {}
            for sub_task_name in sub_tasks:
                sub_task_vectors[sub_task_name] = task_hiddens_dict[sub_task_name][best_num_examples][:, best_layer]
                sub_task_correctness_masks[sub_task_name] = correctness_mask_dict[sub_task_name][best_num_examples]

            # Calculate mean TVs and mean TVs of correct examples for sub-tasks
            mean_tvs = {name: vectors.mean(dim=0) for name, vectors in sub_task_vectors.items()}
            mean_tvs_correct = {name: vectors[mask].mean(dim=0) for (name, vectors), (_, mask) in zip(sub_task_vectors.items(), sub_task_correctness_masks.items())}

            # Define lambda values for linear combination
            lambda_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

            num_test_datasets = 100
            test_datasets = complex_task.create_datasets(num_datasets=num_test_datasets, num_examples=best_num_examples)

            tv_results = {}

            for tv_type in ['mean', 'mean_correct']:
                tvs = mean_tvs if tv_type == 'mean' else mean_tvs_correct

                def objective(lambdas):
                    # Ensure lambdas sum to 1
                    lambdas = lambdas / np.sum(lambdas)

                    combined_tv = sum(lambda_val * tvs[task_name] for lambda_val, task_name in zip(lambdas, sub_tasks))
                    accuracy = test_accuracy(model, tokenizer, complex_task, test_datasets, combined_tv, best_layer, num_test_datasets)

                    # We want to maximize accuracy, so return negative accuracy
                    return -accuracy

                # Initial guess: equal weights
                initial_lambdas = np.ones(len(sub_tasks)) / len(sub_tasks)

                result = minimize(objective, initial_lambdas, method='nelder-mead',
                                  options={'xatol': 1e-8, 'disp': True, 'maxiter': 15, 'maxfev': 50})

                best_lambdas = result.x / np.sum(result.x)  # Normalize to ensure sum is 1
                best_accuracy = -result.fun  # Remember we minimized negative accuracy

                tv_results[tv_type] = {
                    "accuracy": best_accuracy,
                    "lambdas": best_lambdas.tolist()  # Convert numpy array to list for JSON serialization
                }

            original_accuracy = main_results[complex_task_name]["tv_accuracy"][best_num_examples]

            results[complex_task_name] = {
                "best_num_examples": best_num_examples,
                "best_layer": best_layer,
                "original_accuracy": original_accuracy,
                "mean_tv_accuracy": tv_results['mean']['accuracy'],
                "mean_tv_lambdas": tv_results['mean']['lambdas'],
                "mean_tv_pos_accuracy": tv_results['mean_correct']['accuracy'],
                "mean_tv_pos_lambdas": tv_results['mean_correct']['lambdas'],
            }

            print(f"Original accuracy: {original_accuracy:.4f}")
            print(f"Mean TV accuracy: {tv_results['mean']['accuracy']:.4f}")
            print(f"Mean TV lambdas: {tv_results['mean']['lambdas']}")
            print(f"Mean TV (positive) accuracy: {tv_results['mean_correct']['accuracy']:.4f}")
            print(f"Mean TV (positive) lambdas: {tv_results['mean_correct']['lambdas']}")

            print(f"Time taken for task: {time.time() - tic:.2f} seconds")

            save_results(results, results_file)

        except torch.cuda.OutOfMemoryError:
            print(f"CUDA out of memory error occurred for task {complex_task_name}")

    print("Complex task experiment completed.")
    return results

# Main execution

def copy_log_to_experiment_dir(experiment_id: str):
    source_log = "logs/experiments_main.log"
    destination_dir = os.path.join(MAIN_RESULTS_DIR, experiment_id)
    destination_log = os.path.join(destination_dir, "experiments_main.log")
    
    if os.path.exists(source_log):
        shutil.copy2(source_log, destination_log)
        print(f"Log file copied to {destination_log}")
    else:
        print(f"Warning: Log file {source_log} not found")

def main():
    if len(sys.argv) == 1:
        experiment_id = get_new_experiment_id()
        print(f"Experiment ID: {experiment_id}")

        num_examples_range = range(2, 11, 3)
        for model_type, model_variant in MODELS_TO_EVALUATE:

            print("==================================================")
            print(f"{model_type} - {model_variant}")
            print("==================================================")

            print()
            print(f"@@@@@@@@@@@@@@@ MAIN EXPERIMENT @@@@@@@@@@@@@@@")
            print()
            torch.cuda.empty_cache()
            best_layers, task_hiddens_dict, correctness_mask_dict = run_main_experiment(
                model_type, model_variant, num_examples_range, experiment_id=experiment_id, print_icl_samples=False
            )

            print()
            print(f"@@@@@@@@@@@ MEAN TV DISTANCE ACC CORRELATION @@@@@@@@@@@")
            print()
            torch.cuda.empty_cache()
            run_tv_distance_accuracy_correlation(model_type, model_variant, task_hiddens_dict, correctness_mask_dict, experiment_id=experiment_id)

            print()
            print(f"@@@@@@@@@@@@@@@ MEAN TV INJECTION @@@@@@@@@@@@@@@")
            print()
            torch.cuda.empty_cache()
            test_mean_tv_injection(model_type, model_variant, task_hiddens_dict, correctness_mask_dict, experiment_id=experiment_id)

            # Run complex task experiment
            print()
            print(f"@@@@@@@@@@@@@@@ COMPLEX TASKS @@@@@@@@@@@@@@@")
            print()
            torch.cuda.empty_cache()
            run_complex_task_experiment(model_type, model_variant, task_hiddens_dict, correctness_mask_dict, experiment_id=experiment_id)

            del best_layers, task_hiddens_dict, correctness_mask_dict
            gc.collect()
        
        # Copy log file to experiment directory
        copy_log_to_experiment_dir(experiment_id)

    else:
        if len(sys.argv) == 2:
            model_num = int(sys.argv[1])
            model_type, model_variant = MODELS_TO_EVALUATE[model_num]
        elif len(sys.argv) == 3:
            model_type, model_variant = sys.argv[1:]

        best_layers = run_main_experiment(model_type, model_variant)
        run_tv_distance_accuracy_correlation(model_type, model_variant, best_layers)

if __name__ == "__main__":
    main()