import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
import numpy as np

from scripts.utils import main_experiment_results_dir
from scripts.figures.helpers import MODEL_DISPLAY_NAME_MAPPING, TASKS_TO_PLOT
from core.config import FIGURES_DIR


def load_tv_injection_results(experiment_id: str = "camera_ready"):
    results = {}
    experiment_dir = main_experiment_results_dir(experiment_id)
    
    expected_models = [
        "llama_7B", "llama_13B", "llama_30B",
        "gpt-j_6B",
        "pythia_2.8B", "pythia_6.9B", "pythia_12B"
    ]

    for model_name in expected_models:
        file_name = f"{model_name}_mean_tv_injection.pkl"
        file_path = os.path.join(experiment_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    results[model_name] = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {file_name}. Error: {str(e)}")
        else:
            print(f"File not found: {file_name}")

    return results

def load_complex_tasks_results(experiment_id: str = "camera_ready"):
    results = {}
    experiment_dir = main_experiment_results_dir(experiment_id)
    
    expected_models = [
        "llama_7B", "llama_13B", "llama_30B",
        "gpt-j_6B",
        "pythia_2.8B", "pythia_6.9B", "pythia_12B"
    ]

    for model_name in expected_models:
        file_name = f"{model_name}_complex_tasks.pkl"
        file_path = os.path.join(experiment_dir, file_name)
        
        if os.path.exists(file_path):
            try:
                with open(file_path, "rb") as f:
                    results[model_name] = pickle.load(f)
            except Exception as e:
                print(f"Failed to load {file_name}. Error: {str(e)}")
        else:
            print(f"File not found: {file_name}")

    return results

def plot_tv_injection(experiment_id):
    results = load_tv_injection_results(experiment_id)
    accuracies = {}
    for model_name, model_results in results.items():
        accuracies[model_name] = {}
        for task_name, task_results in model_results.items():
            if task_name not in TASKS_TO_PLOT:
                continue
            accuracies[model_name][task_name] = {
                "original": task_results["original_accuracy"],
                "mean": task_results["mean_tv_accuracy"],
                "mean_pos": task_results["mean_tv_pos_accuracy"],
            }

    data = []
    for model_name, model_acc in accuracies.items():
        for task_full_name, task_acc in model_acc.items():
            task_type = task_full_name.split("_")[0]
            task_name = "_".join(task_full_name.split("_")[1:])
            data.append([model_name, task_type, task_name, "Original", task_acc["original"]])
            data.append([model_name, task_type, task_name, "Mean TV", task_acc["mean"]])
            data.append([model_name, task_type, task_name, "Mean TV of Positive", task_acc["mean_pos"]])

    df = pd.DataFrame(data, columns=["model", "task_type", "task_name", "method", "accuracy"])
    df["model"] = df["model"].map(MODEL_DISPLAY_NAME_MAPPING)

    task_order = sorted(zip(df["task_type"].unique(), df["task_name"].unique()), key=lambda x: x[0])
    task_order = [x[1] for x in task_order]

    grouped_accuracies_df = df.pivot_table(
        index=["model", "task_type", "task_name"], columns="method", values="accuracy", aggfunc="first"
    ).reset_index()
    
    # filtered_task_accuracies_df = filter_tasks_with_low_icl_accuracy(grouped_accuracies_df)
    filtered_task_accuracies_df = grouped_accuracies_df

    columns_to_plot = ["Original", "Mean TV", "Mean TV of Positive"]

    df_agg = filtered_task_accuracies_df.groupby("model")[columns_to_plot].agg("mean")

    model_names = sorted(df_agg.index.unique(), key=lambda x: (x.split(" ")[0], float(x.split(" ")[1][:-1])))
    num_models = len(model_names)

    plt.rcParams.update({"font.size": 14})

    fig, ax = plt.subplots(figsize=(7, 8))

    bar_width = 0.3
    hatches = ["/", "\\", "|"]
    for j, column in enumerate(columns_to_plot):
        means = df_agg[column]
        y_positions = np.arange(len(means)) + (j - 1) * bar_width
        ax.barh(
            y_positions,
            means,
            height=bar_width,
            capsize=2,
            color=["grey", "blue", "orange"][j],
            edgecolor="white",
            hatch=hatches[j] * 2,
        )

    ax.set_yticks(np.arange(num_models))
    ax.set_yticklabels([model_name for model_name in model_names])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Accuracy")
    ax.set_xlim([0.0, 1.0])

    legend_elements = [
        Patch(facecolor="grey", edgecolor="white", hatch=hatches[0] * 2, label="Original"),
        Patch(facecolor="orange", edgecolor="white", hatch=hatches[2] * 2, label="Mean TV"),
        Patch(facecolor="blue", edgecolor="white", hatch=hatches[1] * 2, label="Mean of Positive TV"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    save_path = os.path.join(FIGURES_DIR, "tv_injection_results_per_model.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def plot_complex_tasks(experiment_id):
    results = load_complex_tasks_results(experiment_id)
    accuracies = {}
    for model_name, model_results in results.items():
        accuracies[model_name] = {}
        for task_name, task_results in model_results.items():
            if task_name not in TASKS_TO_PLOT:
                continue
            accuracies[model_name][task_name] = {
                "original": task_results["original_accuracy"],
                "mean": task_results["mean_tv_accuracy"],
                # "mean_tv_lambdas": task_results["tv_results"],
                "mean_pos": task_results["mean_tv_pos_accuracy"],
                # "mean_tv_pos_lambdas": task_results["tv_results"],
            }

    data = []
    for model_name, model_acc in accuracies.items():
        for task_full_name, task_acc in model_acc.items():
            task_type = task_full_name.split("_")[0]
            task_name = "_".join(task_full_name.split("_")[1:])
            data.append([model_name, task_type, task_name, "Original", task_acc["original"]])
            data.append([model_name, task_type, task_name, "Mean TV", task_acc["mean"]])
            data.append([model_name, task_type, task_name, "Mean TV of Positive", task_acc["mean_pos"]])

    df = pd.DataFrame(data, columns=["model", "task_type", "task_name", "method", "accuracy"])
    df["model"] = df["model"].map(MODEL_DISPLAY_NAME_MAPPING)

    task_order = sorted(zip(df["task_type"].unique(), df["task_name"].unique()), key=lambda x: x[0])
    task_order = [x[1] for x in task_order]

    grouped_accuracies_df = df.pivot_table(
        index=["model", "task_type", "task_name"], columns="method", values="accuracy", aggfunc="first"
    ).reset_index()
    
    # filtered_task_accuracies_df = filter_tasks_with_low_icl_accuracy(grouped_accuracies_df)
    filtered_task_accuracies_df = grouped_accuracies_df

    columns_to_plot = ["Original", "Mean TV", "Mean TV of Positive"]

    df_agg = filtered_task_accuracies_df.groupby("model")[columns_to_plot].agg("mean")

    model_names = sorted(df_agg.index.unique(), key=lambda x: (x.split(" ")[0], float(x.split(" ")[1][:-1])))
    num_models = len(model_names)

    plt.rcParams.update({"font.size": 14})

    fig, ax = plt.subplots(figsize=(7, 8))

    bar_width = 0.3
    hatches = ["/", "\\", "|"]
    for j, column in enumerate(columns_to_plot):
        means = df_agg[column]
        y_positions = np.arange(len(means)) + (j - 1) * bar_width
        ax.barh(
            y_positions,
            means,
            height=bar_width,
            capsize=2,
            color=["grey", "blue", "orange"][j],
            edgecolor="white",
            hatch=hatches[j] * 2,
        )

    ax.set_yticks(np.arange(num_models))
    ax.set_yticklabels([model_name for model_name in model_names])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Accuracy")
    ax.set_xlim([0.0, 1.0])

    legend_elements = [
        Patch(facecolor="grey", edgecolor="white", hatch=hatches[0] * 2, label="Original"),
        Patch(facecolor="orange", edgecolor="white", hatch=hatches[2] * 2, label="Mean TV"),
        Patch(facecolor="blue", edgecolor="white", hatch=hatches[1] * 2, label="Mean of Positive TV"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    save_path = os.path.join(FIGURES_DIR, "complex_tasks_results_per_model.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")