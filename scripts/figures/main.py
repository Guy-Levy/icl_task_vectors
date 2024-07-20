import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from core.config import FIGURES_DIR

from scripts.figures.helpers import (
    MODEL_DISPLAY_NAME_MAPPING,
    load_main_results,
    extract_accuracies,
    create_accuracies_df,
    create_grouped_accuracies_df,
)

from scripts.figures.tv_injection import plot_tv_injection, plot_complex_tasks

def filter_tasks_with_low_icl_accuracy(grouped_accuracies_df, regular_accuracy_threshold=0.10):
    mask = (grouped_accuracies_df["Regular"] - grouped_accuracies_df["Baseline"]) >= regular_accuracy_threshold
    filtered_task_accuracies_df = grouped_accuracies_df[mask].copy()

    # print the excluded model,task pairs, Hypothesis by commas
    if not mask.all():
        print(
            "Excluded:",
            grouped_accuracies_df[~mask][["model", "task_name", "Regular"]].apply(
                lambda x: f"({x['model']}, {x['task_name']}): {x['Regular']:.2f}", axis=1
            ),
        )
    print("Num excluded / total:", (~mask).sum(), "/", len(grouped_accuracies_df))

    return filtered_task_accuracies_df


def plot_avg_accuracies_per_model(grouped_accuracies_df, new_format=False):
    filtered_task_accuracies_df = filter_tasks_with_low_icl_accuracy(grouped_accuracies_df)

    columns_to_plot = ["Baseline", "Hypothesis", "Regular"]

    # Calculate average accuracy and std deviation for each model
    df_agg = filtered_task_accuracies_df.groupby("model")[columns_to_plot].agg("mean")

    # Plotting

    # Sort the model names, firsts by the base name, then by the size (e.g. "Pythia 6.9B" < "Pythia 13B", "LLaMA 7B" < "LLaMA 13B")
    model_names = sorted(df_agg.index.unique(), key=lambda x: (x.split(" ")[0], float(x.split(" ")[1][:-1])))
    num_models = len(model_names)

    plt.rcParams.update({"font.size": 14})

    fig, ax = plt.subplots(figsize=(7, 8))

    bar_width = 0.3
    hatches = ["/", "\\", "|"]
    for j, column in enumerate(columns_to_plot):
        means = df_agg[column]
        y_positions = np.arange(len(means)) + (j - 1) * bar_width
        # make sure to show the model names from the index as the y ticks
        ax.barh(
            y_positions,
            means,
            height=bar_width,
            capsize=2,
            color=["grey", "blue", "green"][j],
            edgecolor="white",
            hatch=hatches[j] * 2,
        )

    # set the y ticks to be the model names, not numbers
    ax.set_yticks(np.arange(num_models))
    ax.set_yticklabels([model_name for model_name in model_names])
    ax.set_yticklabels(ax.get_yticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Accuracy")
    ax.set_xlim([0.0, 1.0])

    # show legend below the plot
    legend_elements = [
        Patch(facecolor="grey", edgecolor="white", hatch=hatches[0] * 2, label="Baseline"),
        Patch(facecolor="green", edgecolor="white", hatch=hatches[2] * 2, label="In-Context Learning"),
        Patch(facecolor="blue", edgecolor="white", hatch=hatches[1] * 2, label="Task Vector"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    # plt.tight_layout()

    # save the figure
    save_path = os.path.join(FIGURES_DIR, "main_experiment_results_per_model.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")


def plot_accuracy_by_layer(results, model_names, normalize_x_axis=False, filename_suffix="", new_format=False):
    plt.figure(figsize=(7, 5))

    plt.rc("font", size=16)

    plt.title(f"Average Accuracy by Layer")

    regular_accuracy_threshold = 0.75

    num_layers = {
        "llama_7B": 32,
        "llama_13B": 40,
        "llama_30B": 60,
        "gpt-j_6B": 28,
        "pythia_2.8B": 32,
        "pythia_6.9B": 32,
        "pythia_12B": 36,
    }

    # Define different markers for each model
    markers = ["o", "^", "s", "P", "X", "D", "v"]

    for idx, model_name in enumerate(model_names):
        all_tv_dev_accruacy_by_layer = []
        for task_name in results[model_name]:
            if new_format:
                for num_examples in results[model_name][task_name]["tv_dev_accruacy_by_layer"]:
                    all_tv_dev_accruacy_by_layer.append(
                        list(results[model_name][task_name]["tv_dev_accruacy_by_layer"][num_examples].values())
                    )
            else:
                all_tv_dev_accruacy_by_layer.append(
                    list(results[model_name][task_name]["tv_dev_accruacy_by_layer"].values())
                )

        all_tv_dev_accruacy_by_layer = np.array(all_tv_dev_accruacy_by_layer)
        all_tv_dev_accruacy_by_layer = all_tv_dev_accruacy_by_layer[
            all_tv_dev_accruacy_by_layer.max(axis=-1) > regular_accuracy_threshold
        ]

        mean_tv_dev_accruacy_by_layer = np.mean(all_tv_dev_accruacy_by_layer, axis=0)
        std_tv_dev_accruacy_by_layer = np.std(all_tv_dev_accruacy_by_layer, axis=0)

        if new_format:
            layers = np.array(list(results[model_name][list(results[model_name].keys())[0]]["tv_dev_accruacy_by_layer"][list(results[model_name][list(results[model_name].keys())[0]]["tv_dev_accruacy_by_layer"].keys())[0]].keys()))
        else:
            layers = np.array(list(results[model_name][list(results[model_name].keys())[0]]["tv_dev_accruacy_by_layer"].keys()))

        x_values = layers
        if normalize_x_axis:
            x_values = x_values / num_layers[model_name]

        # Use different marker for each model and increase the marker size
        plt.plot(
            x_values,
            mean_tv_dev_accruacy_by_layer,
            marker=markers[idx],
            markersize=10,
            label=MODEL_DISPLAY_NAME_MAPPING[model_name],
            alpha=0.8,
        )
        plt.fill_between(
            x_values,
            mean_tv_dev_accruacy_by_layer - std_tv_dev_accruacy_by_layer,
            mean_tv_dev_accruacy_by_layer + std_tv_dev_accruacy_by_layer,
            alpha=0.1,
        )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")

    plt.ylim(0.0, 1.0)

    # place the legend on the top right corner
    plt.legend(loc="upper right")

    # save the figure
    save_path = os.path.join(FIGURES_DIR, f"accuracy_per_layer{filename_suffix}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

def create_all_figures(experiment_id: str, new_format=False):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_main_results(experiment_id)
    accuracies_df = create_accuracies_df(results, new_format=new_format)
    grouped_accuracies_df = create_grouped_accuracies_df(accuracies_df)

    plot_avg_accuracies_per_model(grouped_accuracies_df, new_format=new_format)
    # plot_accuracy_by_layer(results, model_names=["llama_7B", "llama_13B", "llama_30B"], new_format=new_format)
    plot_accuracy_by_layer(
        results, model_names=["pythia_6.9B", "pythia_12B", "gpt-j_6B"], filename_suffix="_appendix", new_format=new_format
    )

    plot_tv_injection(experiment_id=experiment_id)
    plot_complex_tasks(experiment_id=experiment_id)

if __name__ == "__main__":
    experiment_id = "21"
    new_format = True  # Set this to True for the new dictionary structure
    create_all_figures(experiment_id, new_format=new_format)
