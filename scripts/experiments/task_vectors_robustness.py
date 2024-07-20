from dotenv import load_dotenv

load_dotenv(".env")

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from core.config import FIGURES_DIR
from core.data.task_helpers import get_task_by_name
from core.models.llm_loading import load_model_and_tokenizer
from core.task_vectors import get_task_hiddens, task_vector_accuracy_by_layer
from core.utils.misc import limit_gpus, seed_everything
from scripts.figures.helpers import TASKS_TO_PLOT


def create_task_vectors(model, tokenizer):
    task_vectors = {}

    for task_name in tqdm(TASKS_TO_PLOT):
        num_examples = 4

        task = get_task_by_name(tokenizer, task_name)

        test_datasets = task.create_datasets(num_datasets=50, num_examples=num_examples)
        dev_datasets = task.create_datasets(num_datasets=50, num_examples=num_examples)

        dev_accuracy_by_layer = task_vector_accuracy_by_layer(
            model, tokenizer, task, dev_datasets, layers_to_test=range(10, 20)
        )
        best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

        task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets)

        task_vectors[task_name] = task_hiddens[:, best_intermediate_layer]

    return task_vectors


def create_tsne_plot(task_vectors):
    all_task_vectors = torch.cat(list(task_vectors.values()), dim=0)
    task_vectors_labels = torch.cat([torch.full_like(v[:, 0], i) for i, v in enumerate(task_vectors.values())], dim=0)

    dim_reduction = TSNE(n_components=2, random_state=41)

    task_vectors_2d = dim_reduction.fit_transform(all_task_vectors)

    color_by_task_type = False
    show_names = True

    plt.figure(figsize=(7, 7))

    if color_by_task_type:
        # # color based on the first part of the task name - the task type (split on "_")
        task_types = [task_name.split("_")[0] for task_name in task_vectors.keys()]
        unique_task_types = list(set(task_types))
        task_type_to_color = {task_type: f"C{i}" for i, task_type in enumerate(unique_task_types)}
        c = [task_type_to_color[task_type] for task_type in task_types]
        c = np.repeat(c, 50, axis=0)
    else:
        # color based on the full task name (we need 18 colors)
        c = task_vectors_labels

    plt.scatter(task_vectors_2d[:, 0], task_vectors_2d[:, 1], c=c, cmap="tab20")

    if show_names:
        # write the task name next the first vector for each task
        for i, task_name in enumerate(task_vectors.keys()):
            # make the text apear centered below the cluster
            xs = task_vectors_2d[i * 50 : (i + 1) * 50, 0]
            ys = task_vectors_2d[i * 50 : (i + 1) * 50, 1]
            x = np.mean(xs)
            y = np.min(ys) - 0.1
            plt.text(x, y, task_name, fontsize=6, ha="center", va="top")

    # save the plot
    name_suffix = "task_type" if color_by_task_type else "task_name"
    save_path = os.path.join(FIGURES_DIR, f"task_vectors_tsne_{name_suffix}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)

def get_task_vector_distances(task_vectors):
    within_task_distances = torch.stack(
        [
            torch.tensor(cdist(task_vectors[task_name], task_vectors[task_name], metric="cosine").flatten())
            for task_name in task_vectors.keys()
        ]
    )

    between_task_distances = torch.stack(
        [
            torch.cat(
                [
                    torch.tensor(
                        cdist(task_vectors[task_name], task_vectors[other_task_name], metric="cosine").flatten()
                    )
                    for other_task_name in task_vectors.keys()
                    if task_name != other_task_name
                ]
            )
            for task_name in task_vectors.keys()
        ]
    )

    return within_task_distances, between_task_distances

def create_histograms_plot(task_vectors):
    # calculate task vectors distances - within task and between tasks
    within_task_distances, between_task_distances = get_task_vector_distances(task_vectors)

    # Create batches of 20 tasks
    task_names = list(task_vectors.keys())
    num_batches = (len(task_names) + 19) // 20  # Round up division

    for batch in range(num_batches):
        start_idx = batch * 20
        end_idx = min((batch + 1) * 20, len(task_names))
        batch_task_names = task_names[start_idx:end_idx]

        # create subplots for each batch of tasks
        fig, axs = plt.subplots(5, 4, figsize=(12, 16))
        fig.suptitle(f"Task Vector Distances (Batch {batch + 1})", fontsize=16)
        axs = axs.flatten()

        for i, task_name in enumerate(batch_task_names):
            task_idx = task_names.index(task_name)
            axs[i].hist(within_task_distances[task_idx], bins=50, alpha=0.5, label="Within task", density=True)
            axs[i].hist(between_task_distances[task_idx], bins=50, alpha=0.5, label="Other tasks", density=True)
            axs[i].set_title(task_name, fontsize=10)
            axs[i].legend(fontsize=8)
            axs[i].tick_params(axis='both', which='major', labelsize=6)

        # x label (only for the bottom row)
        for ax in axs[-4:]:
            ax.set_xlabel("Cosine Distance", fontsize=10)

        # y label (only for the left column)
        for ax in axs[::4]:
            ax.set_ylabel("Density", fontsize=10)

        # delete the empty plots
        for ax in axs[len(batch_task_names):]:
            ax.remove()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to accommodate suptitle

        # save the figure, high dpi, without large margins
        save_path = os.path.join(FIGURES_DIR, f"task_vectors_histograms_batch_{batch + 1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free up memory


def print_histograms_stats(task_vectors):
    # calculate task vectors distances - within task and between tasks
    within_task_distances, between_task_distances = get_task_vector_distances(task_vectors)

    table_data = {
        "Task": TASKS_TO_PLOT,
        "in_task_mean": within_task_distances.mean(axis=1),
        "in_task_std": within_task_distances.std(axis=1),
        "other_tasks_mean": between_task_distances.mean(axis=1),
        "other_tasks_std": between_task_distances.std(axis=1),
    }

    df = pd.DataFrame(table_data)

    # rename column names from "knowledge_country_capital" to "Knowledge Country Capital"
    df.columns = df.columns.str.replace("_", " ").str.title()

    # rename task names from "knowledge_country_capital" to "Knowledge Country Capital"
    df["Task"] = df["Task"].str.replace("_", " ").str.title()

    # remove "Present Simple" from task names that contain it (e.g. "Inflection Present Simple Gerund" -> "Inflection Gerund")
    df["Task"] = df["Task"].apply(lambda task_name: task_name.replace("Present Simple", "").strip())

    # remove the first word from the task name (e.g. "Knowledge Country Capital" -> "Country Capital", "Translation Fr En" -> "Fr En") only if the task name has more than 2 words
    df["Task"] = df["Task"].apply(
        lambda task_name: " ".join(task_name.split(" ")[1:]) if len(task_name.split(" ")) > 2 else task_name
    )

    # round numbers to 3 decimal places (withh trailing zeros)
    df = df.round(3)

    # remove theh index column
    df = df.set_index("Task")

    # print as markdown
    print(df.to_markdown())


def main():
    seed_everything(41)
    limit_gpus(range(0, 8))

    model_type, model_variant = "pythia", "12B"
    model, tokenizer = load_model_and_tokenizer(model_type, model_variant)

    task_vectors = create_task_vectors(model, tokenizer)

    create_tsne_plot(task_vectors)
    create_histograms_plot(task_vectors)
    print_histograms_stats(task_vectors)


if __name__ == "__main__":
    main()
