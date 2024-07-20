import os
import pickle
import pandas as pd


from scripts.utils import main_experiment_results_dir, overriding_experiment_results_dir

MODEL_DISPLAY_NAME_MAPPING = {
    "llama_7B": "LLaMA 7B",
    "llama_13B": "LLaMA 13B",
    "llama_30B": "LLaMA 30B",
    "gpt-j_6B": "GPT-J 6B",
    "pythia_2.8B": "Pythia 2.8B",
    "pythia_6.9B": "Pythia 6.9B",
    "pythia_12B": "Pythia 12B",
}

TASKS_TO_PLOT = [
    "first_letter",
    "next_letter",
    "prev_letter",
    "city_country",
    "country_capital",
    "location_country",
    "antonyms",
    "past_present",
    "present_gerund",
    "en_es",
    "array_min",
    "array_max",
    "absolute",
    "round",
    "array_length",
    "division",
    "subtraction",
    "country_PM",
    "city_mayor",
    "person_birthYear",
    "reversed_words",
    "word_length",
    "abs_round",
    "array_average",
    "first_letter_to_upper",
    "next_of_first_letter",
    "prev_of_first_letter",
    "city_PM",
    "city_PMbirthYear",
    "country_capitalMayor",
    "location_capitalMayor",
    "past_oppositeGerund"
]


def load_main_results(experiment_id: str = "camera_ready"):
    results = {}
    experiment_dir = main_experiment_results_dir(experiment_id)
    
    expected_models = [
        "llama_7B", "llama_13B", "llama_30B",
        "gpt-j_6B",
        "pythia_2.8B", "pythia_6.9B", "pythia_12B"
    ]

    for model_name in expected_models:
        file_name = f"{model_name}.pkl"
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

def extract_accuracies(results, new_format=False):
    accuracies = {}
    for model_name, model_results in results.items():
        accuracies[model_name] = {}
        for task_name, task_results in model_results.items():
            if new_format:
                best_num_examples = max(task_results["tv_accuracy"], key=task_results["tv_accuracy"].get)
                accuracies[model_name][task_name] = {
                    "bl": task_results["baseline_accuracy"],
                    "icl": task_results["icl_accuracy"][best_num_examples],
                    "tv": task_results["tv_accuracy"][best_num_examples],
                }
            else:
                accuracies[model_name][task_name] = {
                    "bl": task_results["baseline_accuracy"],
                    "icl": task_results["icl_accuracy"],
                    "tv": task_results["tv_accuracy"],
                }

    return accuracies

def create_accuracies_df(results, new_format=False):
    accuracies = extract_accuracies(results, new_format)

    data = []
    for model_name, model_acc in accuracies.items():
        for task_full_name, task_acc in model_acc.items():
            if task_full_name not in TASKS_TO_PLOT:
                continue
            task_type = task_full_name.split("_")[0]
            task_name = "_".join(task_full_name.split("_")[1:])

            data.append([model_name, task_type, task_name, "Baseline", task_acc["bl"]])
            data.append([model_name, task_type, task_name, "Hypothesis", task_acc["tv"]])
            data.append([model_name, task_type, task_name, "Regular", task_acc["icl"]])

    df = pd.DataFrame(data, columns=["model", "task_type", "task_name", "method", "accuracy"])

    df["model"] = df["model"].map(MODEL_DISPLAY_NAME_MAPPING)

    # order the tasks by alphabetical order, using the task_full_name
    task_order = sorted(zip(df["task_type"].unique(), df["task_name"].unique()), key=lambda x: x[0])
    task_order = [x[1] for x in task_order]

    # df["task_name"] = pd.Categorical(df["task_name"], categories=task_order, ordered=True)

    return df


def create_grouped_accuracies_df(accuracies_df):
    grouped_accuracies_df = accuracies_df.pivot_table(
        index=["model", "task_type", "task_name"], columns="method", values="accuracy", aggfunc="first"
    ).reset_index()
    return grouped_accuracies_df
