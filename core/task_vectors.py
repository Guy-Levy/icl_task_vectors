from typing import Dict, List, Optional, Tuple, Union, Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from core.analysis.evaluation import calculate_accuracy_on_datasets
from core.data.datasets.few_shot_dataset import FewShotDataset
from core.data.tasks.task import Task
from core.models.context_managers.forward_modifiers.hidden_injector import HiddenInjector
from core.models.utils.inference import (
    batch_forward,
    batch_generate,
    decode_predictions,
    get_input_type,
    modified_forward,
    tokenize_datasets,
    traced_forward,
)
from core.models.utils.llm_layers import get_layers
from core.utils.nested import nested_apply

import numpy as np
from scipy.stats import pearsonr
from typing import List, Tuple

def run_icl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    include_train: bool = True,
    print_samples: bool = False,
) -> List[str]:
    format_dataset_kwargs = {"include_train": include_train}
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs=format_dataset_kwargs)
    new_ids = batch_generate(model, tokenizer, inputs=inputs, generate_kwargs={"max_new_tokens": 3})
    # new_ids = batch_generate(model, tokenizer, inputs=inputs, generate_kwargs={"max_new_tokens": 5}) # TODO Guy: play with this
    predictions = decode_predictions(new_ids, tokenizer)

    if print_samples:
        num_samples = min(2, len(test_datasets))
        for i in range(num_samples):
            print(f"Sample {i + 1}:")
            print("Prompt:")
            print(tokenizer.decode(inputs['input_ids'][i]))
            print("Generated answer:")
            print(predictions[i])
            print()

    return predictions


def run_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
):
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
        multi_context=multi_context,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

    task_hiddens = get_task_hiddens(model, tokenizer, task, test_datasets, multi_context=multi_context)
    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens


def run_overriding_task_vector(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    overriding_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
):
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        overriding_datasets,
        layers_to_test=layers_to_test,
    )
    best_intermediate_layer = int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

    task_hiddens_datasets = test_datasets if overriding_datasets is None else overriding_datasets
    task_hiddens = get_task_hiddens(model, tokenizer, task, task_hiddens_datasets)

    predictions = modulated_generate(
        model,
        tokenizer,
        task,
        test_datasets,
        task_hiddens=task_hiddens,
        intermediate_layer=best_intermediate_layer,
        include_train=True,
    )

    return predictions, dev_accuracy_by_layer, task_hiddens


def get_multi_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
) -> torch.Tensor:
    inputs = tokenize_datasets(tokenizer, datasets)

    outputs, forward_trace = traced_forward(model, inputs=inputs)

    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]

    # for each dataset, average task hiddens from other datasets that did not include the test_input from the current dataset
    mask = torch.ones(len(datasets), len(datasets))
    for i, dataset in enumerate(datasets):
        for j, other_dataset in enumerate(datasets):
            if dataset.test_input in other_dataset.train_inputs or dataset.test_input == other_dataset.test_input:
                mask[i, j] = 0

    task_hiddens = torch.cat([task_hiddens[mask[i].bool()].mean(dim=0).unsqueeze(0) for i in range(len(datasets))])

    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer

    return task_hiddens  # (num_datasets, num_layers, hidden_size)


def get_single_context_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    num_test_inputs_to_avg: int = 2,
) -> torch.Tensor:
    new_datasets = [
        FewShotDataset(
            train_inputs=dataset.train_inputs,
            train_outputs=dataset.train_outputs,
            test_input=test_input,
            test_output=task.calc_output(test_input),
        )
        for dataset in datasets
        for test_input in task.sample_inputs(num_test_inputs_to_avg, exclude=(dataset.test_input,))
    ]

    inputs = tokenize_datasets(tokenizer, new_datasets)

    # TODO: replace traced forward with a regular forward and rely on huggingface's saved hidden states
    outputs, forward_trace = traced_forward(model, inputs=inputs)

    task_hiddens = forward_trace.residual_stream.hidden[:, :, -1, :]
    _, num_layers, hidden_size = task_hiddens.shape
    task_hiddens = task_hiddens.view(len(datasets), num_test_inputs_to_avg, num_layers, hidden_size).mean(dim=1)

    task_hiddens = task_hiddens[:, 1:]  # the first one is the embedding layer

    return task_hiddens  # (num_datasets, num_layers, hidden_size)


def get_task_hiddens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    multi_context: bool = False,
) -> torch.Tensor:
    if multi_context:
        return get_multi_context_task_hiddens(model, tokenizer, task, datasets)
    else:
        return get_single_context_task_hiddens(model, tokenizer, task, datasets)


def modulated_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    test_datasets: List[FewShotDataset],
    task_hiddens: torch.tensor,
    intermediate_layer: Union[int, torch.Tensor],
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    return_task_hiddens: bool = False,
    include_train: bool = False,
) -> List[str]:
    inputs = tokenize_datasets(tokenizer, test_datasets, format_dataset_kwargs={"include_train": include_train})

    first_forward_outputs = modulated_forward(
        model,
        inputs=inputs,
        task_hiddens=task_hiddens,
        intermediate_layer=intermediate_layer,
        past_key_values=past_key_values,
    )
    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)
    answers = decode_predictions(first_predicted_token_ids, tokenizer)

    if return_task_hiddens:
        return answers, task_hiddens
    return answers


def modulated_forward(
    model: PreTrainedModel,
    inputs: Dict,
    task_hiddens: torch.Tensor,
    intermediate_layer: int,
    batch_size: Optional[int] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
):
    # TODO: move all this to the HiddenInjector class
    if isinstance(intermediate_layer, int):
        intermediate_layer = torch.tensor(intermediate_layer).repeat(len(inputs["input_ids"]))
    injection_positions = -1 * torch.ones_like(intermediate_layer, dtype=torch.long)
    task_hiddens = task_hiddens[torch.arange(len(intermediate_layer)), intermediate_layer]

    forward_modifiers = [
        HiddenInjector(
            model,
            injection_layers=intermediate_layer,
            injection_positions=injection_positions,
            hiddens_to_inject=task_hiddens,
        )
    ]

    if past_key_values is not None:
        inputs[get_input_type(inputs)] = inputs[get_input_type(inputs)][:, -1].unsqueeze(1)

    first_forward_outputs = modified_forward(
        model,
        inputs=inputs,
        forward_kwargs={"past_key_values": past_key_values},
        forward_modifiers=forward_modifiers,
        batch_size=len(inputs["input_ids"]),  # TODO: need to enable batched forward with HiddenInjector
    )

    return first_forward_outputs


def task_vector_accuracy_by_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None,
    multi_context: bool = False,
) -> Dict[int, float]:
    if layers_to_test is None:
        num_layers = len(get_layers(model))
        layers_to_test = range(num_layers)

    # Get task hiddens
    task_hiddens = get_task_hiddens(model, tokenizer, task, datasets, multi_context=multi_context)

    # Get input past_key_values
    inputs = tokenize_datasets(tokenizer, datasets, format_dataset_kwargs={"include_train": False})
    outputs = batch_forward(model, inputs=inputs, forward_kwargs={"use_cache": True})
    past_key_values = outputs.past_key_values
    past_key_values = nested_apply(past_key_values, lambda x: x[:, :, :-1])  # remove last token from past_key_values
    inputs["input_ids"] = inputs["input_ids"][:, -1].unsqueeze(1)

    # Find best intermediate layer using dev set
    accuracies = []
    for layer_num in layers_to_test:
        answers = modulated_generate(
            model,
            tokenizer,
            task,
            datasets,
            intermediate_layer=layer_num,
            task_hiddens=task_hiddens,
            past_key_values=past_key_values,
        )

        accuracy = calculate_accuracy_on_datasets(task, answers, datasets)
        accuracies.append(accuracy)
    accuracy_by_layer = {layer: accuracy for layer, accuracy in zip(layers_to_test, accuracies)}

    return accuracy_by_layer


def continue_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: Dict,
    first_forward_outputs: CausalLMOutputWithPast,
    test_datasets: List[FewShotDataset],
) -> List[str]:
    """
    Continue generation after the first token. This is currently not supported.
    """
    first_predicted_token_ids = first_forward_outputs.logits[:, -1].argmax(dim=-1).unsqueeze(-1)

    new_input_ids = first_predicted_token_ids
    new_attention_mask = torch.ones_like(new_input_ids)

    full_input_ids = torch.cat([inputs["input_ids"], new_input_ids], dim=-1)
    full_attention_mask = torch.cat([inputs["attention_mask"], new_attention_mask], dim=-1)

    # full_input_ids = new_input_ids
    # full_attention_mask = new_attention_mask

    past_key_values = first_forward_outputs.past_key_values

    max_new_tokens = 1  # Right now we don't support multi-token outputs

    if max_new_tokens > 0:
        output_ids = model.generate(
            **{"input_ids": full_input_ids, "attention_mask": full_attention_mask},
            do_sample=False,
            max_new_tokens=max_new_tokens,
            past_key_values=past_key_values,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        output_ids = full_input_ids

    new_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    answers = decode_predictions(new_ids, tokenizer)

    return answers

def calculate_mean_tv(task_vectors: torch.Tensor) -> torch.Tensor:
    return torch.mean(task_vectors, dim=0)

def calculate_tv_distances(task_vectors: torch.Tensor, mean_tv: torch.Tensor) -> torch.Tensor:
    return torch.norm(task_vectors - mean_tv.unsqueeze(0), dim=1)

def correlate_distances_with_accuracies(distances: torch.Tensor, accuracies: List[float]) -> Tuple[float, float]:
    distances_np = distances.cpu().numpy()
    accuracies_np = np.array(accuracies)
    return pearsonr(distances_np, accuracies_np)

def find_best_intermediate_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None
) -> int:
    dev_accuracy_by_layer = task_vector_accuracy_by_layer(
        model,
        tokenizer,
        task,
        dev_datasets,
        layers_to_test=layers_to_test,
    )
    return int(max(dev_accuracy_by_layer, key=dev_accuracy_by_layer.get))

def get_task_accuracies(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    task_vectors: torch.Tensor,
    intermediate_layer: int
) -> List[float]:
    accuracies = []
    for dataset, tv in zip(datasets, task_vectors):
        prediction = modulated_generate(
            model, 
            tokenizer, 
            task, 
            [dataset], 
            task_hiddens=tv.unsqueeze(0),
            intermediate_layer=intermediate_layer
        )
        accuracies.append(calculate_accuracy_on_datasets(task, prediction, [dataset]))
    return accuracies

def analyze_tv_distance_accuracy_correlation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    task: Task,
    datasets: List[FewShotDataset],
    dev_datasets: List[FewShotDataset],
    layers_to_test: Optional[Iterable[int]] = None
) -> Tuple[float, float, int]:
    # Find the best intermediate layer
    best_layer = find_best_intermediate_layer(model, tokenizer, task, dev_datasets, layers_to_test)
    
    task_vectors = get_task_hiddens(model, tokenizer, task, datasets)

    relevant_task_vectors = task_vectors[:, best_layer]
    mean_tv = calculate_mean_tv(relevant_task_vectors)
    distances = calculate_tv_distances(relevant_task_vectors, mean_tv)

    accuracies = get_task_accuracies(model, tokenizer, task, datasets, task_vectors, best_layer)
    
    correlation, p_value = correlate_distances_with_accuracies(distances, accuracies)
    return correlation, p_value, best_layer