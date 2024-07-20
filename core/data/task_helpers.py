import string
from typing import Dict

from core.data.tasks.increment_task import IncrementTask
from core.data.tasks.list_operation_task import ListOperationTask
from core.data.tasks.token_operation_task import TokenOprationTask
from core.data.tasks.mapping_task import MappingTask
from core.data.tasks.translation_task import TranslationTask
from core.data.tasks.task import Task

from transformers import PreTrainedTokenizer

TASK_TYPE_TO_CLASS = {
    "increment": IncrementTask,
    "list_operation": ListOperationTask,
    "token_operation": TokenOprationTask,
    "mapping": MappingTask,
    "translation": TranslationTask,
    # "sentiment": SentimentTask,
}


ALL_TASKS = {
    # Algorithmic
    "algorithmic_next_letter": {
        "task_type": "increment",
        "task_kwargs": {"increment": +1},
    },
    "algorithmic_prev_letter": {
        "task_type": "increment",
        "task_kwargs": {"increment": -1},
    },
    "algorithmic_list_first": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "first", "list_lenghts": range(2, 5)},
    },
    "algorithmic_list_last": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "last", "list_lenghts": range(2, 5)},
    },
    "algorithmic_list_min": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "min", "list_lenghts": range(2, 5), "elements_space": list(string.digits)},
    },
    "algorithmic_list_max": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "max", "list_lenghts": range(2, 5), "elements_space": list(string.digits)},
    },
    "algorithmic_list_length": {
        "task_type": "list_operation",
        "task_kwargs": {"operation": "length", "list_lenghts": range(1, 4)},
    },
    "algorithmic_to_upper": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "to_upper", "input_space": list(string.ascii_lowercase)},
    },
    "algorithmic_to_lower": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "to_lower", "input_space": list(string.ascii_uppercase)},
    },
    "algorithmic_char_to_int": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "char_to_int", "input_space": list(string.ascii_lowercase[:9])},
    },  # low performance
    "algorithmic_int_to_char": {
        "task_type": "token_operation",
        "task_kwargs": {"operation": "int_to_char", "input_space": list(string.digits[1:])},
    },
    # Translation
    "translation_fr_en": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "fr_en"},
    },
    "translation_it_en": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "it_en"},
    },
    "translation_es_en": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "es_en"},
    },
    "translation_en_fr": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_fr"},
    },
    "translation_en_it": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_it"},
    },
    "translation_en_es": {
        "task_type": "translation",
        "task_kwargs": {"mapping_type": "translation", "mapping_name": "en_es"},
    },
    # Linguistic
    "linguistic_present_simple_gerund": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_gerund"},
    },
    "linguistic_present_simple_past_simple": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_simple"},
    },
    "linguistic_present_simple_past_perfect": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "present_simple_past_perfect"},
    },
    "linguistic_singular_plural": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "singular_plural"},
    },
    "linguistic_plural_singular": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "plural_singular"},
    },
    "linguistic_antonyms": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "linguistic", "mapping_name": "antonyms"},
    },
    # Knowledge
    "knowledge_country_capital": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "country_capital", "allow_prefix": True},
    },
    "knowledge_person_language": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "person_language", "allow_prefix": True},
    },
    "knowledge_location_continent": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "location_continent", "allow_prefix": True},
    },
    "knowledge_location_religion": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "knowledge", "mapping_name": "location_religion", "allow_prefix": True},
    },
    # "sentiment": {
    #     "task_type": "sentiment",
    #     "task_kwargs": {"allow_prefix": True},
    # },
    # Custom
    "next_letter": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "next_letter"},
    },
    "prev_letter": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "prev_letter"},
    },
    "to_uppercase": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "to_uppercase"},
    },
    "count_char_in_string": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "count_char_in_string"},
    },
    "abs_round": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "abs_round"},
    },
    "array_average": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "array_average"},
    },
    "array_min": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "array_min"},
    },
    "array_sum": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "array_sum"},
    },
    "division": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "division"},
    },
    "round": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "round"},
    },
    "string_to_mask_by_char": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "string_to_mask_by_char"},
    },
    "string_to_mask_by_char2": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "string_to_mask_by_char2"},
    },
    "absolute": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "absolute"},
    },
    "array_length": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "array_length"},
    },
    "array_max": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "array_max"},
    },
    "subtraction": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "subtraction"},
    },
    "array_max_diff": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "array_max_diff"},
    },
    "first_letter": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "first_letter"},
    },
    "first_letter_to_upper": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "first_letter_to_upper"},
    },
    "next_of_first_letter": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "next_of_first_letter"},
    },
    "prev_of_first_letter": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "prev_of_first_letter"},
    },
    "city_PM": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "city_PM", "allow_prefix": True},
    },
    "city_country": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "city_country", "allow_prefix": True},
    },
    "country_PM": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "country_PM", "allow_prefix": True},
    },
    "country_capital": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "country_capital", "allow_prefix": True},
    },
    "location_country": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "location_country", "allow_prefix": True},
    },
    "city_PMbirthYear": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "city_PMbirthYear"},
    },
    "city_mayor": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "city_mayor", "allow_prefix": True},
    },
    "country_capitalMayor": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "country_capitalMayor", "allow_prefix": True},
    },
    "location_capitalMayor": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "location_capitalMayor", "allow_prefix": True},
    },
    "person_birthYear": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "person_birthYear"},
    },
    "adjective_superlative": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "adjective_superlative"},
    },
    "antonym_superlative": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "antonym_superlative"},
    },
    "antonyms": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "antonyms"},
    },
    "past_oppositeGerund": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "past_oppositeGerund"},
    },
    "past_present": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "past_present"},
    },
    "present_gerund": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "present_gerund"},
    },
    "antoym_to_reversed": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "antoym_to_reversed"},
    },
    "en_es": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "en_es"},
    },
    "count_char_in_string": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "count_char_in_string"},
    },
    "division_new_format": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "division_new_format"},
    },
    "reversed_words": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "reversed_words", "allow_prefix": True},
    },
    "subtract_new_format": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "subtract_new_format"},
    },
    "word_length": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "word_length"},
    },
}

COMPLEX_TASKS = {
    "first_letter_to_upper": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "first_letter_to_upper"},
        "sub_tasks": ["first_letter", "to_uppercase"],
    },
    "next_of_first_letter": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "next_of_first_letter"},
        "sub_tasks": ["first_letter", "next_letter"],
    },
    "prev_of_first_letter": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "prev_of_first_letter"},
        "sub_tasks": ["first_letter", "prev_letter"],
    },
    "count_char_in_string": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "count_char_in_string"},
        "sub_tasks": ["string_to_mask_by_char", "array_sum"],
    },
    "abs_round": {
        "task_type": "mapping",
        "task_kwargs": {"mapping_type": "custom", "mapping_name": "abs_round"},
        "sub_tasks": ["round", "absolute"],
    },
}


def get_task(task_type: str, task_kwargs: Dict[str, str], tokenizer: PreTrainedTokenizer) -> Task:
    task = TASK_TYPE_TO_CLASS[task_type](**task_kwargs, tokenizer=tokenizer)
    return task


def get_task_by_name(tokenizer: PreTrainedTokenizer, task_name: str) -> Task:
    task_args = ALL_TASKS[task_name]
    task = get_task(task_args["task_type"], task_args["task_kwargs"], tokenizer)
    return task


def get_all_tasks(tokenizer: PreTrainedTokenizer) -> Dict[str, Task]:
    tasks = {task_name: get_task_by_name(tokenizer, task_name) for task_name in ALL_TASKS}
    return tasks

def get_all_complex_tasks(tokenizer: PreTrainedTokenizer) -> Dict[str, Task]:
    tasks = {task_name: get_task_by_name(tokenizer, task_name) for task_name in COMPLEX_TASKS}
    return tasks