import json
import random
import re

def sort_json_by_values(input_file, output_file):
    # Read the JSON data from the file
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Sort the data by values
    sorted_data = dict(sorted(data.items(), key=lambda item: (item[1], len(item[0]))))
    
    # Write the sorted data back to a new JSON file
    with open(output_file, 'w') as file:
        json.dump(sorted_data, file, indent=4)
    
    print(f"Sorted JSON data has been written to {output_file}")

def shuffle_json_file(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)
    
    # Convert the dictionary items to a list of tuples and shuffle it
    items = list(data.items())
    random.shuffle(items)
    
    # Convert the shuffled list of tuples back to a dictionary
    shuffled_data = dict(items)
    
    # Write the shuffled data back to a JSON file
    with open(output_file, 'w') as file:
        json.dump(shuffled_data, file, indent=4)

def create_list_of_unique_values(input_file):
    # Load JSON data from file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract values (assuming all values are strings as in your example)
    values = list(data.values())

    # Get unique values using a set
    unique_values = set(values)

    # Convert set back to list if needed
    unique_values_list = list(unique_values)

    # Print or use unique_values_list as needed
    print(unique_values_list)

def composite_json(input1, input2, output):
    # Read file1.json (A to B mappings)
    with open(input1, 'r') as f:
        a_to_b = json.load(f)

    # Read file2.json (B to C mappings)
    with open(input2, 'r') as f:
        b_to_c = json.load(f)

    # Create a dictionary for A to C mappings
    a_to_c = {}

    # Iterate over A to B mappings
    for a, b in a_to_b.items():
        if b in b_to_c:
            a_to_c[a] = b_to_c[b]

    # Write to output.json
    with open(output, 'w') as f:
        json.dump(a_to_c, f, indent=2)

    print("Composition A to C has been written to output.json.")

def transform_json(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    transformed_data = {key: values[0] for key, values in data.items()}
    
    with open(output_file, 'w') as outfile:
        json.dump(transformed_data, outfile, indent=4)

def switch_key_value(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)
    
    switched_data = {str(value): key for key, value in data.items()}
    
    with open(output_file, 'w') as outfile:
        json.dump(switched_data, outfile, indent=4)

def gerund_form(verb):
    # Rule 4: If a verb ends in "ie", change "ie" to "y" and add "ing"
    if re.search(r'ie$', verb):
        return re.sub(r'ie$', 'ying', verb)
    
    # Rule 5: If a verb ends in "ee", simply add "ing"
    if re.search(r'ee$', verb):
        return verb + 'ing'
    
    # Rule 2: If a verb ends in "e", drop the "e" and add "ing"
    if re.search(r'e$', verb):
        return re.sub(r'e$', 'ing', verb)
    
    # Rule 3: If a verb ends in a consonant-vowel-consonant (CVC) pattern, double the final consonant and add "ing"
    v = {'a','e','i','o','u'}
    if len(verb) > 2 and verb[-1] not in v and verb[-2] in v and verb[-3] not in v:
        # Ensure the final consonant is stressed, here we assume single syllable words
        return re.sub(r'([^aeiou])$', r'\1\1ing', verb)
    
    # Rule 1: For most verbs, simply add "ing"
    return verb + 'ing'

def match_present_to_gerund(input_file1, output_file):
    with open(input_file1, 'r') as infile:
        data1 = json.load(infile)
    
    composed_data = {key: [gerund_form(value) for value in values] for key, values in data1.items()}
    
    with open(output_file, 'w') as outfile:
        json.dump(composed_data, outfile, indent=4)
