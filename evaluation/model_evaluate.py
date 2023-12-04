from octis.dataset.dataset import Dataset
from itertools import product
from config import BENCHMARK, MODEL_CONFIG, MODELS
import logging
import json
import os
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset: str = BENCHMARK["Dataset"]):
    """Load data from a specified file path or standard dataset."""
    try:
        data = Dataset()
        data.fetch_dataset(dataset)
    except:
        logging.info(f"You are using custom dataset: {dataset}")
        data.load_custom_dataset_from_folder(dataset)
    return data

def load_model_params(params):
    """Load model parameters."""
    # Generating all possible combinations of parameters
    parameter_combinations = list(product(*params.values()))

    # Converting tuples to dictionaries
    param_list = []
    for combination in parameter_combinations:
        parameter = dict(zip(params.keys(), combination))
        param_list.append(parameter)
    return param_list

def create_metadata(model_name, data):
    """Create metadata for the evaluation."""
    metadata = {}
    metadata['Model'] = model_name
    dataset_name = BENCHMARK["Dataset"]
    # only pick the file name
    metadata['Dataset'] = dataset_name.split("/")[-1]
    metadata['Dataset_size'] = len(data.get_corpus())
    return metadata

def train_model(model_name, model, data, params):
    """Train the model against sets of params."""
    param_list = load_model_params(params)
    results = []
    metadata = create_metadata(model_name, data)
    results.append(metadata)

    # Train the model with each set of parameters
    for param in param_list:
        result = {}
        metrics = model(data, param)
        result["metrics"] = metrics
        result["params"] = param
        results.append(result)
    logging.info(f"Model trained with params: {params}")
    return results

def save_results(results, output_path):
    """Save the metrics."""
    # Using os to create the directory if it doesn't exist
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    with open(output_path, "w") as f:
        json.dump(results, f)
    logging.info(f"Results saved to {output_path}")

def main():
    """Run the evaluation."""
    dataset = BENCHMARK["Dataset"]
    models = BENCHMARK["Models"]

    data = load_data(dataset)

    for model_name in models:
        params = MODEL_CONFIG[model_name]
        model = MODELS[model_name]

        results = train_model(model_name, model, data, params)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = f"./metrics_{timestamp}/{model_name}.json"
        save_results(results, output_path)

if __name__ == "__main__":
    main()