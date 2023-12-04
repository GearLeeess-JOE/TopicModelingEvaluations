from octis.dataset.dataset import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(dataset: str):
    """Load data from a specified file path or standard dataset."""
    try:
        data = Dataset()
        data.fetch_dataset(dataset)
    except:
        logging.error(f"You are using custom dataset: {dataset}")
        data.load_custom_dataset_from_folder(dataset)
    return data