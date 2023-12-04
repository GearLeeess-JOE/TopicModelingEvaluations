import logging
from utilis import get_metrics
from octis.models.CTM import CTM


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("gensim").setLevel(logging.ERROR)

def _ctm(data, params):
    """Train CTM model."""
    print(f"Training CTM model on parameters: {params}")
    # Add additional parameters
    params["use_partitions"] = False
    params["num_epochs"] = 50

    model = CTM(**params)  # Create model
    # Ingore logging from CTM
    model_output = model.train_model(data) # Train the model

    logging.info("CTM model trained")
    metrics = get_metrics(data.get_corpus(), model_output)
    return metrics