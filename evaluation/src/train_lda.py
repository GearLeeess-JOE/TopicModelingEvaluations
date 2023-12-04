import logging
from octis.models.LDA import LDA
from utilis import get_metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("gensim").setLevel(logging.ERROR)

def _lda(data, params):
    """Train LDA model."""
    print(f"Training LDA model on parameters: {params}")
    # Ingore training logging messages from now

    model = LDA(**params)  # Create model
    model_output = model.train_model(data) # Train the model

    logging.info("LDA model trained")
    metrics = get_metrics(data.get_corpus(), model_output)
    return metrics

