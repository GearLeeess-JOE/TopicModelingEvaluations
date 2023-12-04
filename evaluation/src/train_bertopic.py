from bertopic import BERTopic
from utilis import get_metrics
from sentence_transformers import SentenceTransformer
import logging

# class TextEmbedding:
#     def __init__(self, model_id, texts):
#         self.model_id = model_id
#         self.texts = texts
#         self.embeddings = self.generate_embeddings()

#     def generate_embeddings(self):
#         """Generate embeddings for the data"""
#         # load model
#         self.model = SentenceTransformer(self.model_id)
#         # Generate embeddings
#         embeddings = self.model.encode(self.texts, show_progress_bar=True)
#         return embeddings
    
#     def get_embeddings(self):
#         """Get embeddings for the data"""
#         return self.embeddings

def _bertopic(data, params):
    """Train BERTopic model"""
    # Prepare BERTopic model
    params["calculate_probabilities"] = False
    model_id = params["embedding_model"]

    data = data.get_corpus()
    texts = [" ".join(words) for words in data]
    
    embeddings = SentenceTransformer(model_id).encode(texts, show_progress_bar=True)

    logging.info(f"Training BERTopic model on parameters: {params}")
    model = BERTopic(**params)
    topics, _ = model.fit_transform(texts, embeddings)

    # Flatten the data to create a single list of words
    corpus = [word for words in data for word in words]

    # Generate topics using BERTopic and ensure topic words are in the corpus
    model_output = []
    for i in range(len(set(topics)) - 1):
        topic_words = []
        for vals in model.get_topic(i)[:10]:
            # Add the word if it's in the corpus, otherwise add the first word of the corpus
            topic_word = vals[0] if vals[0] in corpus else corpus[0]
            topic_words.append(topic_word)
        model_output.append(topic_words)

    logging.info("BERTopic model trained")
    metrics = get_metrics(data, model_output)
    return metrics
