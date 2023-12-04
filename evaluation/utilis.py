import logging

from bertopic import BERTopic
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.models.CTM import CTM
from octis.models.LDA import LDA
from sentence_transformers import SentenceTransformer
from top2vec import Top2Vec

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("gensim").setLevel(logging.ERROR)

def get_metrics(corpus, model_output, topk=10):
    """Prepare evaluation measures using OCTIS."""
    # Define metrics
    cv = Coherence(texts=corpus, topk=topk, measure="c_v")
    uci = Coherence(texts=corpus, topk=topk, measure="c_uci")
    npmi = Coherence(texts=corpus, topk=topk, measure="c_npmi")
    umass = Coherence(texts=corpus, topk=topk, measure="u_mass")
    topic_diversity = TopicDiversity(topk=topk)

    # Define methods
    coherence_cv = [(cv, "cv")]
    coherence_npmi = [(npmi, "npmi")]
    coherence_uci = [(uci, "uci")]
    coherence_umass = [(umass, "umass")]
    diversity = [(topic_diversity, "diversity")]
    metric_functions = [
        (coherence_cv, "Coherence_cv"),
        (coherence_npmi, "Coherence_npmi"),
        (coherence_uci, "Coherence_uci"),
        (coherence_umass, "Coherence_umass"),
        (diversity, "Topic Diversity"),
    ]

    metrics = {}
    for scorers, _ in metric_functions:
        for scorer, name in scorers:
            score = scorer.score(model_output)
            metrics[name] = float(score)
            
    return metrics

def save_model(model, output_path):
    """Save the trained model."""
    try:
        model.save(output_path)
        logging.info(f"Model saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")
        raise

def _lda(data, params):
    """Train LDA model."""
    print(f"Training LDA model on parameters: {params}")
    # Ingore training logging messages from now

    model = LDA(**params)  # Create model
    model_output = model.train_model(data) # Train the model

    logging.info("LDA model trained")
    metrics = get_metrics(data.get_corpus(), model_output)
    return metrics

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

def _bertopic(data, params):
    """Train BERTopic model"""
    # Prepare BERTopic model
    data = data.get_corpus()
    texts = [" ".join(words) for words in data]

    model_id = params["embedding_model"]
    embeddings = SentenceTransformer(model_id).encode(texts, show_progress_bar=True)

    # Prepare BERTopic model Parameters
    umap_model = None
    hdbscan_model = None
    representation_model = None
    # Check if parameters are specified
    Bert_params = params.copy()
    if Bert_params.get("UMAP_PARAMS"):
        from umap import UMAP
        umap_model = UMAP(**Bert_params["UMAP_PARAMS"], 
                          random_state=190014610)
        Bert_params.pop("UMAP_PARAMS")
    if Bert_params.get("HDBSCAN_PARAMS"):
        import hdbscan
        hdbscan_model = hdbscan.HDBSCAN(**Bert_params["HDBSCAN_PARAMS"])
        Bert_params.pop("HDBSCAN_PARAMS")
    if Bert_params.get("representation_model"):
        if Bert_params["representation_model"] == "KeyBERT":
            from bertopic.representation import KeyBERTInspired
            representation_model = KeyBERTInspired()
        elif Bert_params["representation_model"] == "MMR":
            from bertopic.representation import MaximalMarginalRelevance
            representation_model = MaximalMarginalRelevance(diversity=0.5)
        else:
            raise ValueError("Invalid representation model")
        Bert_params.pop("representation_model")

    logging.info(f"Training BERTopic model on parameters: {params}")
    # Fine-tune topic representations
    model = BERTopic(umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    representation_model=representation_model,
                    calculate_probabilities=False,
                    **Bert_params)
    topics, _ = model.fit_transform(texts, embeddings)

    # Flatten the data to create a single list of words
    corpus = [word for document in data for word in document]

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
    metrics = get_metrics(data, {"topics":model_output})
    # Get Topic_size
    metrics['result_num_topic'] = len(model_output)
    return metrics

def _top2vec(data, params):
    """Train Top2Vec model and process the output topics.

    Args:
    - corpus_data: Data object containing the corpus.
    - model_params: Dictionary of parameters for Top2Vec.

    Returns:
    - List of processed topics with validated words.
    """
    # Convert the corpus data into a format suitable for Top2Vec
    data = data.get_corpus()
    texts = [" ".join(words) for words in data]
    # Get the embedding model
    model_id = params["embedding_model"]
    if model_id != "doc2vec":
        use_embedding_model_tokenizer = True
    else:
        use_embedding_model_tokenizer = False
        
    # Create and train the Top2Vec model
    logging.info(f"Training Top2Vec model on parameters: {params}")
    # Create a tailored Top2Vec model parameter dictionary
    top2vec_params = params.copy()
    top2vec_params.pop("num_topics")
    top2vec_model = Top2Vec(documents=texts,
                            min_count=10,
                            use_embedding_model_tokenizer=use_embedding_model_tokenizer,
                              **top2vec_params)

    # Attempt hierarchical topic reduction if specified
    num_topic = params.get('num_topics')
    is_reduction_applied = False
    if num_topic:
        try:
            logging.info(f"Applying hierarchical topic reduction with {num_topic} topics")
            top2vec_model.hierarchical_topic_reduction(num_topic)
            is_reduction_applied = True
        except Exception as error:
            print(f"Topic reduction error: {error}")

    # Fetching topics from the model
    topics = top2vec_model.get_topics(reduced=is_reduction_applied)[0]

    # Ensuring topic words are present in the original corpus
    corpus = set(word for document in data for word in document)
    model_output = []
    for topic in topics:
        validated_words = [word if word in corpus else next(iter(corpus)) 
                           for word in topic[:10]]
        model_output.append(validated_words)

    # Update model parameters based on actual topic processing
    if not is_reduction_applied:
        params.update({'num_topic': len(model_output), 'reduction': False})

    logging.info("Top2Vec model trained")
    metrics = get_metrics(data, {"topics":model_output})
    return metrics