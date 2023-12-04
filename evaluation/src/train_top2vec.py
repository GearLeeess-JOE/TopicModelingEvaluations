from top2vec import Top2Vec
from utilis import get_metrics
import logging


def _top2vec(data, params):
    """Train Top2Vec model and process the output topics.

    Args:
    - corpus_data: Data object containing the corpus.
    - model_params: Dictionary of parameters for Top2Vec.

    Returns:
    - List of processed topics with validated words.
    """
    # Convert the corpus data into a format suitable for Top2Vec
    formatted_documents = [" ".join(doc) for doc in data.get_corpus()]
    params['documents'] = formatted_documents

    # Create and train the Top2Vec model
    top2vec_model = Top2Vec(**params)

    # Attempt hierarchical topic reduction if specified
    target_topic_count = params.get('nr_topics')
    is_reduction_applied = False
    if target_topic_count:
        try:
            top2vec_model.hierarchical_topic_reduction(target_topic_count)
            is_reduction_applied = True
        except Exception as error:
            print(f"Topic reduction error: {error}")

    # Fetching topics from the model
    topics = top2vec_model.get_topics(reduced=is_reduction_applied)[0]

    # Ensuring topic words are present in the original corpus
    unique_corpus_words = set(word for document in data.get_corpus() for word in document)
    processed_topics = []
    for topic in topics:
        validated_words = [word if word in unique_corpus_words else next(iter(unique_corpus_words)) 
                           for word in topic[:10]]
        processed_topics.append(validated_words)

    # Update model parameters based on actual topic processing
    if not is_reduction_applied:
        params.update({'nr_topics': len(processed_topics), 'reduction': False})

    return processed_topics

def _top2vec(data, params):
    """Train Top2Vec"""
    nr_topics = None
    data = data.get_corpus()
    texts = [" ".join(words) for words in data]

    params["documents"] = data

    model = Top2Vec(**params)

    if nr_topics:
        try:
            _ = model.hierarchical_topic_reduction(nr_topics)
            params["reduction"] = True
            params["nr_topics"] = nr_topics
        except:
            params["reduction"] = False
            nr_topics = False


    if nr_topics:
        topic_words, _, _ = model.get_topics(reduced=True)
    else:
        topic_words, _, _ = model.get_topics(reduced=False)

    topics_old = [list(topic[:10]) for topic in topic_words]
    all_words = [word for words in data for word in words]
    model_output = []
    for topic in topics_old:
        words = []
        for word in topic:
            if word in all_words:
                words.append(word)
            else:
                print(f"error: {word}")
                words.append(all_words[0])
        model_output.append(words)

    if not nr_topics:
        params["nr_topics"] = len(model_output)
        params["reduction"] = False

    return model_output