from utilis import _lda, _ctm, _bertopic ,_top2vec

BENCHMARK ={
    "Dataset": "/home/kw215/Documents/research_codes/Topic-modeling-evaluations/evaluation/modactions",
    # "Dataset":"20NewsGroup",
    "Models":["BERTopic"]
}

MODELS = {
    "LDA": _lda,
    "CTM": _ctm,
    "BERTopic": _bertopic,
    "Top2Vec": _top2vec
}

MODEL_CONFIG = {
    "LDA": {
        "num_topics": [30, 50, 100],
        # "alpha": ['auto'], #[0.1, 0.5, 0.9],
        # "eta": [0.5] #[0.1, 0.5, 0.9]
    },
    "CTM": {
        "num_topics": [30, 50, 100],
        "bert_model": ["all-MiniLM-L12-v2"],
        # "passes": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        # "alpha": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9],
        # "eta": [0.01, 0.1, 0.3, 0.5, 0.7, 0.9]
    },
    "BERTopic": {
        "nr_topics": [50],
        "min_topic_size": [20, 50], # Lower min_topic_size gets lower diversity
        # "embedding_model": ["paraphrase-multilingual-MiniLM-L12-v2","all-MiniLM-L12-v2"],
        "embedding_model": ["paraphrase-multilingual-MiniLM-L12-v2", "/home/kw215/Documents/research_codes/Topic-modeling-evaluations/evaluation/tsdae_all_MiniLM-v12-v2-drug-submissions"],
        # "embedding_model": ["/home/kw215/Documents/research_codes/Topic-modeling-evaluations/evaluation/tsdae_all-MiniLM-L12-v2-3epoch"],      
        # all-mpnet-base-v2 performed worst for only resulted in less than 10 topics
        # "embedding_model": ["/home/kw215/Documents/research_codes/Topic-modeling-evaluations/evaluation/tsdae_roberta_base_drug_submissions"],
        # UMAP Parameters
        # "UMAP_PARAMS": [{
        #     'n_neighbors': 10,
        #     'n_components': 15, 
        # },
        # {
        #     'n_neighbors': 5,
        #     'n_components': 8,
        # }],
        # "HDBSCAN_PARAMS": [{
        #     'metric': 'euclidean',
        #     'cluster_selection_method': 'eom', # eom, leaf
        #     'prediction_data': True
        # }],
        "representation_model": ["KeyBERT"],#"MMR"
        "n_gram_range": [(1, 1), (1, 3)]
    },
    "Top2Vec": {
        "num_topics": [30, 50, 100],
        "embedding_model": ["all-MiniLM-L12-v2"],
    }
}

