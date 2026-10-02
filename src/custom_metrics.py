import numpy as np

def precision_at_k(recommended, relevant, k=10):
    '''
    Params:
    recommended: list of item_id sorted by score in descending order (top-n).
    relevant: set of relevant item_id for the user.
    '''
    recom_k = recommended[:k]
    return len(set(recom_k).intersection(relevant)) / k

def recall_at_k(recommended, relevant, k=10):
    '''
    Params:
    recommended: list of item_id sorted by score in descending order (top-n).
    relevant: set of relevant item_id for the user.
    '''
    recom_k = recommended[:k]
    return len(set(recom_k).intersection(relevant)) / len(relevant)

def ndcg_at_k(recommended, relevant, k=10):
    '''
    considers the position of the relevant item in recommendation 
    Params:
    recommended: list of item_id sorted by score in descending order (top-n).
    relevant: set of relevant item_id for the user.
    '''
    recom_k = recommended[:k]
    dcg = 0
    for i, item in enumerate(recom_k, 1):
        if item in relevant:
            dcg += 1 / np.log2(i+1)
    n_rel = min(len(relevant), k)
    idcg = sum(1 / np.log2(i+1) for i in range(1, n_rel+1))
    return dcg/idcg if idcg > 0 else 0
