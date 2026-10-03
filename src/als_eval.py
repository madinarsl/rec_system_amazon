import numpy as np
import time

def evaluate_als(model, user_idx_local, gt_padded, gt_sizes, K=10, batch_size=5000):
    """evaluates fitted ALS model using batches"""
    n_val = len(user_idx_local)
    discounts = 1.0 / np.log2(np.arange(1, K + 1) + 1)
    
    all_hits_per_pos, all_hits, all_sizes = [], [], []
    t0 = time.time()
    for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        batch_users = user_idx_local[start:end]
        
        scores = model.predict_scores(user_indices=batch_users)  # (batch, n_items)
        
        top_k = np.argpartition(scores, -K, axis=1)[:, -K:]  # not full sorting but only for k elements

        gt_batch = gt_padded[start:end]
        sizes_batch = gt_sizes[start:end]
        
        matches = (top_k[:, :, None] == gt_batch[:, None, :])
        hits_per_pos = matches.any(axis=2)
        hits = hits_per_pos.sum(axis=1).astype(np.float64)
        
        all_hits_per_pos.append(hits_per_pos)
        all_hits.append(hits)
        all_sizes.append(sizes_batch)

    pr_time = time.time() - t0
    hits_per_pos = np.concatenate(all_hits_per_pos)
    hits = np.concatenate(all_hits)
    sizes = np.concatenate(all_sizes)
    
    dcg = (hits_per_pos * discounts).sum(axis=1)
    idcg = np.zeros(len(sizes))
    for k in range(1, min(gt_padded.shape[1], K) + 1):
        idcg += (sizes >= k) * discounts[k - 1]
    
    return {
        "precision@10": (hits / K).mean(),
        "recall@10": (hits / np.maximum(sizes, 1)).mean(),
        "ndcg@10": (dcg / np.maximum(idcg, 1e-9)).mean(),
        "predict time (s)": pr_time
    }

def evaluate_implicit(model, user_idx_local, R, gt_padded, gt_sizes, K=10):
    """
    evaluatates implicit ALS via model.recommend()
    """
    n_val = len(user_idx_local)
    discounts = 1.0 / np.log2(np.arange(1, K + 1) + 1)
    t0 = time.time()
    ids, scores = model.recommend(
        userid=user_idx_local,
        user_items=R[user_idx_local],
        N=K,
        filter_already_liked_items=False,
        recalculate_user=False,
    )
    matches = (ids[:, :, None] == gt_padded[:, None, :])
    hits_per_pos = matches.any(axis=2)         
    hits = hits_per_pos.sum(axis=1).astype(np.float64)  
    
    pr_time = time.time() - t0

    dcg = (hits_per_pos * discounts).sum(axis=1)
    idcg = np.zeros(n_val)
    for k in range(1, min(gt_padded.shape[1], K) + 1):
        idcg += (gt_sizes >= k) * discounts[k - 1]
    
    return {
        "precision@10": (hits / K).mean(),
        "recall@10": (hits / np.maximum(gt_sizes, 1)).mean(),
        "ndcg@10": (dcg / np.maximum(idcg, 1e-9)).mean(),
        'predict time (s)': pr_time
    }
