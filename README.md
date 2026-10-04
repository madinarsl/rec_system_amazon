# Recommendation System for Amazon Books

> **Status:** Work in progress. Currently implemented: data preparation, EDA, baselines, and ALS (custom NumPy + `implicit` library). Upcoming: LightGBM ranker, hybrid model, and a demo service.


## Problem

E-commerce platforms rely on recommendation systems to surface relevant items from catalogs of millions. This project builds a personalized recommender for books, targeting real-world challenges:

- **Extreme sparsity:** 99.998% of the user-item matrix is empty.
- **Long-tail distribution:** top-10 items cover only ~0.9% of interactions.
- **Cold-start:** new users and books appear daily.
- **Temporal drift:** tastes and catalogs change over time.


## Dataset

**Source:** [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) (McAuley Lab, UCSD) — `Books` category.

| Property | Value |
|---|---|
| Raw reviews | 10,606,530 |
| Period | 2018-01 - 2023-09 |
| Unique users | ~692,000 |
| Unique items | ~308,000 |
| Interactions | 3,527,385 |
| Sparsity (user × item) | 99.998% |

**Preprocessing pipeline:**
1. Filtered reviews to 2018–2023.
2. Aggregated duplicate `(user_id, parent_asin)` pairs (mean rating, last timestamp).
3. Joined with book metadata (`title`, `author`, `price`, `category`, etc.).
4. Removed cold-start users (< 3 interactions) and items (< 5 interactions).
5. Applied **time-based split** to prevent data leakage:
   - **Train:** 2018-01 - 2022-02
   - **Val:** 2022-02 - 2022-10
   - **Test:** 2022-10 - 2023-09
6. Removed val/test users and items unseen during training.


## Approach

### 1. Baselines

- **Random:** recommends random items.
- **Popularity:** recommends top-K most interacted items to everyone.

### 2. Collaborative Filtering — Implicit ALS

Implemented implicit ALS from scratch in NumPy following [Hu, Koren, Volinsky (2008)](https://doi.org/10.1109/ICDM.2008.22).

**Key ideas:**
- **Preferences:** `p_ui = 1` if the user interacted with the item.
- **Confidence:** `c_ui = 1 + α · (rating − 1)`, with `α = 20`. Higher ratings get exponentially more weight.
- **Optimization:** alternating least squares with the identity `C = I + (C − I)` to exploit sparsity, reducing complexity from `O(n_items)` to `O(I_u)` per user.
- Validated against the `implicit` library


## Results on validation

| Model | Precision@10 | Recall@10 | NDCG@10 | Fit time | Predict time |
|---|---|---|---|---|---|
| Random | 0.000007 | 0.000042 | 0.000018 | — | — |
| Popularity | 0.000405 | 0.002704 | 0.001664 | — | — |
| Custom ALS (50k, k=32) | 0.004528 | 0.015496 | 0.009496 | 71.7 s | 54.4 s |
| `implicit` ALS (50k, k=32) | 0.004496 | 0.015826 | 0.009521 | 3.4 s | 11.8 s |
| **`implicit` ALS (full train, k=64)** | **0.003039** | **0.017118** | **0.009278** | 47.0 s | 83.4 s |

**Key findings:**
- Custom NumPy ALS **matches the `implicit` library within 2%** on identical data - the from-scratch implementation is correct.
- `implicit` is **~20× faster** on training - production choice.
- Full training maintains **NDCG@10 ≈ 0.0093** while scaling to all 692k users. 
- Precision drop (0.0045 - 0.0030) with full training is expected: evaluation now covers cold users with small ground truth. 


## Tech Stack (Python 3.11+)
- pandas, NumPy, SciPy - data manipulation, sparse matrices
- DuckDB - fast SQL on Parquet for aggregations
- implicit - production-grade ALS
- Matplotlib, Seaborn - visualization