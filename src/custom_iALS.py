import numpy as np
from numba import njit, prange

class ImplicitALS:
    def __init__(self, n_factors=64, reg=0.1, alpha=20.0, n_iter=15, random_state=42):
        self.n_factors = n_factors  # size of latent vector
        self.reg = reg
        self.alpha = alpha  # coef in confidence formula, controls variance of weights
        self.n_iter = n_iter
        self.random_state = random_state
        self.X = None  # user factors
        self.Y = None  # item factors

    def fit(self, R, C, n_users, n_items):
        """
        R: sparse (n_users, n_items) - preferences
        C: sparse (n_users, n_items) - confidence
        """
        #initialising x and y as random small numbers from normal distribution
        rng = np.random.default_rng(self.random_state)
        k = self.n_factors
        self.X = rng.normal(0, 0.01, size=(n_users, k)).astype(np.float32)
        self.Y = rng.normal(0, 0.01, size=(n_items, k)).astype(np.float32)

        YtY = self.Y.T @ self.Y  # (k,k) contribution of all items
        reg_I = self.reg * np.eye(k, dtype=np.float32)  #regularization

        for it in range(self.n_iter):
            # updating X (users) with fixed Y 
            self.X = self._update_factors(R=R.tocsr(), C=C.tocsr(), Y=self.Y, YtY=YtY, reg_I=reg_I, n_entities=n_users)
            XtX = self.X.T @ self.X  # updating XtX after updating X
            
            # updating Y (items) with fixed X
            self.Y = self._update_factors(R=R.T.tocsr(), C=C.T.tocsr(),Y=self.X, YtY=XtX, reg_I=reg_I, n_entities=n_items)
            YtY = self.Y.T @ self.Y # updating YtY for next iteration

        return self
    
    def _update_factors(self, R, C, Y, YtY, reg_I, n_entities):
        """updating X or Y"""
        k = self.n_factors
        result = np.zeros((n_entities, k), dtype=np.float32)   # empty interaction matrix

        for u in range(n_entities):
            start, end = R.indptr[u], R.indptr[u + 1]  # indices for all non-zero elements
            if start == end:  # no interactions
                continue
            #for user:
            item_indices = R.indices[start:end]
            conf = C.data[start:end]  # confidence values for user-item
            pref = R.data[start:end]  # preferences (1)
            
            Y_u = Y[item_indices].T # (k, |interactions|) only rows form Y with user's items
            Y_u_scaled = Y_u * (conf - 1.0)  # (k, |interactions|) each column multiples by its confidence-1
            A = YtY + reg_I + Y_u_scaled @ Y_u.T
            b = Y_u_scaled @ pref + Y_u @ pref  
            result[u] = np.linalg.solve(A, b)
        
        return result

    def predict_scores(self, user_indices=None, batch_size=None):
        '''Params:
        user_indices: predicts only for given users (None - for all users),
        batch_size: processes predictions for users in batches and concatenates results (None - for all users at once)
        '''
        if user_indices is None: # all users from fitted
            n_query = self.X.shape[0]
        else:
            user_indices = np.asarray(user_indices)
            n_query = len(user_indices)

        n_items = self.Y.shape[0]

        # no batches or matrix isnt that heavy (<2 GB)
        if batch_size is None or n_query*n_items < 5e8:  
            if user_indices is None:
                return self.X @ self.Y.T
            return self.X[user_indices] @ self.Y.T

        # with batches
        scores = np.empty((n_query, n_items), dtype=np.float32)
        for start in range(0, n_query, batch_size):
            end = min(start + batch_size, n_query)
            if user_indices is None:
                X_batch = self.X[start:end]
            else:
                X_batch = self.X[user_indices[start:end]]
            scores[start:end] = X_batch @ self.Y.T
        return scores

