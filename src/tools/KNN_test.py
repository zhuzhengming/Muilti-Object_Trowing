import numpy as np
import faiss

import time


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        # self.y = None
        self.x = None
        self.k = k

    def fit(self, X):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.x = np.copy(X)
        self.index.add(X.astype(np.float32))
        # self.y = y

    def predict(self, X, k=None):
        # ascending order, the first one is zero.
        if k is None:
            k = self.k
        distances, indices = self.index.search(X.astype(np.float32), k=k)
        # votes = self.y[indices]

        near_x = self.x[indices, :]
        return near_x
        # predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        # return predictions


if __name__ == '__main__':
    knn = FaissKNeighbors(k=5)
    nums = 10000
    x = np.random.uniform([0, 0], [1, 1], size=(nums, 2))
    x = x.astype(np.float32)

    t0 = time.time()
    knn.fit(x)
    results = knn.predict(x[:2, :])
    print('time cost', time.time() - t0)
    print(results)
