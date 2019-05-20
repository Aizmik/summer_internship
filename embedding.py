import numpy as np
from utilities import initalize


class Embedding(object):
    def __init__(self, vocab_size: int, embed_size: int,
                 init_range: float = 1.0):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.W = initalize((vocab_size, embed_size), init_range)
        self.dW = np.zeros(self.W.shape)

        self.params = [
            ('W', self.W, self.dW)
        ]

    def init_sequence(self):
        self.t = 0
        self.x = {}
        self.dW[:] = 0

    def forward(self, x: np.int32):
        self.x[self.t] = x
        self.t += 1

        return self.W[int(x)]

    def backward(self, delta: np.float64):
        self.t -= 1
        x = self.x[self.t]
        self.dW[x] += delta
