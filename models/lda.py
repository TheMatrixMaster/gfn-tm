"""@thematrixmaster
Baseline class for training an LDA model. This class only implements the generative 
model and does not include any inference methods. The class is meant to be extended
by other classes that implement inference methods such as Gibbs sampling, variational
expectation maximization or gflownets.
"""

import numpy as np
from tqdm import tqdm
from utils.dataset import make_vocab
from collections import defaultdict

class LDA:
    K: int              # number of topics
    V: int              # number of words in vocabulary
    D: int              # number of documents
    n_wd: np.ndarray    # 2d array holding word counts for each document (BOW)
    n_dk: np.ndarray    # 2d array holding topic counts for each document
    n_kw: np.ndarray    # 2d array holding word counts for each topic
    alphas: np.ndarray  # dirichlet prior params for topic mixture per document with shape (K,)
    betas: np.ndarray   # dirichlet prior params for word mixture per topic with shape (V,)

    docs: [[str]]       # list of documents, each being a sequence of unprocessed tokens
    vocab: defaultdict  # vocabulary dict mapping words to unique index identifier
    r_vocab: [str]      # reverse vocabulary list where index of word is their identifier

    def __init__(
            self,
            K: int,
            docs: [[str]],
            vocab: defaultdict=None,
            r_vocab: [str]=None,
            alphas=None,
            beta=1e-3
    ) -> None:
        self.K = K
        self.D = len(docs)
        self.docs = docs

        if vocab is None or r_vocab is None:
            self.vocab, self.r_vocab = make_vocab(docs=docs)
        else:
            self.vocab, self.r_vocab = vocab, r_vocab

        self.V = len(self.r_vocab)
        
        self.n_wd = np.zeros((self.V, self.D), dtype=int)   # might not need this
        self.n_dk = np.zeros((self.D, self.K), dtype=int)
        self.n_kw = np.zeros((self.K, self.V), dtype=int)

        self.alphas = np.ones(K, dtype=float) if alphas is None else alphas
        self.betas = np.repeat(beta, self.V)
    
    def fit(self) -> None:
        """
        Fits the LDA model using some form of Bayesian inference
        """
        pass

    def infer(self, docs: [[str]]) -> None:
        """
        Fix global word-topic distribution, and update the topic-document mixture for 
        the new documents "docs" for a few iterations using some inference method until
        the data likelihood convergences to some maxima
        """
        pass

    def _conditional_prob(self, d, w, n_dk=None, docs=None) -> np.ndarray:
        """
        Computes the posterior topic assignment categorical likelihood of token w in document d,
        (i.e.) P(z_{dn}^i=1 | z_{(-dn)}, w)

        Parameters:
            d: index of document of interest in self.docs
            w: index of token in self.docs[d]
            n_dk: optionally override self.n_dk for inference
            docs: optionally override self.docs for inference

        Returns:
            probs: probabilities of assigning topics 1 through k to word w in document d
        """
        n_dk = self.n_dk if n_dk is None else n_dk
        docs = self.docs if docs is None else docs

        assert 0 <= d and d < n_dk.shape[0], ("Document index out of range")
        assert 0 <= w and w < len(docs[d]), ("Word index out of range")

        tok_id = self.vocab[docs[d][w]]
        beta = self.betas[tok_id]
        probs = np.empty(self.K)
        
        for i in range(self.K):
            _1 = (self.n_kw[i, tok_id] + beta) / (self.n_kw[i, :].sum() + self.V*beta)
            _2 = (n_dk[d, i] + self.alphas[i]) / (n_dk[d, :].sum() + self.K*self.alphas[i])         
            probs[i] = _1 * _2
        
        return probs / probs.sum()
    
    def loglikelihood(self, n_dk=None, docs: [[str]]=None, verbose=True) -> float:
        """
        Calculate complete log likelihood of the dataset under the current model
        """
        phi, theta = self.get_topic_mixtures(n_dk=n_dk)
        docs = self.docs if docs is None else docs
        ll = 0.0
        for d in tqdm(range(len(docs)), disable=not verbose):
            for w in range(len(docs[d])):
                tok_id = self.vocab[docs[d][w]]
                ll += np.log(np.dot(phi[:, tok_id], theta[d, :]))

        return ll
    
    def get_phi(self) -> np.ndarray:
        """
        Returns:
            phi: the expected word mixture over topics after inference
        """
        phi = np.zeros((self.K, self.V), dtype=float)
        
        for j in range(self.V):
            for i in range(self.K):
                phi[i, j] = (self.n_kw[i, j] + self.betas[i]) /\
                    (self.n_kw[i, :].sum() + self.V*self.betas[i])
        
        return phi
    
    def get_theta(self, n_dk=None) -> np.ndarray:
        """
        Returns:
            theta: the expected topic mixture over documents after inference
        """
        n_dk = self.n_dk if n_dk is None else n_dk
        theta = np.zeros((n_dk.shape[0], self.K), dtype=float)
        
        for d in range(n_dk.shape[0]):
            for i in range(self.K):
                theta[d, i] = (n_dk[d, i] + self.alphas[i]) /\
                    (n_dk[d, :].sum() + self.K*self.alphas[i])
        
        return theta

    def get_topic_mixtures(self, n_dk=None) -> (np.ndarray, np.ndarray):
        """
        Returns:
            phi: the expected word mixture over topics after inference
            theta: the expected topic mixture over documents after inference
            n_dk: optionally override self.n_dk for inference
        """
        phi = self.get_phi()
        theta = self.get_theta(n_dk=n_dk)
        
        return phi, theta
        
        