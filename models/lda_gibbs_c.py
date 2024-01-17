from numpy import ndarray
import lda
import numpy as np
from models.lda import LDA
from utils.metrics import compute_topic_coherence, compute_topic_diversity

class LDA_Gibbs_C(LDA):
    n_iter: int                 # max number of iterations to run gibbs
    rng: np.random.Generator    # random number generator

    def __init__(
            self,
            K: int,
            docs: [[str]],
            rng: np.random.Generator,
            n_iter=2000,
            alpha=0.1,
            beta=1e-3,
            **kwargs,
    ) -> None:
        self.model = lda.LDA(
            n_topics=K,
            n_iter=n_iter,
            alpha=alpha,
            eta=beta,
            random_state=rng
        )
        super().__init__(K, docs=docs, alpha=alpha, beta=beta, **kwargs)

    def fit(self, X_bow=None) -> None:
        """
        Fits the LDA model using Gibbs sampling.
        """
        if X_bow is not None:
            self.model.fit(X_bow)
        else:
            self.model.fit(self.n_dw)
        
        ll = self.model.loglikelihood()
        phi = self.model.topic_word_
        
        if X_bow is not None:
            tc = compute_topic_coherence(phi, X_bow, self.r_vocab, is_bow=True)
        else:
            tc = compute_topic_coherence(phi, self.n_dw, self.r_vocab, is_bow=False)

        td = compute_topic_diversity(phi)

        return ll, tc, td
    
    def get_topic_mixtures(self) -> (np.ndarray, np.ndarray):
        return self.model.topic_word_, self.model.doc_topic_
    
    def infer(self, X_bow: ndarray) -> (np.ndarray, np.ndarray):
        """
        Fix global word-topic distribution, and update the topic-document mixture for
        the new documents "X_bow" for a few iterations using some inference method until
        the data likelihood convergences to some maxima
        """
        doc_topic = self.model.fit_transform(X_bow)
        ll = self.model.loglikelihood()
        return doc_topic, ll