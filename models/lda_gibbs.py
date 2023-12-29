"""@thematrixmaster
Implementation of LDA with Gibbs sampling inference
"""

import numpy as np
from models.lda import LDA
from tqdm import tqdm
from utils.metrics import compute_topic_diversity, compute_topic_coherence, tokenize_docs

class LDA_Gibbs(LDA):
    n_iter: int                 # max number of iterations to run gibbs
    assign: np.ndarray          # topic assignment matrix over each word for each document (Z_dw)
    rng: np.random.Generator    # random number generator
    patience: int               # number of iterations to wait for convergence
    eval_every: int             # evaluate model every eval_every iterations
    save_samples: bool          # whether to save the samples from the sampler

    def __init__(
            self,
            K: int,
            docs: [[str]],
            rng: np.random.Generator,
            n_iter=10,
            patience=None,
            save_samples=False,
            eval_every=1,
            **kwargs,
    ) -> None:
        super().__init__(K, docs, **kwargs)
        
        self.rng = rng
        self.patience = patience
        self.n_iter = n_iter
        self.eval_every = eval_every
        self.save_samples = save_samples
        max_doc_length = max([len(doc) for doc in docs])

        if save_samples:
            self.assign = np.empty((self.D, max_doc_length, n_iter+1), dtype=int)
        else:
            self.assign = np.empty((self.D, max_doc_length, 2), dtype=int)

        self.assign[:] = -1
        self._init_gibbs()

    def _init_gibbs(self) -> None:
        """
        Initializes the t=0 state for Gibbs sampling and updates the initial
        word-topic and document-topic assignment.
        """
        
        for d in range(self.D):
            for w in range(len(self.docs[d])):
                # randomly assign a topic to each word w per document d
                tok_id = self.vocab[self.docs[d][w]]
                self.assign[d, w, 0] = self.rng.integers(self.K)

                # increment counters for word-topic and topic-document mixtures
                i = self.assign[d, w, 0]
                self.n_kw[i, tok_id] += 1
                self.n_dk[d, i] += 1

    def fit(self, verbose=True, eval=True) -> ([float], [float], [float]):
        """
        Run collapsed Gibbs sampling

        Parameters:
            verbose: whether to print out status updates
            eval: whether to evaluate log likelihood during sampling
        
        Returns:
            ll: list of log likelihoods for each iteration
        """
        # initialize required variables
        self._init_gibbs()
        patience = self.patience
        hi = -np.inf
        ll = []
        tc = []
        td = []
        
        if verbose:
            print("\n", "="*10, "START SAMPLER", "="*10)
        
        # run the sampler
        for t in range(self.n_iter):
            for d in tqdm(range(self.D)):
                for w in range(len(self.docs[d])):
                    tok_id = self.vocab[self.docs[d][w]]
                    
                    # decrement counter of previous assignment
                    if self.save_samples:
                        i_t = self.assign[d, w, t]
                    else:
                        i_t = self.assign[d, w, t%2]

                    self.n_kw[i_t, tok_id] -= 1
                    self.n_dk[d, i_t] -= 1

                    # assign new topics
                    probs = self._conditional_prob(d, w)
                    i_tp1 = np.argmax(self.rng.multinomial(1, probs))

                    # re-increment previous topic counter
                    self.n_kw[i_t, tok_id] += 1
                    self.n_dk[d, i_t] += 1

                    # increment counter with new assignment
                    self.n_kw[i_tp1, tok_id] += 1
                    self.n_dk[d, i_tp1] += 1

                    if self.save_samples:
                        self.assign[d, w, t+1] = i_tp1
                    else:
                        self.assign[d, w, (t+1)%2] = i_tp1

            if verbose:
                print(f"Sampled {t+1}/{self.n_iter}")

            # evaluate log likelihood with current parameters
            if eval & ((t+1) % self.eval_every == 0):
                print("Evaluating the model...")
                ll.append(self.loglikelihood())
                phi = self.get_phi()
                tc.append(compute_topic_coherence(phi, self.tok_docs, self.r_vocab, k=10))
                td.append(compute_topic_diversity(phi, k=25))

                print(f"Log likelihood: {ll[-1]}, Topic quality: {tc[-1]*td[-1]}")

                if self.patience is None:
                    continue

                # check for convergence with patience
                hi = max(hi, ll[-1])

                if len(ll) < self.patience:
                    continue
                
                if ll[-1] < hi:
                    patience -= 1
                else:
                    patience = self.patience

                if patience == 0:
                    break
        
        return ll, tc, td

    def infer(self, docs: [[str]], patience=None, n_iter=None):
        """
        Fix global word-topic distribution, and update the topic-document mixture for
        the new documents "docs" for a few iterations using some inference method until
        the data likelihood convergences to some maxima
        """
        D = len(docs)
        n_dk = np.zeros((D, self.K), dtype=int)
        n_iter = self.n_iter if n_iter is None else n_iter
        max_doc_length = max([len(doc) for doc in docs])

        if self.save_samples:
            assign = np.empty((D, max_doc_length, self.n_iter+1), dtype=int)
        else:
            assign = np.empty((D, max_doc_length, 2), dtype=int)
        
        assign[:] = -1

        loc_patience = self.patience if patience is None else patience
        hi = -np.inf
        ll = []

        # initialize required variables
        for d in range(D):
            for w in range(len(docs[d])):
                # randomly assign a topic to each word w per document d
                assign[d, w, 0] = self.rng.integers(self.K)

                # increment counters for topic-document mixture only
                i = assign[d, w, 0]
                n_dk[d, i] += 1

        for t in tqdm(range(self.n_iter)):
            for d in range(D):
                for w in range(len(docs[d])):             
                    # decrement previous topic counter previous assignment
                    if self.save_samples:
                        i_t = assign[d, w, t]
                    else:
                        i_t = assign[d, w, t%2]

                    self.n_dk[d, i_t] -= 1
                    
                    # assign new topics using fixed word-topic distribution
                    probs = self._conditional_prob(d, w, n_dk=n_dk, docs=docs)
                    i_tp1 = np.argmax(self.rng.multinomial(1, probs))

                    # re-increment previous topic counter
                    self.n_dk[d, i_t] += 1

                    # increment counter with new assignment
                    n_dk[d, i_tp1] += 1

                    if self.save_samples:
                        assign[d, w, t+1] = i_tp1
                    else:
                        assign[d, w, (t+1)%2] = i_tp1

            # evaluate log likelihood with current parameters
            ll.append(self.loglikelihood(n_dk=n_dk, docs=docs, verbose=False))
            print(f"Log likelihood: {ll[-1]}")

            if patience is None:
                continue

            # check for convergence with patience
            hi = max(hi, ll[-1])

            if len(ll) < patience:
                continue

            if ll[-1] < hi:
                loc_patience -= 1
            
            if loc_patience == 0:
                break
        
        _, theta = self.get_topic_mixtures(n_dk=n_dk)

        return ll, theta, n_dk, assign
        
        