"""@thematrixmaster
Compute metrics for evaluating topic models
"""

import numpy as np


def tokenize_docs(docs: [[str]], vocab: {str: int}) -> [[int]]:
    """
    Tokenizes a list of documents into a list of lists of tokens

    Parameters:
        docs: list of documents
        vocab: vocabulary mapping words to indices

    Returns:
        list of lists of tokens
    """
    return [[vocab[word] for word in doc] for doc in docs]


def top_k_words(phi: np.ndarray, r_vocab: {int: str}, k: int=10):
    """
    Returns the top k words for each topic using phi which 
    is a matrix of size (K, V) where K is the number of topics
    and V is the vocabulary size
    """
    topk = np.argsort(phi, axis=1)[:, -k:]
    return [[(r_vocab[i], i) for i in row] for row in topk]


def top_k_docs(theta: np.ndarray, doc_ids: [int], k: int=10):
    """
    Returns the top k documents for each topic using theta which
    is a matrix of size (D, K) where D is the number of documents
    and K is the number of topics.
    """
    topk = np.argsort(theta, axis=0)[-k:, :]
    return [[(doc_ids[i], i) for i in row] for row in topk]


def nearest_neighbors(model, word):
    nearest_neighbors = model.wv.most_similar(word, topn=20)
    nearest_neighbors = [comp[0] for comp in nearest_neighbors]
    return nearest_neighbors
    

def compute_topic_diversity(phi: np.ndarray, k: int=25):
    """
    We define topic diversity to be the percentage of unique words in the top 
    k words of all topics. Diversity close to 0 indicates redundant topics.
    diversity close to 1 indicates more varied topics
    """
    topk = np.argsort(phi, axis=1)[:, -k:]
    unique = np.unique(topk)
    return len(unique) / (phi.shape[0] * k)


def normalized_mutual_information(w_i: int, w_j: int, docs: [[int]]):
    """
    Computes the normalized mutual information between two words $w_i$ and $w_j$
    drawn randomly from the same document with the following formula:

    f(w_i,w_j)=\frac{\log\frac{P(w_i,w_j)}{P(w_i)P(w_j)}}{-\log P(w_i,w_j)}

    where $P(w_i,w_j)$ is the probability of $w_i$ and $w_j$ of appearing in the 
    same document, and $P(w_i)$ and $P(w_j)$ are the probabilities of $w_i$ and $w_j$
    appearing in any document.
    """
    D = len(docs)
    N_wi = np.sum([w_i in doc for doc in docs])
    N_wj = np.sum([w_j in doc for doc in docs])
    N_wi_wj = np.sum([(w_i in doc and w_j in doc) for doc in docs])
    if N_wi_wj == 0:
        return -1
    return -1 + (np.log(N_wi) + np.log(N_wj) - 2 * np.log(D)) / (np.log(N_wi_wj) - np.log(D))


def compute_topic_coherence(phi: np.ndarray, docs: [[int]], r_vocab: {int: str}, k: int=10):
    """
    We measure topic quality according to [ETM Blei et al.] by taking the product of 
    topic coherence and topic diversity. Topic coherence is a quantitative measure of 
    the interpretability of a topic [Mimno et al., 2011]. It is the average pointwise 
    mutual information of two words drawn randomly from the same document (Lau et al., 2014)
    $$
    TC = \frac{1}{K}\sum_{k=1}^{K}\frac{1}{45}\sum_{i=1}^{10}\sum_{j=1}^{10} f(w_i^{(k)}, w_j^{(k)})
    $$
    """
    num_topics = phi.shape[0]
    topk = top_k_words(phi, r_vocab, k)
    tc = 0
    for i in range(num_topics):
        top_words = topk[i]
        for j in range(len(top_words)):
            for l in range(j+1, len(top_words)):
                tc += normalized_mutual_information(top_words[j][1], top_words[l][1], docs)

    norm = (k * (k-1)) / 2
    return tc / (num_topics * norm)
    
