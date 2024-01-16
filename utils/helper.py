"""@thematrixmaster
Helper methods for data processing
"""

import torch 
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from gensim.models.fasttext import FastText as FT_gensim

def read_mat_file(key: str, filename: str):
    """
    read the preprocess mat file whose key and and path are passed as parameters

    Args:
        key ([type]): [description]
        path ([type]): [description]

    Returns:
        [type]: [description]
    """
    term_path = Path().cwd().joinpath('data', '20ng', filename)
    doc = loadmat(term_path)[key]
    return doc


def split_train_test_matrix(dataset):
    """Split the dataset into the train set, the validation and the test set"""
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=1)
    X_test_1, X_test_2 = train_test_split(X_test, test_size=0.5, random_state=1)
    X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
    return X_train, X_val, X_test_1, X_test_2


def get_data(doc_terms_file_name="tf_idf_doc_terms_matrix", terms_filename="tf_idf_terms"):
    """read the data and return the vocabulary as well as the train, test and validation tests"""

    doc_term_matrix = read_mat_file("doc_terms_matrix", doc_terms_file_name)
    terms = read_mat_file("terms", terms_filename)
    vocab = terms
    train, validation, test_1, test_2 = split_train_test_matrix(doc_term_matrix)

    return vocab, train, validation, test_1, test_2


def get_batch(doc_terms_matrix, indices, device):
    """
    get the a sample of the given indices 

    Basically get the given indices from the dataset

    Args:
        doc_terms_matrix ([type]): the document term matrix
        indices ([type]):  numpy array 
        vocab_size ([type]): [description]

    Returns:
        [numpy arayy ]: a numpy array with the data passed as parameter
    """
    data_batch = doc_terms_matrix[indices, :]
    data_batch = torch.from_numpy(data_batch.toarray()).float().to(device)
    return data_batch


def read_embedding_matrix(
    vocab: np.ndarray,
    device: torch.device,
    load_trained: bool=True,
    model_path: str = None,
    embeddings_path: str = None,
    rho_size: int = 300,
):
    """
    read the embedding  matrix passed as parameter and return it as an vocabulary of each word 
    with the corresponding embeddings
    """
    assert model_path is not None or embeddings_path is not None,\
        "path to the embeddings or embedding model must be passed as parameter"

    if load_trained:
        embeddings_matrix = np.load(embeddings_path, allow_pickle=True)
        assert embeddings_matrix.shape[0] == len(vocab)
        assert embeddings_matrix.shape[1] == rho_size
    else:
        model_gensim = FT_gensim.load(model_path)
        assert model_gensim.wv.vector_size == rho_size

        print("Starting getting the word embeddings ++++ ")
        embeddings_matrix = np.zeros(shape=(len(vocab), rho_size))
        vocab = vocab.ravel()
        for index, word in tqdm(enumerate(vocab)):
            vector = model_gensim.wv.get_vector(word)
            embeddings_matrix[index] = vector

        print("Done getting the word embeddings ")
        with open(embeddings_path, 'wb') as file_path:
            np.save(file_path, embeddings_matrix)

    embeddings = torch.from_numpy(embeddings_matrix).to(device)
    return embeddings