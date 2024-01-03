"""@matrixmaster
Loads dataset and preprocesses it
"""

import os
from typing import Optional
from pandas.core.api import DataFrame as DataFrame
import torch
import torch.distributions as dist
import pandas as pd
import numpy as np
from collections import defaultdict


def make_vocab(docs: [[str]]) -> (defaultdict, [str]):
    """
    Takes in a list of documents and creates a vocabulary dictionary
    mapping raw tokens to their unique integer identifier

    Parameters:
        docs: a list of documents, each a sequence of raw tokens

    Returns:
        vocab: a dictionary mapping raw tokens to a unique integer identifier
        r_vocab: a list that positions the unique word at their unique id/index
    """
    vocab = defaultdict(int)
    r_vocab = []
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = len(vocab)
                r_vocab.append(word)
    return vocab, r_vocab


class Dataset:
    docs: [[str]]                           # list of documents, each being a sequence of unprocessed tokens
    doc_ids: [int]                          # list of document ids
    doc_targets: Optional[pd.DataFrame]     # optional list of document targets for supervised learning
    vocab: dict                             # vocabulary dict mapping words to unique index identifier
    r_vocab: [str]                          # reverse vocabulary list where index of word is their identifier
    D: int                                  # number of documents
    V: int                                  # number of words in vocabulary
    rng: np.random.Generator                # random number generator

    def __init__(self, path: str, target_paths: [str], sorted=True, rng=None) -> None:
        self.rng = rng if rng else np.random.default_rng()
        self.docs, self.doc_ids = self.load_documents(path, sorted=sorted)
        self.vocab, self.r_vocab = self.make_vocab(docs=self.docs)
        self.D = len(self.docs)
        self.V = len(self.r_vocab)
        self.doc_targets = self.load_targets(paths=target_paths) if target_paths else None

    def load_documents(self, path: str, sorted: bool) -> ([[str]], [int]):
        """
        Loads the documents and their ids

        Parameters:
            path: path to the dataset
        
        Returns:
            docs: list of documents, each being a sequence of unprocessed tokens
            doc_ids: list of document ids
        """
        pass

    def make_vocab(self, docs: [[str]]) -> (defaultdict, [str]):
        """
        Takes in a list of documents and creates a vocabulary dictionary
        mapping raw tokens to their unique integer identifier

        Parameters:
            docs: a list of documents, each a sequence of raw tokens

        Returns:
            vocab: a dictionary mapping raw tokens to a unique integer identifier
            r_vocab: a list that positions the unique word at their unique id/index
        """
        return make_vocab(docs=docs)

    def load_targets(self, paths: [str]) -> pd.DataFrame:
        """
        Loads the targets for each document

        Parameters:
            paths: paths to the target datasets
        
        Returns:
            targets: pandas dataframe indexed by document id with targets as columns
        """
        pass

    def __len__(self) -> int:
        return self.D
    
    def __getitem__(self, idx: int) -> ([str], int):
        return self.docs[idx], self.doc_ids[idx]
    
    def get_targets(self, doc_id: int) -> Optional[pd.Series]:
        """
        Returns the targets for a given document id

        Parameters:
            doc_id: the document id
        
        Returns:
            targets: the targets for the document id
        """
        assert self.doc_targets is not None, "Dataset does not have targets"
        return self.doc_targets.loc[doc_id]
    
    def train_test_split(self, full_size=1.0, test_size: float=0.2, shuffle=True):
        """
        Splits the dataset into train and test sets. ids and docs must be 
        shuffled together with the same permutation to maintain order

        Parameters:
            full_size: what percentage of the dataset to use
            test_size: the size of the test set as a fraction of the dataset size
            shuffle: whether to shuffle the dataset before splitting
        
        Returns:
            train_ids: list of document ids for the training set
            test_ids: list of document ids for the test set
            train_docs: list of documents for the training set
            test_docs: list of documents for the test set
        """
        assert 0.0 <= test_size <= 1.0, "test_size must be between 0 and 1"
        assert 0.0 <= full_size <= 1.0, "full_size must be between 0 and 1"

        if shuffle:
            perm = self.rng.permutation(self.D)
            self.docs = [self.docs[i] for i in perm]
            self.doc_ids = [self.doc_ids[i] for i in perm]
        
        if full_size != 1.0:
            print(f"Using {full_size*100}% of the dataset")
            first_split = int(self.D * full_size)
            doc_ids = self.doc_ids[:first_split]
            docs = self.docs[:first_split]

            self.vocab, self.r_vocab = make_vocab(docs=self.docs)
            self.D = len(self.docs)
            self.V = len(self.r_vocab)
        else:
            first_split = self.D
            doc_ids = self.doc_ids
            docs = self.docs

        second_split = int(first_split * test_size)
        train_ids = doc_ids[second_split:]
        test_ids = doc_ids[:second_split]
        train_docs = docs[second_split:]
        test_docs = docs[:second_split]

        print(f"Train size: {len(train_ids)}")
        print(f"Test size: {len(test_ids)}")

        assert len(train_ids) == len(train_docs), "Train ids and docs must be the same length"
        assert len(test_ids) == len(test_docs), "Test ids and docs must be the same length"

        return train_ids, test_ids, train_docs, test_docs
        
    
    @staticmethod
    def split_documents(docs: [[str]], test_size: float) -> ([[str]], [[str]]):
        """
        Split each document's words into set subsets with the second subset 
        having test_size percent of the words

        Parameters:
            docs: list of documents, each being a sequence of unprocessed tokens
            test_size: the size of the second subset as a fraction of the document size

        Returns:
            train_docs: list of documents for the training set
            test_docs: list of documents for the test set
        """
        train_docs = []
        test_docs = []

        for doc in docs:
            split = int(len(doc) * test_size)
            train_docs.append(doc[split:])
            test_docs.append(doc[:split])

        return train_docs, test_docs
    
    def __iter__(self):
        for i in range(self.D):
            yield self.docs[i], self.doc_ids[i]

    def __str__(self) -> str:
        """
        Print the summary statistics of the dataset
        """
        return f"Dataset: {self.D} documents, {self.V} unique words"
    

class eICUDataset(Dataset):
    def __init__(self, path: str, target_paths: [str], **kwargs) -> None:
        super().__init__(path=path, target_paths=target_paths, **kwargs)

    def load_documents(self, path: str, sorted=True) -> ([[str]], [int]):
        df = pd.read_csv(path, index_col=0, header=0).dropna()

        if sorted:
            df.sort_values(by="drugstartoffset", inplace=True)
        else:
            df = df.sample(frac=1, random_state=self.rng)
        
        df = df.groupby("patientunitstayid").agg({"drugname": list})

        doc_ids = df.index.tolist()
        docs = df["drugname"].tolist()

        return docs, doc_ids

    def load_targets(self, paths: [str]) -> pd.DataFrame:
        """
        Loads the targets for each document

        Parameters:
            paths: paths to the target datasets
        
        Returns:
            targets: pandas dataframe indexed by document id with targets as columns
        """
        targets = pd.DataFrame(index=self.doc_ids)

        for path in paths:
            tmp = pd.read_csv(path, index_col=0, header=0).dropna()
            tmp = tmp.sort_values(by=tmp.columns[-1])
            tmp = tmp[~tmp.index.duplicated(keep='first')]
            targets = targets.join(tmp, how="left", lsuffix="_left", rsuffix="_right")

        return targets
    

class mimicDataset(Dataset):
    def __init__(self, path: str, target_paths: [str], **kwargs) -> None:
        super().__init__(path, target_paths, **kwargs)

    def load_documents(self, path: str, sorted=True) -> ([[str]], [int]):
        df = pd.read_csv(path, index_col=0, header=0).dropna()

        if sorted:
            df.sort_values(by="SUBJECT_ID", inplace=True)
        else:
            df = df.sample(frac=1, random_state=self.rng)
        
        df = df.groupby('SUBJECT_ID')['ICD9_CODE']\
            .apply(list).reset_index(name='ICD9_CODE')

        doc_ids = df.index.tolist()
        docs = df["ICD9_CODE"].tolist()

        return docs, doc_ids
    
    def load_targets(self, paths: [str]) -> DataFrame:
         # Keep codes with only numbers
        codes = pd.read_csv(paths[0], index_col=0, header=0).dropna()
        mask = codes['ICD9_CODE'].str.contains('[a-zA-Z]')
        codes = codes[~mask]
        codes['ICD9_CODE'] = codes['ICD9_CODE'].astype(int)
        codes.set_index('ICD9_CODE', inplace=True)
        codes = codes[codes.index.isin(self.vocab.keys())]
        codes.sort_index(inplace=True)
        codes = codes[~codes.index.duplicated(keep='first')]
        return codes['ICD9_CODE'].to_dict()


class syntheticDataset(Dataset):
    def __init__(
        self,
        K: int = 3,
        V: int = 100,
        D: int = 256,
        doc_length: int = 16,
        alpha: float = 0.1,
        beta: float = 0.1,
        **kwargs
    ) -> None:
        self.K = K
        self.D = D
        self.V = V
        self.alpha = alpha
        self.beta = beta
        self.doc_length = doc_length
        self.theta_prior = dist.Dirichlet(torch.full((K,), alpha))
        self.phi_prior = dist.Dirichlet(torch.full((V,), beta))
        super().__init__(path=None, target_paths=None, **kwargs)

    def load_documents(self, path: str, sorted=True) -> ([[str]], [int]):
        topic_word = self.phi_prior.sample((self.K,))
        doc_topic = self.theta_prior.sample((self.D,))
        self.theta = doc_topic.log().clone()
        self.phi = topic_word.log().clone()
        dists = torch.matmul(doc_topic, topic_word)

        self.docs = []
        self.doc_ids = []

        for i in range(self.D):
            doc = []
            for j in range(self.doc_length):
                word = dist.Categorical(dists[i]).sample()
                doc.append(str(word.item()))
            self.docs.append(doc)
            self.doc_ids.append(i)

        return self.docs, self.doc_ids

    def make_vocab(self, docs: [[str]]) -> (defaultdict, [str]):
        vocab = [str(i) for i in range(self.V)]
        r_vocab = vocab
        return dict(zip(vocab, range(self.V))), r_vocab

    def load_targets(self, paths: [str]) -> pd.DataFrame:
        pass