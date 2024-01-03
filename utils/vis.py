"""@thematrixmaster
Methods for visualizing the results of topic modeling results
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import seaborn as sns
from typing import List, Tuple
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tueplots import bundles
from metrics import top_k_docs

# Load plotting style
# plt.rcParams.update(bundles.neurips2023())

plot_format = 'png'


def plot_topic_document_correlation(doc_ids: List[int], theta: np.ndarray, save_path=None) -> None:
    """
    Plots the topic-document correlation matrix

    Parameters:
        doc_ids: list of document ids
        theta: document-topic distribution with shape (D, K)
        save_path: path to save the plot
    """
    theta = theta[doc_ids]
    corr = np.corrcoef(theta)
    sns.heatmap(corr, cmap='rocket_r', square=True, xticklabels=False, yticklabels=False)
    plt.title('Topic-document correlation matrix')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


def heatmap_topic_samples(assign: np.ndarray, tok_doc: [int], r_vocab, num_topics:int=8, save_path=None) -> None:
    """
    Generates a heatmap of the topic sampling over iterations. 
    The x-axis is the iteration count and the y-axis is the word index.
    The color of each box represents the topic index. we are given assign which 
    is a matrix of shape (W, n_iter) where W is the number of words in the document
    and n_iter is the number of iterations. The value of assign[i, j] is the topic 
    index of the ith word in the jth iteration. Use the reverse vocabulary mapping
    to get the word from the word index, and name the y-axis with the word.

    Parameters:
        assign: topic assignment matrix for each word in a chosen document 
                over iterations with shape (W, n_iter) where W is the number
                of words in the document and n_iter is the number of iterations
        tok_doc: tokenized document with shape (W,)
        r_vocab: reverse vocabulary mapping word indices to words
        save_path: path to save the plot
    """
    W, n_iter = assign.shape

    yticklabels = [r_vocab[i] for i in tok_doc]
    topic_labels = [f'Topic {i}' for i in range(num_topics)]

    # use a discrete colormap containing num_topics colors
    cmap = sns.color_palette("deep", num_topics)

    # Create a heatmap of the topic assignment matrix
    # set figure size manually
    fig, ax = plt.subplots()
    sns.heatmap(assign, cmap=cmap, square=False, xticklabels=False, \
        yticklabels=yticklabels, ax=ax)
    plt.title('Topic assignment matrix')
    plt.xlabel('Number of sampling iterations')

    # Get the colorbar object from the Seaborn heatmap
    colorbar = ax.collections[0].colorbar
    # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
    r = colorbar.vmax - colorbar.vmin
    colorbar.set_ticks([colorbar.vmin + 0.5 * r / (num_topics) + r * i / (num_topics) for i in range(num_topics)])
    colorbar.set_ticklabels(topic_labels)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


def plot_topic_coherence(coherence: [float], save_path=None) -> None:
    """
    Plots the topic coherence of the model over iterations

    Parameters:
        coherence: list of topic coherence values
        save_path: path to save the plot
    """
    plt.plot(coherence, marker='o', markersize=3)
    plt.title('Topic coherence')
    plt.xlabel('Iteration')
    plt.ylabel('Topic coherence')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


def plot_topic_diversity(diversity: [float], save_path=None) -> None:
    """
    Plots the topic diversity of the model over iterations

    Parameters:
        diversity: list of topic diversity values
        save_path: path to save the plot
    """
    plt.plot(diversity, marker='o', markersize=3)
    plt.title('Topic diversity')
    plt.xlabel('Iteration')
    plt.ylabel('Topic diversity')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


def plot_topic_quality(coherence: [float], diversity: [float], save_path=None) -> None:
    """
    Plots the topic quality (product of coherence and diversity) of the model 
    over iterations

    Parameters:
        coherence: list of topic coherence values
        diversity: list of topic diversity values
        save_path: path to save the plot
    """
    plt.plot(np.array(coherence) * np.array(diversity), marker='o', markersize=3, color='blue')
    plt.title('Topic quality')
    plt.xlabel('Iteration')
    plt.ylabel('Topic quality')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


def plot_log_likelihood(ll: [float], save_path=None) -> None:
    """
    Plots the log likelihood of the model over iterations

    Parameters:
        ll: list of log likelihoods
        save_path: path to save the plot
    """
    plt.plot(ll, marker='o', markersize=3)
    plt.title('Log likelihood')
    plt.xlabel('Iteration')
    plt.ylabel('Log likelihood')

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


def plot_top_words_by_topic(phi: np.ndarray, r_vocab: [str], n=10, save_path=None) -> None:
    """
    Plots a heatmap of the top words for each topic

    Parameters:
        phi: topic-word distribution with shape (K, V)
        r_vocab: reverse vocabulary mapping indices to words
        n: number of top words to plot per topic
        save_path: path to save the plot
    """
    K, V = phi.shape
    cols = [f'Topic {i}' for i in range(K)]
    df = pd.DataFrame(phi.T, index=r_vocab, columns=cols)

    rows = []
    for col in df.columns:
        df = df.sort_values(by=col, ascending=False)
        rows.append(df.head(n))

    topk = pd.concat(rows, axis=0)

    fig, ax = plt.subplots(figsize=(5, 12))
    sns.heatmap(topk, cmap='rocket_r', square=False, yticklabels=True, 
                ax=ax, linewidths=.5, cbar_kws=dict(shrink=0.2, label="topic prob"))
    
    ax.set_title(f'Top {n} words per topic')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


def plot_top_docs(theta: np.ndarray, n=100, save_path=None) -> None:
    """
    Plots a heatmap of the top documents for each topic

    Parameters:
        theta: document-topic distribution with shape (D, K)
        n: number of top documents to plot per topic
        save_path: path to save the plot
    """
    cols = [f'Topic {i}' for i in range(theta.shape[1])]
    idx = []

    idx = np.array(idx)
    df = pd.DataFrame(theta, columns=cols)

    rows = []
    for col in df.columns:
        df = df.sort_values(by=col, ascending=False)
        rows.append(df.head(n))

    topk = pd.concat(rows, axis=0)

    fig, ax = plt.subplots()
    sns.heatmap(topk, cmap='rocket_r', cbar=True, yticklabels=False, 
                ax=ax, cbar_kws=dict(shrink=0.2, label="topic prob"))

    ax.set_title(f'Top {n} docs per topic')
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, format=plot_format)
    else:
        plt.show()

    plt.clf()


if __name__ == "__main__":
    folder = "results/eICU_LDA_gfn_34_2023-12-31_02-58-26"

    ll = np.load(f"{folder}/ll.npy")
    tc = np.load(f"{folder}/tc.npy")
    td = np.load(f"{folder}/td.npy")
    phi = pd.read_csv(f"{folder}/phi.csv", index_col=0)
    theta = pd.read_csv(f"{folder}/theta.csv", index_col=0)

    # samples = np.load(f"{folder}/samples.npy")
    # samples_ids = np.load(f"{folder}/samples_ids.npy")
    # with open(f"{folder}/docs.pkl", "rb") as f:
    #     tok_docs = pickle.load(f)

    num_topics = theta.shape[1]
    r_vocab = list(phi.columns)
    vocab = {w: i for i, w in enumerate(r_vocab)}

    phi = phi.to_numpy()
    theta = theta.to_numpy()

    plot_log_likelihood(ll, save_path=f"{folder}/train_ll.{plot_format}")
    plot_topic_quality(tc, td, save_path=f"{folder}/train_topic_quality.{plot_format}")
    plot_topic_coherence(tc, save_path=f"{folder}/train_topic_coherence.{plot_format}")
    plot_topic_diversity(td, save_path=f"{folder}/train_topic_diversity.{plot_format}")
    plot_top_words_by_topic(phi, r_vocab, save_path=f"{folder}/train_top_words.{plot_format}")

    # for topic, (sample, doc_id, tok_doc) in enumerate(zip(samples, samples_ids, tok_docs)):
    #     sample = sample[:len(tok_doc), :]
    #     assert np.all(sample != -1), "Sample contains -1 values"
    #     heatmap_topic_samples(sample, tok_doc, r_vocab, num_topics, save_path=f"{folder}/train_topic_{topic}_samples_doc_{doc_id}.pdf")

    # test_ll = np.load(f"{folder}/test_ll.npy")
    # test_ppl = np.loadtxt(f"{folder}/test_ppl.txt")
    # test_theta = pd.read_csv(f"{folder}/test_theta.csv", index_col=0)

    # test_samples = np.load(f"{folder}/test_samples.npy")
    # test_samples_ids = np.load(f"{folder}/test_samples_ids.npy")
    # with open(f"{folder}/test_docs.pkl", "rb") as f:
    #     test_tok_docs = pickle.load(f)

    # test_theta = test_theta.to_numpy()
    # num_topics = test_theta.shape[1]

    # print(f"Test perplexity: {test_ppl}")
    
    # plot_log_likelihood(test_ll, save_path=f"{folder}/test_ll.pdf")

    # for topic, (sample, doc_id, tok_doc) in enumerate(zip(test_samples, test_samples_ids, test_tok_docs)):
    #     sample = sample[:len(tok_doc), :]
    #     sample = sample[sample != -1]
    #     assert np.all(sample != -1), "Sample contains -1 values"
    #     try:
    #         heatmap_topic_samples(sample, tok_doc, r_vocab, num_topics, save_path=f"{folder}/test_topic_{topic}_samples_doc_{doc_id}.pdf")
    #     except:
    #         print(f"Error plotting topic {topic} samples for document {doc_id}")