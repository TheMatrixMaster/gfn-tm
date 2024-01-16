"""@thematrixmaster
Baseline class for training an ETM model with a Logistic Normal prior over \theta.
"""

import torch
import pickle
import numpy as np
from scipy.io import loadmat
import torch.optim as optim
from torchtyping import TensorType as TT
import torch.nn.functional as F 
from torch import nn
from tqdm import tqdm
from collections import defaultdict
from gensim.models.fasttext import FastText as FT_gensim

from utils.metrics import compute_topic_coherence, compute_topic_diversity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ETM(nn.Module):
    def __init__(
        self,
        num_topics: int,                    # number of topics
        vocab_size: int,                    # size of vocabulary
        t_hidden_size: int,                 # size of hidden layer in encoder
        rho_size: int,                      # num of dimensions in word embedding
        theta_act: str,                     # activation function for theta encoder
        embeddings: TT["V", "L"] = None,    # pre-trained word embeddings
        train_embeddings: bool = True,      # whether to train word embeddings from scratch
        enc_drop=0.5,                       # dropout rate for encoder
        clip=0.0,                           # gradient clipping
        batch_size: int = 1000,             # batch size of documents
        vocab: defaultdict(int) = None,     # vocabulary dictionary mapping words to indices
        r_vocab: [str] = None,              # reverse vocabulary list where index of word is their identifier
    ):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.clip = clip
        self.batch_size = batch_size
        self.vocab = vocab
        self.r_vocab = r_vocab

        assert theta_act in ['tanh', 'relu', 'softplus', 'rrelu', 'leakyrelu', 'elu', 'selu', 'glu'], \
            'theta_act should be in "tanh, relu, softplus, rrelu, leakyrelu, elu, selu, glu"'
        
        self.theta_act = self.get_activation(theta_act)

        ## define the V x L word embedding matrix \rho
        self.train_embeddings = train_embeddings
        if self.train_embeddings:
            self.rho = nn.Parameter(torch.randn(vocab_size, rho_size))
        else:
            self.rho = embeddings.clone().float().to(device)

        ## define the K x L topic embedding matrix \alpha
        self.alphas = nn.Linear(rho_size, num_topics, bias=False)
    
        ## define the encoder that outputs variational parameters for \theta conditioned on x
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size, t_hidden_size), 
            self.theta_act,
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act: str):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    # theta ~ mu + std N(0,1)
    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows: TT["B", "V"]) -> (TT["B", "K"], TT["B"], float):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)                 # bsz x K
        logsigma_theta = self.logsigma_q_theta(q_theta)     # bsz x K

        # KL[q(theta) || p(theta)] = lnq(theta) - lnp(theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self) -> TT["K", "V"]:
        ## softmax over vocab dimension
        beta = F.softmax(self.alphas(self.rho), dim=0).transpose(1, 0)
        return beta

    def get_theta(self, normalized_bows: TT["B", "V"]) -> (TT["B", "K"], float):
        ## gets topic proportions for a batch of documents
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta

    def decode(self, theta: TT["B", "K"], beta: TT["K", "V"]) -> TT["B", "V"]:
        ## returns the reconstructed distribution of words for a batch of documents
        res = torch.mm(theta, beta)
        preds = torch.log(res+1e-6)
        return preds

    def forward(
        self,
        X: TT["B", "V"],
        X_normalized: TT["B", "V"],
        theta=None,
        aggregate: bool=True
    ):
        ## get \theta
        if theta is None:
            theta, kld_theta = self.get_theta(X_normalized)
        else:
            kld_theta = None

        ## get \beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(theta, beta)
        recon_loss = -(preds * X).sum(dim=1)
        if aggregate:
            recon_loss = recon_loss.mean()
        return recon_loss, kld_theta
    
    def get_optimizer(self, optimizer: str, lr: float, wdecay: float):
        """
        Get the model default optimizer 

        Args:
            sefl ([type]): [description]
        """
        if optimizer == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wdecay)
        elif optimizer == 'adagrad':
            optimizer = optim.Adagrad(self.parameters(), lr=lr, weight_decay=wdecay)
        elif optimizer == 'adadelta':
            optimizer = optim.Adadelta(self.parameters(), lr=lr, weight_decay=wdecay)
        elif optimizer == 'rmsprop':
            optimizer = optim.RMSprop(self.parameters(), lr=lr, weight_decay=wdecay)
        elif optimizer == 'asgd':
            optimizer = optim.ASGD(self.parameters(), lr=lr, t0=0, lambd=0., weight_decay=wdecay)
        else:
            print('Defaulting to vanilla SGD')
            optimizer = optim.SGD(self.parameters(), lr=lr)
        self.optimizer = optimizer
        return optimizer
    
    def train_one_batch(self, batch: (TT["B", "V"], TT["B", "V"])):
        self.train()
        self.optimizer.zero_grad()
        self.zero_grad()
        X, X_normalized = batch
        recon_loss, kld_theta = self.forward(X, X_normalized, aggregate=True)
        total_loss = recon_loss + kld_theta
        total_loss.backward()
        
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip)
        
        self.optimizer.step()
        return recon_loss.item(), kld_theta.item()
    
    def fit(self, X: TT["D", "V"], X_normalized: TT["D", "V"], num_epochs: int):
        nelbo, tc, td = [], [], []

        num_docs = X.shape[0]
        assert num_docs > self.batch_size, "num_docs should be greater than batch_size"

        for epoch in range(num_epochs):
            print("Epoch: {}".format(epoch))

            ## get a new permutation of the training data
            batch_indices = torch.randperm(num_docs).split(self.batch_size)

            for idx, batch_idx in tqdm(enumerate(batch_indices)):
                batch = (X[batch_idx, :], X_normalized[batch_idx, :])
                recon_loss, kld_theta = self.train_one_batch(batch)

                nelbo.append(recon_loss + kld_theta)

                # TODO: consider bow representation
                beta = self.get_beta().detach().cpu().numpy()
                tc.append(compute_topic_coherence(beta, X, self.r_vocab))
                td.append(compute_topic_diversity(beta))

            print(f"""
                Epoch: {epoch+1}/{num_epochs},
                negative ELBO: {nelbo[-1]},
                Topic quality: {tc[-1]*td[-1]},
                Topic coherence: {tc[-1]},
                Topic diversity: {td[-1]}
            """)

        return nelbo, tc, td
    
    def infer(self, X_1: TT["D", "V"], X_2: TT["D", "V"], batch_size: int=200, bow_norm: bool=True):
        """
        Compute perplexity on document completion by fitting on X_1, then predicting on X_2.
        """
        self.eval()

        with torch.no_grad():
            ## get \beta here
            beta = self.get_beta()
            
            acc_loss, cnt = 0, 0
            indices = torch.split(torch.tensor(range(X_1.shape[0])), batch_size)

            for _, batch_idx in enumerate(indices):
                batch_1 = X_1[batch_idx, :].to(device)
                if bow_norm:
                    normalized_data_batch_1 = batch_1 / batch_1.sum(1).unsqueeze(1)
                else:
                    normalized_data_batch_1 = batch_1

                theta, _ = self.get_theta(normalized_data_batch_1)

                ## get predition loss on second half of data
                batch_2 = X_2[batch_idx, :].to(device)
                res = torch.mm(theta, beta)
                preds = torch.log(res)
                recon_loss = -(preds * batch_2).sum(1)
                loss = recon_loss / batch_2.sum(1).unsqueeze(1).squeeze()
                loss = np.nanmean(loss.numpy())
                acc_loss += loss
                cnt += 1

            cur_loss = acc_loss / cnt
            ppl_dc = round(np.exp(cur_loss), 1)

            beta = beta.data.cpu().numpy()
            tc = compute_topic_coherence(beta, X_1+X_2, self.r_vocab)
            td = compute_topic_diversity(beta, 25)
            return ppl_dc, tc, td
            
