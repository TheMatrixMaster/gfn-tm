"""@thematrixmaster
Implementation of latent-dirichlet allocation with GflowNet-EM inference
The overall algorithm is as follows:

E-step:
    - Train a continuous GflowNet to sample a point in the k-dimensional probability 
        simplex conditioned on each document d. This is equivalent to sampling from the
        posterior distribution p(z_d | d) where z_d is the topic assignment vector for
        document d, and d is the document itself.
    - The reward function for sampling a topic assignment z_d for document d is the 
        joint log likelihood of p(z_d, d) ~= p(z_d)*p(d | z_d), where p(z_d) is the 
        dirichlet prior over topic assignments and p(d | z_d) is the likelihood of 
        observing document d given topic assignment z_d under a multinomial.
M-step:
    - We update the global word-topic mixture phi and the parameters of the GflowNet
        using gradient ascent on log p(phi) + log p(z_d, d), where p(phi) is the dirichlet
        prior distribution over the word-topic mixture, and p(z_d, d) is the joint 
        likelihood (and gfn reward) described above.
    - Note that this loss is somewhat analogous to the evidence lower bound (ELBO)
        in variational inference. p(d, z_d) is the expected complete log likelihood
        and p(phi) is the entropy of the word-topic mixture with respect to the dirichlet
        prior. The main difference is that we no longer need to have a variational
        approximation to the posterior distribution p(z_d | d) since we can sample
        directly from it using the trained GflowNet.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torchtyping import TensorType as TT
from utils.modules import NeuralNet
from models.lda import LDA

from utils.metrics import compute_topic_diversity, compute_topic_coherence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GFlowNet_EM(nn.Module):
    """
    This continuous gflownet samples a point in the K-dimensional probability simplex
    conditioned on the context. In our use case of LDA where K corresponds to the number
    of topics, we train a gflownet policy to sample a continuous vector of topic proportions 
    theta_d over each document d in the batch, conditioned on the word count matrix of 
    that document.

    Given that the gflownet excels at sampling discrete action trajectories to build 
    discrete compositional objects over a DAG action space, we sample theta using a 
    stick-breaking Dirichlet process as a mixing prior over an unconstrained family 
    of distributions parametrized by the gflownet. Here is the sampling process for a
    single document d

    for each 1,...,K-1 do
        infer the unnormalized topic logits pf_logits from the gfn
        infer the unnormalized mixture component logits mix_logits from the gfn
        infer the beta dist params (alpha, beta) for each mixture comp. from the gfn
        infer the flow normalizing constant Z from the gfn

        sample an unmasked topic index k from categorical(pf_logits)
        sample a mixture component m from categorical(mix_logits[k])
        sample a stick breaking proportion q from beta(alpha[k][m], beta[k][m])
        break the remaining stick into proportions q and (1-q)
        set theta_d[k] to q*remaining_density
        mask topic k
    done

    set the density for the last unassigned topic to the remaining density on the stick
    """

    K: int                      # dimension of simplex to sample from (also num of topics)
    n_mixture_components: int   # number of mixture components to use per topic
    context_dim: int            # the input dimension of the context data to condition policy on
    uniform_pb: bool            # whether to use a uniform backwards policy for TB loss
    model: nn.Module            # sampling model to use

    def __init__(
        self,
        K: int,
        context_dim: int,
        n_mixture_components: int = 4,
        n_hidden_layers: int = 3, 
        hidden_dim: int = 32,
        uniform_pb: bool = False,
        activation_fn: str = "elu",
        *args, 
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.K = K
        self.context_dim = context_dim
        self.n_mixture_components = n_mixture_components
        self.uniform_pb = uniform_pb
        self.model = NeuralNet(
            input_dim=2*K+1+self.context_dim,
            output_dim=K*(2+3*n_mixture_components)+1,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation_fn=activation_fn
        )

    def forward(
        self,
        theta: TT["num_docs", "K"],
        theta_mask: TT["num_docs", "K"],
        remaining: TT["num_docs"],
        context: TT["num_docs", "context_dim"]
    ) -> (
        TT["num_docs", "K"],
        TT["num_docs", "K"],
        TT["num_docs", "K", "n_mixture_components", 3],
        TT["num_docs"]
    ):
        """
        Samples a batch of transitions

        Parameters:
            theta: the topic proportions we wish to infer for all documents in the batch
            theta_mask: binary mask indicating which topic indices have been assigned density
            remaining: the remaining density on the stick
            context: the word count matrix for each document that we condition the sampling model on
        
        Returns:
            pb_logits: logits for the backwards policy Pb(s_{t-1}|s_t)
            pf_logits: logits for the forwards policy Pf(s_t|s_{t-1})
            pf_mixture_params: logits and beta dist parameters (alpha, beta) for 
                                all K*M mixture components
            log_flow: flow normalizing constant Z
        """
        x = torch.concat([theta, theta_mask, remaining.unsqueeze(1), context], dim=1)
        x = self.model(x)

        pb_mask = 1e9*(1-theta_mask)
        pb_logits = torch.log_softmax(x[:, :self.K] * (0 if self.uniform_pb else 1) - pb_mask, dim=1)

        pf_mask = 1e9*theta_mask
        pf_logits = torch.log_softmax(x[:, self.K:2*self.K] - pf_mask, dim=1)

        if torch.isnan(pf_logits).any():
            print(x[:, self.K:2*self.K])
            print(pf_mask)
            print(pf_logits)
            raise ValueError("pf_logits has NaNs")

        pf_mixture_params = x[:, 2*self.K:self.K*(2+3*self.n_mixture_components)]
        pf_mixture_params = pf_mixture_params.view(-1, self.K, self.n_mixture_components, 3)
        log_flow = x[:, -1]

        return pb_logits, pf_logits, pf_mixture_params, log_flow
    
    def sample_trajectories(
        self,
        context: TT["num_docs", "context_dim"],
        temperature: float = 1.,
        epsilon: float = 0.,
    ) -> (
        TT["num_docs", "K"],
        TT["num_docs"],
        TT["num_docs"],
        TT["num_docs"]
    ):
        """
        Samples a batch of full trajectories

        Parameters:
            context: the word count matrix for each document that we condition the sampling model on
            temperature: controls spikiness of softmax over topic proportions pf_logits
            epsilon: greedy epsilon for off-policy exploration
        """
        num_docs = context.shape[0]
        assert context.shape == (num_docs, self.context_dim), (
            f"Context tensor has invalid shape {context.shape}, expected ({num_docs, self.context_dim})"
        )

        # initialize topic proportions, mask, stick, etc.
        remaining = torch.ones((num_docs,)).to(device)
        theta = torch.zeros((num_docs, self.K)).to(device)
        theta_mask = torch.zeros_like(theta)
        logPF = torch.zeros_like(remaining)
        logPB = torch.zeros_like(remaining)

        for step in range(self.K+1):

            pb_logits, pf_logits, pf_mixture_params, log_flow = self.forward(
                theta, theta_mask, remaining, context
            )

            if step == 0:
                # we use the first step estimate for the flow normalizing constant Z
                logZ = log_flow
            else:
                # otherwise, we gather the backwards transition probabilities from the parent 
                # positions chosen in the previous step into the logPB estimate
                logPB += pb_logits.gather(1, cur_topic_idx).squeeze(1)
        
            if step < self.K:
                # if we haven't reached the maximum number of steps, we sample a topic index
                # from the forward transition probabilities masked by the set mask to prevent
                # illegal forward moves with shape (D,K)
                if epsilon > 0:
                    sampling_probs = (1-epsilon) * (pf_logits/temperature).softmax(dim=1) \
                        + epsilon * (1-theta_mask) / (1-theta_mask).sum(1).unsqueeze(1)
                else:
                    sampling_probs = (pf_logits/temperature).softmax(dim=1)
                
                # sample an unmasked topic index for each document in the batch with shape (D,)
                try:
                    cur_topic_idx = torch.multinomial(sampling_probs, 1)
                except:
                    print(pf_logits)
                    print(sampling_probs)
                
                # gather the forward transition probabilities for the sampled topic indices
                # into the logPF estimate (one action per document in the batch) => shape (D,)
                logPF += pf_logits.gather(1, cur_topic_idx).squeeze(1)

                # for K topics, we break the stick K-1 times
                if step < self.K-1:

                    # gather the beta parameters for the mixture components associated 
                    # with the sampled topic indices. This has shape (D,M,3) due to the squeeze
                    topic_mix_params_idx = cur_topic_idx[...,None,None]\
                        .repeat(*((1,1)+pf_mixture_params.shape[-2:]))
                    topic_mix_params = pf_mixture_params\
                        .gather(1, topic_mix_params_idx).squeeze(1)
                    
                    # gather the unnormalized weights for the chosen mixture components (index 0) with shape (D,M)
                    topic_mix_logits = topic_mix_params[...,0].log_softmax(1)
                    # sample a mixture using categorical over the mixture logits for each chosen topic per document
                    cur_mixture_idx = torch.multinomial(topic_mix_logits.exp(), 1)

                    # given that the state transitions are also defined by the choice of mixture component
                    # we need to gather the chosen mixture component logits into the logPF estimate
                    logPF += topic_mix_logits.gather(1, cur_mixture_idx).squeeze(1)
                    
                    # gather the beta parameters for the sampled mixture component for each document
                    # this has shape (D,3) since each document now has a single mixture component with 3 params
                    cur_mixture_params_idx = cur_mixture_idx[...,None]\
                        .repeat(*((1,1)+pf_mixture_params.shape[-1:]))
                    cur_mixture_params = topic_mix_params\
                        .gather(1, cur_mixture_params_idx).squeeze(1)
                    
                    # sample q from a beta distribution parametrized by the gfn estimates of the beta params
                    # for the sampled mixture in each document with shape (D,)
                    betas = torch.distributions.Beta(cur_mixture_params[:, 1].exp()+1e-2,\
                                                    cur_mixture_params[:, 2].exp()+1e-2)
                    samples = betas.sample()

                    # define the beta distribution over all M mixtures for the chosen topics shape (D,M)
                    all_betas = torch.distributions.Beta(topic_mix_params[...,1].exp()+1e-2,\
                                                        topic_mix_params[...,2].exp()+1e-2)
                    
                    # given that state transitions are also defined by the choice of sampled Qs, we need to
                    # add the probability of sampling Qs from the whole beta distribution over all mixture
                    # choices to the logPF estimate
                    # TODO: does log_prob actually marginalize over all combinations of mixture components?
                    all_log_probs = all_betas.log_prob(samples.unsqueeze(1))    # shape (D,M)

                    # P(Qs) = \int P(Qs|mixtures)P(mixtures)dmixtures
                    betas_log_probs = (all_log_probs + topic_mix_logits).logsumexp(1)
                    
                    # normalize the probability of sampling Qs by the remaining density on the stick so that 
                    # they are comparable as the amount of remaining density decreases
                    # TODO: why do we need this normalization?
                    logPF += betas_log_probs - remaining.detach().log()
                else:
                    # if we are at the last step, we just take the remaining density on the stick
                    samples = torch.ones_like(samples)
                
                # Update the sampled topic proportions to the sampled Qs (density broken off the stick)
                theta.scatter_(1, cur_topic_idx, (samples * remaining).unsqueeze(1))

                # Update the remaining density on the stick
                remaining = remaining * (1 - samples)

                # Update the set mask
                theta_mask.scatter_(1, cur_topic_idx, torch.ones(num_docs,1).to(device))
        
        return theta, logZ, logPF, logPB
    

class LDA_GFN(LDA):
    """
    This is the main model for LDA with GFlowNet-EM inference. It is a wrapper around
    the GFlowNet_EM sampling model that also defines the loss function for the model
    and the training loop.
    """

    K: int                              # dimension of simplex to sample from (also num of topics)
    V: int                              # size of the vocabulary
    D: int                              # number of documents
    phi: TT["K", "V"]                   # word-topic mixture
    eval_every: int                     # number of iterations between evaluations
    n_iter: int                         # number of iterations to train for
    policy: GFlowNet_EM                 # gflownet policy to sample topic proportions
    gfn_optimizer: optim.Adam           # optimizer for gflownet policy
    phi_optimizer: optim.Adam           # optimizer for word-topic mixture
    phi_step_threshold: float           # threshold on loss for updating phi
    save_samples: bool                  # whether to save the samples from the sampler

    def __init__(
        self,
        K: int,
        docs: [[str]],
        n_iter=10000,
        eval_every=100,
        save_samples=False,
        phi_step_threshold=1,
        **kwargs,
    ) -> None:
        super().__init__(K=K, docs=docs, **kwargs)
        
        self.n_iter = n_iter
        self.eval_every = eval_every
        self.save_samples = save_samples
        self.phi_step_threshold = phi_step_threshold
        self.phi = nn.Parameter(self.phi_prior.sample((self.K,)).log()*0.01)
        self.policy = GFlowNet_EM(K=K, context_dim=self.V).to(device)
        self.gfn_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.phi_optimizer = optim.Adam([self.phi], lr=1e-2)

    def loglikelihood(
        self,
        theta: TT["num_docs", "K"],
        phi: TT["K", "V"],
        n_dw: TT["num_docs", "V"],
    ) -> TT["num_docs"]:
        """
        Estimate the joint log likelihood of the data (word count matrix n_wd) and the
        topic assignments z_d for each document d in the batch using the current model
        parameters phi and theta.
        """
        theta_soft = theta*0.99 + 0.01/self.K
        lp = self.theta_prior.log_prob(theta_soft)
        lpw = theta_soft @ phi.softmax(dim=1)
        ll = (lpw.log() * n_dw).sum(dim=1)
        return lp + ll

    def fit(self, verbose=True, eval=True) -> ([float], [float], [float]):
        """
        Fit the model using GFlowNet-EM inference
        """
        losses = []
        updates = []
        log_rewards = []

        ll = []
        tc = []
        td = []

        for it in range(self.n_iter):
            # E-step
            theta, logZ, logPF, logPB = self.policy.sample_trajectories(self.n_dw)
            logR = self.loglikelihood(theta, self.phi, self.n_dw)
            self.gfn_optimizer.zero_grad()
            
            # step on TB loss with respect to the gfn policy parameters
            loss = ((logZ + logPF - logR - logPB)**2).mean()
            loss.backward()
            self.gfn_optimizer.step()
            
            # log the losses and rewards for plotting later
            losses.append(loss.item())
            log_rewards.append(logR.mean().item())
            
            # M-step (update phi only if the loss is below the threshold)
            should_update = loss.item() < self.phi_step_threshold
            if should_update:
                self.phi_optimizer.zero_grad()

                # loss is equal to -log p(phi) - log p(z_d, d) where the first term is the 
                # the dirichlet prior distribution over the word-topic mixture, and the second
                # term is the joint likelihood (and gfn reward)
                lp = self.phi_prior.log_prob(self.phi.softmax(dim=1)).sum() / self.D
                ll = self.loglikelihood(theta, self.phi, self.n_dw).mean()
                loss_gen = -ll-lp
                
                loss_gen.backward()
                self.phi.grad.nan_to_num_()

                # step on the optimizer for the topic-word proportions
                # Notice a crucial detail here that the updated topic-word proportions 
                # no longer sum to 1 (not a valid multinomial) after we step on the gradient.
                # To obtain an interpretable topic-word distribution, we need to renormalize
                # these proportions row-wise using a softmax or (could try) standard normalization
                self.phi_optimizer.step()

            # check how often we update the topic-word proportions
            updates.append(1 if should_update else 0)
            
            # eval every 100 iterations
            if it%self.eval_every==0:
                ll.append(logR.mean().item())
                phi = self.phi.softmax(dim=1).cpu().detach().numpy()
                tc.append(compute_topic_coherence(phi, self.tok_docs, self.r_vocab, k=10))
                td.append(compute_topic_diversity(phi, k=25))

                print(f"""
                    Iteration {it}:
                    loss={np.array(losses[-100:]).mean()},
                    log-rewards={np.array(log_rewards[-100:]).mean()},
                    update-ratio={np.array(updates[-100:]).mean()},
                    logZ={logZ.mean().item()},
                    topic-coherence={tc[-1]},
                    topic-diversity={td[-1]}
                """)
    
        return ll, tc, td
                