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