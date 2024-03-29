"""@matrixmaster
Main runner file to train and test different topic model

Usage:

"""

import sys
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from models.lda_gibbs import LDA_Gibbs
from models.lda_gfn import LDA_GFN
from utils.dataset import eICUDataset, mimicDataset, syntheticDataset
from utils.metrics import tokenize_docs, top_k_docs

DEFAULT_SEED = 34

def main(args):
    seed = args.seed if args.seed != 0 else DEFAULT_SEED
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # load the dataset
    if args.dataset == "eICU":
        target_paths = [
            "data/eicu/eicu_mortality_comp565.csv",
            "data/eicu/eicu_sepsis_comp565.csv",
            "data/eicu/eicu_ventilator_comp565.csv",
        ]
        dataset = eICUDataset(
            path="data/eicu/eicu_drug_comp565.csv",
            target_paths=target_paths,
            sorted=args.sorted,
            rng=rng
        )
    elif args.dataset == "MIMIC-III":
        dataset = mimicDataset(
            path="data/mimic3/MIMIC3_DIAGNOSES_ICD_subset.csv.gz",
            target_paths=None,
            sorted=args.sorted,
            rng=rng
        )
    elif args.dataset == "synthetic":
        dataset = syntheticDataset(
            K=3,
            D=256,
            V=100,
            alpha=0.1,
            beta=0.01,
            doc_length=16,
        )
    else:
        raise NotImplementedError
    
    # print the dataset summary statistics
    print(dataset)

    # split the dataset into train and test sets
    train_ids, test_ids, train_docs, test_docs = dataset.train_test_split(
        test_size=args.testset_size,
        full_size=args.dataset_size,
        shuffle=True,
    )
    
    # initialize the model
    if args.model == "LDA" and args.inference_method == "gibbs":
        model = LDA_Gibbs(
            rng=rng,
            K=args.K,
            alpha=0.1,
            beta=0.01,
            docs=train_docs,
            vocab=dataset.vocab,
            r_vocab=dataset.r_vocab,
            n_iter=args.n_iter,
            patience=args.patience,
            save_samples=args.save_samples,
            eval_every=1,
        )
    elif args.model == "LDA" and args.inference_method == "gfn":
        model = LDA_GFN(
            K=args.K,
            alpha=0.1,
            beta=0.1,
            docs=train_docs,
            vocab=dataset.vocab,
            r_vocab=dataset.r_vocab,
            n_iter=args.n_iter,
            save_samples=args.save_samples,
            phi_step_threshold=1.,
            gfn_lr=0.001,
            phi_lr=0.01,
            Z_lr=0.1,
            # n_mixture_components=4,
            # n_hidden_layers=3,
            # hidden_dim=32,
            eval_every=1,
        )
    else:
        raise NotImplementedError
    
    # fit the model
    ll, tc, td = model.fit(verbose=True, eval=True)

    # # make results folder
    # folder = f"results/{args.dataset}_{args.model}_{args.inference_method}_{seed}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # os.makedirs(folder, exist_ok=True)

    # # save the topic mixtures and metrics
    # phi, theta = model.get_topic_mixtures()
    # topic_names = [f"topic_{i}" for i in range(model.K)]

    # phi_df = pd.DataFrame(phi, columns=model.r_vocab, index=topic_names)
    # phi_df.to_csv(f"{folder}/phi.csv")

    # theta_df = pd.DataFrame(theta, columns=topic_names, index=train_ids)
    # theta_df.to_csv(f"{folder}/theta.csv")

    # np.save(f"{folder}/ll.npy", ll)
    # np.save(f"{folder}/tc.npy", tc)
    # np.save(f"{folder}/td.npy", td)

    with torch.no_grad():
        for iter in range(3):
            i = np.random.randint(256)
            doc_ = model.n_dw[i:i+1].repeat(256,1)

            plt.figure(figsize=(6,3))
            plt.subplot(121)
            plt.title('sampled posterior')

            theta, logPF, logPB = model.policy.sample_trajectories(doc_)
            logR = model.loglikelihood(theta,model.phi,doc_)

            plt.scatter((theta[:,0]+theta[:,1]/2).cpu(), (theta[:,1]*np.sqrt(3)/2).cpu(),s=2)
            plt.scatter([0,0.5,1],[0,np.sqrt(3)/2,0])

            plt.subplot(122)
            plt.title('true posterior')

            # importance sampling num_docsx100
            theta_samples = dataset.theta_prior.sample((256*100,)).to('cuda')
            logR = model.loglikelihood(theta_samples,model.phi,doc_.repeat(100,1))
            theta_idx = torch.distributions.Categorical(logits=logR).sample((256,))
            best_thetas = theta_samples[theta_idx]

            plt.scatter((best_thetas[:,0]+best_thetas[:,1]/2).cpu(), (best_thetas[:,1]*np.sqrt(3)/2).cpu(),s=2)
            plt.scatter([0,0.5,1],[0,np.sqrt(3)/2,0])
            plt.savefig(f'./results/test-synthetic-theta-doc-{iter}.png')

        # test phi
        plt.figure(figsize=(15,3), facecolor='white')

        plt.subplot(211)
        plt.title('Inferred with GFN')
        plt.imshow(model.phi.log_softmax(1).cpu().numpy(),vmax=-1,vmin=-5,cmap='Blues')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('vocab')
        plt.ylabel('topic')

        plt.subplot(212)
        plt.title('Ground truth')
        plt.imshow(dataset.phi.log_softmax(1).cpu().numpy(),vmax=-1,vmin=-5,cmap='Blues')
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('vocab')
        plt.ylabel('topic')
        plt.savefig(f'./results/test-synthetic-phi.png')
        
        for mat in [ model.phi.softmax(1), dataset.phi.softmax(1) ]:
            for i in range(dataset.K):
                print(i)
                print(*mat[i].topk(10).indices.cpu().numpy())
    #             for v,n in zip(*mat[i].topk(10)):
    #                 print(n.item(),v.item())

            print()

    """
    # save only the samples for the top document for each topic
    # also save the document tokens for the top document for each topic
    if args.save_samples:
        top_docs = top_k_docs(theta, list(train_ids), k=1)[0]
        top_docs_idx = [doc[1] for doc in top_docs]
        top_docs_ids = [doc[0] for doc in top_docs]
        samples = model.assign[top_docs_idx, :, :]
        
        np.save(f"{folder}/samples.npy", samples)
        np.save(f"{folder}/samples_ids.npy", top_docs_ids)

        tok_docs = tokenize_docs(train_docs, dataset.vocab)
        tok_docs = [tok_docs[doc] for doc in top_docs_idx]

        with open(f"{folder}/docs.pkl", "wb") as f:
            pickle.dump(tok_docs, f)

    # infer the test set and evaluate the model perplexity
    infered_docs, test_docs = dataset.split_documents(test_docs, 0.5)
    ll, theta, n_dk, samples = model.infer(infered_docs, patience=5)
    test_ll = model.loglikelihood(docs=test_docs, n_dk=n_dk)

    norm = np.sum([len(doc) for doc in test_docs])
    ppl = np.exp(-test_ll / norm)

    print(f"Test perplexity: {ppl}")

    # save the perplexity, samples, and topic mixtures of the test set
    np.savetxt(f"{folder}/test_ppl.txt", [ppl])
    np.save(f"{folder}/test_ll.npy", ll)

    theta_df = pd.DataFrame(theta, columns=topic_names, index=test_ids)
    theta_df.to_csv(f"{folder}/test_theta.csv")

    if args.save_samples:
        top_docs = top_k_docs(theta, list(test_ids), k=1)[0]
        top_docs_idx = [doc[1] for doc in top_docs]
        top_docs_ids = [doc[0] for doc in top_docs]
        samples = samples[top_docs_idx, :, :]

        np.save(f"{folder}/test_samples_ids.npy", top_docs_ids)
        np.save(f"{folder}/test_samples.npy", samples)

        tok_docs = tokenize_docs(test_docs, dataset.vocab)
        tok_docs = [tok_docs[doc] for doc in top_docs_idx]

        with open(f"{folder}/test_docs.pkl", "wb") as f:
            pickle.dump(tok_docs, f)
    """


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed for reproducibility")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["eICU", "MIMIC-III", "synthetic"],
        default="eICU",
        help="dataset to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["LDA", "ETM"],
        default="LDA",
        help="model to use"
    )
    parser.add_argument(
        "--inference_method",
        type=str,
        choices=["gibbs", "variational", "gfn"],
        default="gibbs",
        help="inference method to use"
    )
    parser.add_argument("--K", type=int, default=10, help="number of topics")
    parser.add_argument(
        "--n_iter",
        type=int,
        default=10,
        help="number of iterations to run the sampler"
    )
    parser.add_argument(
        "--sorted",
        action="store_true",
        help="whether to process the document words in order",
        default=False
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="whether to stop the sampler early",
        default=False
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="number of iterations to wait for convergence"
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="whether to save the samples from the sampler",
        default=False
    )
    parser.add_argument(
        "--dataset_size",
        type=float,
        default=1.0,
        help="percentage of the dataset to use"
    )
    parser.add_argument(
        "--testset_size",
        type=float,
        default=0.2,
        help="percentage of the dataset to use as the test set"
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="evaluate the model every eval_every iterations"
    )

    args = parser.parse_args()
    main(args)
    