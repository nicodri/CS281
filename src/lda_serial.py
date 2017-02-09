import numpy as np
from scipy.special import psi
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from lda_cython import e_step

eps = 1e-100


def digamma(data):
    if (len(data.shape) == 1):
        return psi(data) - psi(np.sum(data))
    return psi(data) - psi(np.sum(data, axis=1))[:, np.newaxis]


def get_samples(C, S, max_iter):
    batches_temp = np.zeros(C * max_iter, dtype=int)
    sample = np.arange(C, dtype=int)
    for k in xrange(max_iter):
        #np.random.shuffle(sample)
        batches_temp[k * C: (k + 1) * C] = sample
    # List of mini-batches
    batches = np.split(batches_temp, C * max_iter / S)
    return batches


def batch_lda(corpus, lambda_=None, num_topics=10, num_iter=10, alpha=0.5, eta=0.001, threshold=0.000001):
    '''
    Batch Variational Inference EM algorithm for LDA, goes over all the data at each iteration.
    (from algorithm 1 in Blei 2010)
    corpus is a list of lists of [word_index, count] for each document
    corpus is a matrix of count: (docs, voca)
    Args:
        lambda_: to set a specific lambda for the initialization
    '''
    C, V = corpus.shape

    # Initialisation
    if not np.any(lambda_):
        lambda_ = np.random.gamma(100, 1./100, size=(num_topics, V))
    else:
        lambda_ = lambda_.copy()

    gamma_d_k = np.ones((C, num_topics))
    sample = range(C)
    np.random.shuffle(sample)

    for t in xrange(num_iter):
        old_lambda_ = lambda_
        # #### E-step
        lambda_int = np.zeros((num_topics, V))
        for d in sample:
            gamma_d_k, lambda_int = e_step(d, corpus, gamma_d_k, lambda_,
                                           lambda_int, alpha, threshold)

        # #### M-step
        lambda_ = eta + lambda_int

        # Check if convergence
        if (np.mean(np.abs((lambda_ - old_lambda_) / old_lambda_)) < threshold):
            break

    return lambda_, gamma_d_k


def stochastic_lda(corpus, batches=None, lambda_=None,
                   ordering=False, S=1, num_topics=10, max_iter=300, tau=1,
                   kappa=0.5, alpha=0.5, eta=0.001, threshold=0.000001):
    '''
    Stochastic Variational Inference EM algorithm for LDA.
    (from algorithm 2 in Blei 2010)
    corpus is a list of lists of [word_index, count] for each document
    corpus is a matrix of count: (docs, voca)
    Args:
        lambda_: to set a specific lambda for the initialization
        batches: to set an order on the use of the corpus
        S: size of the mini-batches
    '''
    C, V = corpus.shape

    # Initialisation
    if not np.any(lambda_):
        lambda_ = np.random.gamma(100, 1./100, size=(num_topics, V))
    else:
        lambda_ = lambda_.copy()

    gamma_d_k = np.ones((C, num_topics))

    # Sampling
    if not np.any(batches):
        batches = get_samples(C, S, max_iter)

    for t in xrange(len(batches)):
        # #### E-step
        lambda_int = np.zeros((num_topics, V))

        for d in batches[t]:
            gamma_d_k, lambda_int = e_step(d, corpus, gamma_d_k, lambda_, lambda_int, alpha, threshold)

        # #### M-step
        rho = (tau + t)**(-kappa)
        indices = np.unique(np.nonzero(corpus[batches[t], :])[1])
        lambda_int = eta + C / (1. * S) * lambda_int
        lambda_[:, indices] = (1 - rho)*lambda_[:, indices] + rho*lambda_int[:, indices]

    return lambda_, gamma_d_k


def e_step(d, corpus, gamma_d_k, lambda_, lambda_int, alpha, threshold):
    # Info for the current doc
    ids = np.nonzero(corpus[d, :])[0]
    counts = corpus[d, ids]

    gamma_d = gamma_d_k[d, :]
    E_log_beta = digamma(lambda_)[:, ids]
    for t in xrange(100):  # TODO: Wait for convergence
        # Used to check convergence
        old_gamma = gamma_d

        E_log_thetad = digamma(gamma_d)

        # shape of phi is (num_topics, len(ids))
        phi = np.exp(E_log_beta + E_log_thetad[:, np.newaxis])
        phi /= phi.sum(axis=0)

        gamma_d = alpha + np.dot(phi, counts)

        # Normalization of gamma
        # gamma_d /= np.sum(gamma_d)

        # Check if convergence
        if (np.mean((gamma_d - old_gamma)**2) < threshold):
            break

    gamma_d_k[d, :] = gamma_d
    lambda_int[:, ids] += counts[np.newaxis, :] * phi

    return gamma_d_k, lambda_int


# Display the selected topics
def print_topic_words(lambda_, vocabulary, num_topics, num_words):
    '''
    Display the first num_words for the topic distribution lambda_ from a
    vocabulary.
    '''
    for t in xrange(num_topics):
        topic_distribution = sorted([(i, p) for i, p in enumerate(lambda_[t, :])], key=lambda x: x[1], reverse=True)
        top_words = [vocabulary[tup[0]] for tup in topic_distribution[:num_words]]
        print 'Topic number ', t
        print top_words
