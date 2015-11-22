import numpy as np
import scipy as sp
import scipy.special as spec
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

dtm = np.zeros((100, 100))
dtm_ = np.zeros((10, 100))
for i in range(10):
    dtm_[i,] = [0]*10*i + [10]*10 + [0]*(10*(9-i))
for i in range(10):
    dtm[10*i:10*(i+1),]= dtm_

def rho(tau,kappa,t):
    return (tau+t)**(-kappa)

def digamma(row):
    return spec.psi(row)-spec.psi(np.sum(row))

nvoc = 100.
ndoc = 100.
ntopic = 10.
nu = 1./100
alpha = 1.

topics = np.random.gamma(100.,1./100.,(100,10))
phi = np.random.gamma(100.,1./100.,(100,10))
gamma  = np.random.gamma(100.,1./100.,(100,10))

tau = 10
kappa = 0.5
itemax = 100
intint = 100
idx = range(100)
np.random.shuffle(idx)
idx *= 3
for t in range(itemax):
    old_topics = topics
    doc = idx[t]
    ids = np.nonzero(dtm[doc,:])[0]
    cts = dtm[doc,ids]
    gamma_ = gamma[doc,:]
    Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
    dott = lambda x: np.dot(x,cts)
    for tt in range(intint):
        old_gamma_ = gamma_
        Elogtheta = digamma(gamma_)
        E_sum = Elogbeta + Elogtheta
        phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)
        gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
        if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.0001:
            break  
    gamma[doc,:] = gamma_
    topics_temp = nu + ndoc * phi * dtm[doc,ids][:, np.newaxis]
    rt = rho(tau,kappa,t)
    topics[ids,:] = (1-rt)*topics[ids,:] + rt*topics_temp
    #if np.sqrt(np.mean((old_topics-topics)**2))<0.001:
    #    break

for i in range(10):
    plt.plot(range(100),topics[:,i])
    plt.show()
    raw_input()