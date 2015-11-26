import numpy as np
import scipy as sp
import scipy.special as spec
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import sklearn.cluster

def rho(tau,kappa,t):
    return (tau+t)**(-kappa)

def digamma(mat):
    if (len(mat.shape) == 1):
        return(spec.psi(mat) - spec.psi(np.sum(mat)))
    else:
        return(spec.psi(mat) - spec.psi(np.sum(mat, 0))[np.newaxis,:])

def lda(dtm,ntopic,tau,kappa,itemax):
    nvoc = dtm.shape[1]
    ndoc = dtm.shape[0]
    nu = 1./ndoc
    alpha = 1./ndoc
    
    topics = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    phi = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    gamma  = np.random.gamma(100.,1./100.,(ndoc,ntopic))
    
    intint = 100
       
    for t in range(itemax):
        for d in range(ndoc):
            ids = np.nonzero(dtm[d,:])[0]
            cts = dtm[d,ids]
            gamma_ = gamma[d,:]
            # Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
            Elogbeta = digamma(topics)[ids,:]
            dott = lambda x: np.dot(x,cts)
            for tt in range(intint):
                old_gamma_ = gamma_
                Elogtheta = digamma(gamma_)
                E_sum = Elogbeta + Elogtheta
                phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.00001:
                    break  
            gamma[d,:] = gamma_
            topics_temp = nu + ndoc * phi * dtm[d,ids][:, np.newaxis]
            rt = rho(tau,kappa,(t+1)*d)
            topics[ids,:] = (1-rt)*topics[ids,:] + rt*topics_temp
    return topics,gamma

def lda_kmeans(dtm,ntopic,tau,kappa,itemax, kmeans = True):
    nvoc = dtm.shape[1]
    ndoc = dtm.shape[0]
    nu = 1./ndoc
    alpha = 1.
    
    topics = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    phi = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    gamma  = np.random.gamma(100.,1./100.,(ndoc,ntopic))
    
    intint = 100
    if kmeans:
        for t in range(itemax):
            if t == 0:
                for d in range(ndoc):
                    old_topics = topics
                    ids = np.nonzero(dtm[d,:])[0]
                    cts = dtm[d,ids]
                    gamma_ = gamma[d,:]
                    # Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
                    Elogbeta = digamma(topics)[ids,:]
                    dott = lambda x: np.dot(x,cts)
                    for tt in range(intint):
                        old_gamma_ = gamma_
                        Elogtheta = digamma(gamma_)
                        E_sum = Elogbeta + Elogtheta
                        phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                        gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                        if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.0001:
                            break  
                    gamma[d,:] = gamma_
                    topics_temp = nu + ndoc * phi * dtm[d,ids][:, np.newaxis]
                    rt = rho(tau,kappa,(t+1)*d)
                    topics[ids,:] = (1-rt)*topics[ids,:] + rt*topics_temp
            else:
                for d in sorted_index:
                    old_topics = topics
                    ids = np.nonzero(dtm[d,:])[0]
                    cts = dtm[d,ids]
                    gamma_ = gamma[d,:]
                    # Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
                    Elogbeta = digamma(topics)[ids,:]
                    dott = lambda x: np.dot(x,cts)
                    for tt in range(intint):
                        old_gamma_ = gamma_
                        Elogtheta = digamma(gamma_)
                        E_sum = Elogbeta + Elogtheta
                        phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                        gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                        if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.0001:
                            break  
                    gamma[d,:] = gamma_
                    topics_temp = nu + ndoc * phi * dtm[d,ids][:, np.newaxis]
                    rt = rho(tau,kappa,(t+1)*d)
                    topics[ids,:] = (1-rt)*topics[ids,:] + rt*topics_temp
            sorted_model  = sklearn.cluster.KMeans(n_clusters  = ntopic)
            sorted_index = np.argsort(sorted_model.fit_predict(gamma))

    else:
        for t in range(itemax):
            for d in range(ndoc):
                old_topics = topics
                ids = np.nonzero(dtm[d,:])[0]
                cts = dtm[d,ids]
                gamma_ = gamma[d,:]
                # Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
                Elogbeta = digamma(topics)[ids,:]
                dott = lambda x: np.dot(x,cts)
                for tt in range(intint):
                    old_gamma_ = gamma_
                    Elogtheta = digamma(gamma_)
                    E_sum = Elogbeta + Elogtheta
                    phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                    gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                    if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.0001:
                        break  
                gamma[d,:] = gamma_
                topics_temp = nu + ndoc * phi * dtm[d,ids][:, np.newaxis]
                rt = rho(tau,kappa,(t+1)*d)
                topics[ids,:] = (1-rt)*topics[ids,:] + rt*topics_temp

    return topics,gamma,tau,kappa

def batch_variational(dtm,ntopic,itemax,ordering=False,start = False, lambda_ = None):
    nvoc = dtm.shape[1]
    ndoc = dtm.shape[0]
    nu = 1./ntopic
    alpha = 1./ntopic

    topics = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    if start:
        topics = lambda_
    phis = [np.random.gamma(100.,1./100.,(nvoc,ntopic)) for i in range(ndoc)]
    topics_temp = [np.random.gamma(100.,1./100.,(nvoc,ntopic)) for i in range(ndoc)]
    gamma  = np.random.gamma(100.,1./100.,(ndoc,ntopic))

    intint = 100
    # idx = []
    # for i in range(itemax):
    #   indx = range(ndoc)
    #   np.random.shuffle(indx)
    #   idx.extend(indx)
    if ~ordering:
        for it in range(itemax):
            for t in range(ndoc):
                cts = dtm[t,:]
                gamma_ = gamma[t,:]
                Elogbeta = digamma(topics)

                for tt in range(intint):
                    old_gamma_ = gamma_
                    Elogtheta = digamma(gamma_)
                    E_sum = Elogbeta + Elogtheta
                    phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                    gamma_ = alpha + np.dot(phi.T,cts)
                    if np.mean(abs(gamma_ - old_gamma_))<0.00001:
                        break
                gamma[t,:] = gamma_
                topics_temp[t] = phi * dtm[t,:][:, np.newaxis]

            topics = np.ones(topics.shape)*nu    
            for t in range(ndoc):
                topics += topics_temp[t]
    else:
        for it in range(itemax):
            sorted_index = range(ndoc)
            for t in range(ndoc):
                cts = dtm[t,:]
                gamma_ = gamma[t,:]
                Elogbeta = digamma(topics)

                for tt in range(intint):
                    old_gamma_ = gamma_
                    Elogtheta = digamma(gamma_)
                    E_sum = Elogbeta + Elogtheta
                    phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                    gamma_ = alpha + np.dot(phi.T,cts)
                    if np.mean(abs(gamma_ - old_gamma_))<0.00001:
                        break
                gamma[t,:] = gamma_
                topics_temp[t] = phi * dtm[t,:][:, np.newaxis]

            topics = np.ones(topics.shape)*nu    
            for t in range(ndoc):
                topics += topics_temp[t]
            sorted_model  = sklearn.cluster.KMeans(n_clusters  = ntopic)
            sorted_index = np.argsort(sorted_model.fit_predict(gamma))

    return topics,gamma

def lda_minibatch_ordering(dtm,ntopic,mb_size,tau,kappa,itemax,start = False,lambda_ = None):
    nvoc = dtm.shape[1]
    ndoc = dtm.shape[0]
    nu = 1./ndoc
    alpha = 1./ndoc
    
    topics = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    if start:
        topics = lambda_    
    phi = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    gamma  = np.random.gamma(100.,1./100.,(ndoc,ntopic))
    
    intint = 100
    idx = range(ndoc)
    minibatchesids = np.array_split(idx,len(idx)/mb_size)

    for it in range(itemax):
        topics_temp = np.zeros(topics.shape)
        indices = []
        for t in range(len(minibatchesids)):
            for id_ in minibatchesids[t]:
                doc = idx[id_]
                ids = np.nonzero(dtm[doc,:])[0]
                indices.extend(ids)
                cts = dtm[doc,ids]
                gamma_ = gamma[doc,:]
                Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
                dott = lambda x: np.dot(x,cts)
                
                for tt in range(intint):
                    old_gamma_ = gamma_
                    Elogtheta = digamma(gamma_)
                    E_sum = Elogbeta + Elogtheta
                    phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                    gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                    if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.000001:
                        break  
                gamma[doc,:] = gamma_
                topics_temp[ids,:] += phi * dtm[doc,ids][:, np.newaxis]
            
        indices  = np.unique(indices)
        topics_temp[indices,:] = nu + (ndoc/len(minibatchesids[t])) * topics_temp[indices,:]
        rt = rho(tau,kappa,it)
        topics[indices ,:] = (1-rt)*topics[indices,:] + rt*topics_temp[indices,:]

        if it % 2 == 0:
            sorted_model  = sklearn.cluster.KMeans(n_clusters  = ntopic)
            sorted_index = np.argsort(sorted_model.fit_predict(gamma))
            minibatchesids = np.array_split(sorted_index,len(sorted_index)/mb_size)

    return topics,gamma
    
def lda_minibatch(dtm,ntopic,mb_size,tau,kappa,itemax,start = False,lambda_ = None):
    nvoc = dtm.shape[1]
    ndoc = dtm.shape[0]
    nu = 1./ndoc
    alpha = 1./ndoc
    
    topics = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    if start:
        topics = lambda_ 
    phi = np.random.gamma(100.,1./100.,(nvoc,ntopic))
    gamma  = np.random.gamma(100.,1./100.,(ndoc,ntopic))
    
    intint = 100
    idx = range(ndoc)
    #np.random.shuffle(idx)
    idx *= itemax
    #np.random.shuffle(idx)
    minibatchesids = np.array_split(idx,len(idx)/mb_size)
    
    for t in range(len(minibatchesids)):
        topics_temp = np.zeros(topics.shape)
        indices = []
        
        for id_ in minibatchesids[t]:
            doc = idx[id_]
            ids = np.nonzero(dtm[doc,:])[0]
            indices.extend(ids)
            cts = dtm[doc,ids]
            gamma_ = gamma[doc,:]
            Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
            dott = lambda x: np.dot(x,cts)
            
            for tt in range(intint):
                old_gamma_ = gamma_
                Elogtheta = digamma(gamma_)
                E_sum = Elogbeta + Elogtheta
                phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.000001:
                    break  
            gamma[doc,:] = gamma_
            topics_temp[ids,:] += phi * dtm[doc,ids][:, np.newaxis]
            
        indices  = np.unique(indices)
        topics_temp[indices,:] = nu + (ndoc/len(minibatchesids[t])) * topics_temp[indices,:]
        rt = rho(tau,kappa,t)
        topics[indices ,:] = (1-rt)*topics[indices,:] + rt*topics_temp[indices,:]
    
    return topics,gamma,tau,kappa,nu,alpha

def inference(lda,newdocs,ite):
    alpha = lda[5]
    nu = lda[4]
    topics = lda[0]
    tau = lda[2]
    kappa = lda[3]
    
    phi = np.random.gamma(100.,1./100.,(topics.shape))
    
    if len(newdocs.shape)==1:
        gamma_new  = np.random.gamma(100.,1./100.,(1,lda[1].shape[1]))
        for it in range(ite):
            ids = np.nonzero(newdocs)[0]
            cts = newdocs[ids]
            gamma_ = gamma_new
            Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
            dott = lambda x: np.dot(x,cts)
            for tt in range(100):
                old_gamma_ = gamma_
                Elogtheta = digamma(gamma_)
                E_sum = Elogbeta + Elogtheta
                phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.0001:
                    break  
            gamma_new = gamma_
            topics_temp = nu +  phi * cts[:, np.newaxis]
            rt = rho(tau,kappa,it)
            topics[ids,:] = (1-rt)*topics[ids,:] + rt*topics_temp
    
    else:
        gamma_new  = np.random.gamma(100.,1./100.,(newdocs.shape[0],lda[1].shape[1]))
        idx = range(newdocs.shape[0])
        idx *= ite
        np.random.shuffle(idx)
        for it in range(len(idx)):
            doc = idx[it]
            ids = np.nonzero(newdocs[doc,:])[0]
            cts = newdocs[doc,ids]
            gamma_ = gamma_new[doc,:]
            Elogbeta = np.apply_along_axis(digamma, axis=0, arr=topics)[ids,:]
            dott = lambda x: np.dot(x,cts)
            for tt in range(100):
                old_gamma_ = gamma_
                Elogtheta = digamma(gamma_)
                E_sum = Elogbeta + Elogtheta
                phi = np.exp(E_sum)/np.exp(E_sum).sum(axis=1)[:, np.newaxis]
                gamma_ = alpha + np.apply_along_axis(dott,axis=0,arr=phi)
                if np.sqrt(np.mean((gamma_-old_gamma_)**2))<0.00001:
                    break  
            gamma_new[doc,:] = gamma_
            topics_temp = nu +  phi * cts[:, np.newaxis]
            rt = rho(tau,kappa,it)
            topics[ids,:] = (1-rt)*topics[ids,:] + rt*topics_temp
            
    return topics,gamma_new

def perplexity_test(lda,newdocs,ite,perword = False):
    
    new = inference(lda,newdocs,ite)
    
    topics = new[0]
    gammas = new[1]
    
    topics = topics/topics.sum(axis=0)
    
    if len(gammas.shape) == 1:
        gammas = gammas/np.sum(gammas)
        doc_idx = np.nonzero(newdocs)[0]
        doc_cts = newdocs[doc_idx]
        return np.exp(-np.log(np.sum(np.dot(topics[doc_idx,:],gammas)*doc_cts))/np.sum(doc_cts))
    
    else:
        norm = lambda x: x/np.sum(x)
        gammas = np.apply_along_axis(norm,axis = 1,arr = gammas)
        
        num = 0
        denom = 0
        
        for i in range(gammas.shape[0]):
            doc_idx = np.nonzero(newdocs[i,:])[0]
            doc_cts = newdocs[i,doc_idx]
            num = np.sum(np.log(np.dot(topics[doc_idx,:],gammas[i,:]))*doc_cts)
            denom += np.sum(doc_cts)
            
        if ~perword:
            return num
        else:
            return num/denom

def show_topics(lda,whichtopic,topwords,wordlist):
    topics = lda[0]
    for t in whichtopic:
        idx = np.argsort(topics[:,t])[::-1][:topwords].astype(int)
        words = []
        for i in idx:
            words.append(wordlist[i])
        print "Topic %s:" % t
        print ""
        print words
        print ""
