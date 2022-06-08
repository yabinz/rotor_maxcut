import numpy as np

def procedure_cut(nqs, config):
    num_nodes = config['num_visible']
    ind_increase = np.argsort(nqs.state)

    # sort and shift to [0,2*pi)
    state_shift = nqs.state[list(ind_increase)] + np.pi 

    alpha = 0.0
    gamma = -1.0e+15
    index_i = 0

    if np.max(state_shift) <= np.pi :
        index_j = num_nodes
    else:
        index_j = np.min ( np.where (state_shift > np.pi ))

    state_shift = np.append(  state_shift, [2 * np.pi + 0.1] )
    # print("index_i=" + str(index_i) +", index_j="+str(index_j) + ", gama_value=" + str(gamma))

    while alpha <= np.pi:
        ind_neg = np.where(np.logical_or ( state_shift[range(num_nodes)] < alpha,
                                           state_shift[range(num_nodes)] >= (alpha + np.pi) ) )
        ind_neg = ind_neg[:][0]
        cut = np.ones(num_nodes)
        cut[ind_neg] = -1
        cut2 = np.ones(num_nodes)
        cut2[list(ind_increase)] = cut
        cut_value = evaluate_cut_value(cut2,config)
        if cut_value > gamma:
            gamma = cut_value
            cut_global = np.copy(cut2)
        if state_shift[index_i] <= (state_shift[index_j] - np.pi):
            alpha = state_shift[index_i]
            index_i += 1
        else:
            alpha = state_shift[index_j] - np.pi
            index_j +=1
        # print("index_i=" + str(index_i) +", index_j="+str(index_j) + ", cut_value=" + str(cut_value)+", gama_value=" + str(gamma) + ",alpha=" +str(alpha/np.pi)+" pi")
    
    return cut_global, gamma
 

def evaluate_cut_value( cut, config):
    num_nodes = config['num_visible']
    wmat = np.zeros((num_nodes,num_nodes))
    for edge in config['edges']:
        j = edge['j']
        k = edge['k']
        G = edge['G']
        wmat[j][k] = -2.0 * G

    xmat=np.tensordot(cut,cut,axes=0)
    return 0.5*np.sum(np.multiply((1-xmat),wmat))





