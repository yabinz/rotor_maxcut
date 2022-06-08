# import argparse
# import os

import time
# from datetime import datetime
# from numpy.random import seed
import numpy as np
from sampler import metropolis_sampler,metropolis_sampler2, metropolis_sampler_fast, mean
# from logger import Logger
# from util import read_config

from optimization import stoch_reconfig, stoch_reconfig_precond,stoch_reconfig_fast,stoch_reconfig_fast_precond
from optimization import stoch_reconfig_fast_debug
# # procedure cut from BMZ paper
from cut import procedure_cut
from util import average,cartesian
import copy
from numpy.random import uniform, normal
from new_logger import Logger


# seed(1)


def vmc_solve(config, nqs=None,logger=None):
    

    if config['debug']:
        from nqs_sho import NQS, propose_update
        from wavefunction_sho import local_energy, log_psi, log_psi_vars
    else:
        from nqs import NQS, propose_update
        from wavefunction import local_energy, log_psi, log_psi_vars

    if nqs is None:
        nqs = NQS(config=config) # initialize nqs with the n and H from config file

    # start = time.time()
    for step in range(1, config['num_step'] + 1):
        # averages, nqs = metropolis_sampler(step,
        #                                    local_energy,
        #                                    log_psi,
        #                                    log_psi_vars,
        #                                    nqs_init=nqs,
        #                                    config=config,
        #                                    propose_update=propose_update,
        #                                    logger=logger)
        averages, nqs_history, start_index = metropolis_sampler2(step,
                                           local_energy,
                                           log_psi,
                                           log_psi_vars,
                                           nqs_init=nqs,
                                           config=config,
                                           propose_update=propose_update,
                                           logger=logger)
        nqs = nqs_history[-1]
        # vars = stoch_reconfig_precond(step, nqs, averages, config, logger)
        vars = stoch_reconfig(step, nqs, averages, config, logger)
        nqs = NQS(config=config, state=nqs.state, vars=vars)
        logger.write_step(step)
    
    nqs_local = nqs_history[start_index]
    cut_local, cut_value_local = procedure_cut(nqs_local, config)
    cut_list=[cut_local]
    cut_value_list=[cut_value_local]
    for nqs_index in range(start_index + 1,len(nqs_history)):
        nqs_local = nqs_history[nqs_index]
        cut_local, cut_value_local = procedure_cut(nqs_local, config)
        cut_list.append(cut_local)
        cut_value_list.append(cut_value_local)
    ind_increase = np.argsort(np.array(cut_value_list))
    ind_best=ind_increase[-1]
    cut = cut_list[ind_best]
    cut_value = cut_value_list[ind_best]
    print_energy(config['num_step'], averages)
    # end = time.time()
    # print('Calculation time for rotor solve', end - start)

    # start = time.time()
    # cut, cut_value = procedure_cut(nqs, config)
    # end = time.time()
    # print('Calculation time for procedure-cut from BMZ', end - start)

    # num_nodes=config['num_visible']
    # for ind in range(num_nodes):
    #     print("x_" + str(ind) + " = " + str(cut[ind]))

    # print("cut value = " + str(cut_value))
    # return nqs
    return cut, cut_value

# test nontrivial initialzation for QAOA
def vmc_solve2(config, nqs=None,logger=None):
    

    if config['debug']:
        from nqs_sho import NQS, propose_update
        from wavefunction_sho import local_energy, log_psi, log_psi_vars
    else:
        from nqs import NQS, propose_update
        from wavefunction import local_energy, log_psi, log_psi_vars

    if nqs is None:
        nqs = NQS(config=config) # initialize nqs with the n and H from config file
        theta_vals=[7.389002409260946e-03,
                2.611814809446297e+00,
                3.182795244605068e+00,
                2.874242180664720e-01,
                5.304688967007147e+00,
                2.456711080003687e+00,
                1.356445208545408e-01,
                4.195448111187450e+00,
                6.020576991150995e+00,
                1.089877228640619e+00,
                3.782904217009150e+00,
                2.894981913654509e+00]
        cs_vec=cartesian(theta_vals)
        
        num_visible, num_hidden = config['num_visible'], config['num_hidden']
        
        vars = np.concatenate([np.reshape(cs_vec,(2*num_visible,)), 
                                np.zeros((2*num_hidden,)),
                                normal(0.0,1.0,(num_hidden*num_visible,))])
        nqs = NQS(config=config, vars=vars) 

    # start = time.time()
    time_sample=0.0
    time_optimize=0.0
    for step in range(1, config['num_step']):
        start = time.time()
        averages, nqs = metropolis_sampler(step,
                                            local_energy,
                                            log_psi,
                                            log_psi_vars,
                                            nqs_init=nqs,
                                            config=config,
                                            propose_update=propose_update,
                                            logger=logger)
        end = time.time()
        time_sample += (end - start)
        start = time.time()
        vars = stoch_reconfig(step, nqs, averages, config, logger)
        
        nqs = NQS(config=config, state=nqs.state, vars=vars)
        end = time.time()
        time_optimize += (end - start)
        # print('Success at step ' + str(step))

    start = time.time()    
    averages, nqs_history, start_index = metropolis_sampler2(config['num_step'],
                                                             local_energy,
                                                             log_psi,
                                                              log_psi_vars,
                                                              nqs_init=nqs,
                                                              config=config,
                                                              propose_update=propose_update,
                                                              logger=logger)
    end = time.time()
    time_sample += (end - start)

    start = time.time() 
    nqs=nqs_history[-1]
    nqs_local = nqs_history[start_index]
    cut_local, cut_value_local = procedure_cut(nqs_local, config)
    cut_list=[cut_local]
    cut_value_list=[cut_value_local]
    for nqs_index in range(start_index + 1,len(nqs_history)):
        nqs_local = nqs_history[nqs_index]
        cut_local, cut_value_local = procedure_cut(nqs_local, config)
        cut_list.append(cut_local)
        cut_value_list.append(cut_value_local)
        #import pdb; pdb.set_trace()
    end = time.time()
    time_cut = end - start
    print(' --------------------------- ')
    print(' Total sampling time: ', time_sample)
    print(' Rest of gradient descent time: ', time_optimize)
    print(' Total cut generating time: ', time_cut)

    print_energy(config['num_step'], averages)
    print('Printing cuts from samples to file...')   
    cut_file_name =     'history_cut_file.txt'
    cut_file = open(cut_file_name, 'a')
    cut_file.write('\n num_visible = '+ str(config['num_visible']) +
                    ', num_hidden = ' + str(config['num_hidden']))
    ind_increase = np.argsort(np.array(cut_value_list))
    ind_best=ind_increase[-1]
    ind_last=len(cut_value_list)-1
    cut_file.write('\n best observed: ' + str(cut_value_list[ind_best]) )
    posnode_list=np.where(cut_list[ind_best] > 0)
    posnode_list=posnode_list[:][0].tolist()
    cut_file.writelines(", %d " % node_index  for node_index in posnode_list)
    cut_file.write('\n last sampled : ' + str(cut_value_list[ind_last]) )
    posnode_list=np.where(cut_list[ind_last] > 0)
    posnode_list=posnode_list[:][0].tolist()
    cut_file.writelines(", %d " % node_index  for node_index in posnode_list)

    # for cut_index in list(ind_increase):
    #     cut_file.write('\n' + str(cut_value_list[cut_index]) )
    #     posnode_list=np.where(cut_list[cut_index] > 0)
    #     posnode_list=posnode_list[:][0].tolist()
    #     # import pdb; pdb.set_trace()
    #     cut_file.writelines(", %d " % node_index  for node_index in posnode_list)
    cut_file.close()
    print('See file: ' + cut_file_name)   
    
    
    # end = time.time()
    # print('Calculation time for rotor solve', end - start)

    # start = time.time()
    # cut, cut_value = procedure_cut(nqs, config)
    # end = time.time()
    # print('Calculation time for procedure-cut from BMZ', end - start)

    # num_nodes=config['num_visible']
    # for ind in range(num_nodes):
    #     print("x_" + str(ind) + " = " + str(cut[ind]))

    # print("cut value = " + str(cut_value))
    return nqs


def vmc_solve_fast(config, nqs=None,logger=None, cut_value_reference=None):
    

    if config['debug']:
        from nqs_sho import NQS, propose_update
        from wavefunction_sho import local_energy, log_psi, log_psi_vars
    else:
        from nqs import NQS, propose_update
        from wavefunction import local_energy, log_psi, log_psi_vars

    if config['flag_initialize'] <=0:
        initial_step = 0
    else:
        initial_step = config['flag_initialize']
    if logger and nqs is not None:
        
        logger.log_scalar('b_norm', np.sqrt(np.sum(nqs.bs**2)))
        logger.log_scalar('c_norm', np.sqrt(np.sum(nqs.cs**2)))
        cut_local, cut_value_local = procedure_cut(nqs, config)
        
        if logger:
            logger.log_scalar('cut_value', cut_value_local)
            if cut_value_reference is not None:
                logger.log_scalar('improve_rate', cut_value_local/cut_value_reference)
        
        logger.write_step(initial_step)

    if nqs is None:
        nqs = NQS(config=config) # initialize nqs with the n and H from config file

    # start = time.time()
    time_sample=0.0
    time_optimize=0.0
    time_linear_operator=0.0

    

    for step in range(1, config['num_step']+1):
        start = time.time()
        operators, averages , nqs_history, start_index = metropolis_sampler_fast(step,
                                            local_energy,
                                            log_psi,
                                            log_psi_vars,
                                            nqs_init=nqs,
                                            config=config,
                                            propose_update=propose_update,
                                            logger=logger)
        end = time.time()
        time_sample += (end - start)
        start = time.time()
        
        mat_vec = lambda x: fisher_mat_vec_fast(x, operators, nqs_history[start_index:], averages[0], config['sr_reg'])
        end = time.time()
        time_linear_operator += end-start
        # print('time for constructing the operator ={:5.2e}'.format(time_linear_operator))
        nqs = nqs_history[-1]
        vars = stoch_reconfig_fast(step, nqs, mat_vec, averages , config, logger)
        # vars = stoch_reconfig_fast_debug(step, nqs, operators, averages ,nqs_history[start_index:], config, logger)
        nqs = NQS(config=config, state=nqs.state, vars=vars)
        end = time.time()
        time_optimize += (end - start)

        nqs_local = nqs_history[start_index]
        cut_local, cut_value_local = procedure_cut(nqs_local, config)
        cut_list=[cut_local]
        cut_value_list=[cut_value_local]
        for nqs_index in range(start_index + 1,len(nqs_history)):
            nqs_local = nqs_history[nqs_index]
            cut_local, cut_value_local = procedure_cut(nqs_local, config)
            cut_list.append(cut_local)
            cut_value_list.append(cut_value_local)
        ind_increase = np.argsort(np.array(cut_value_list))
        ind_best=ind_increase[-1]
        cut = cut_list[ind_best]
        cut_value = cut_value_list[ind_best]
        if logger:
            logger.log_scalar('cut_value', cut_value)
            if cut_value_reference is not None:
                logger.log_scalar('improve_rate', cut_value/cut_value_reference)
        
        logger.write_step(step+initial_step)
        # print('Success at step ' + str(step))

    start = time.time()
    nqs_local = nqs_history[start_index]
    cut_local, cut_value_local = procedure_cut(nqs_local, config)
    cut_list=[cut_local]
    cut_value_list=[cut_value_local]
    for nqs_index in range(start_index + 1,len(nqs_history)):
        nqs_local = nqs_history[nqs_index]
        cut_local, cut_value_local = procedure_cut(nqs_local, config)
        cut_list.append(cut_local)
        cut_value_list.append(cut_value_local)
    ind_increase = np.argsort(np.array(cut_value_list))
    ind_best=ind_increase[-1]
    cut = cut_list[ind_best]
    cut_value = cut_value_list[ind_best]
    # print_energy(config['num_step'], averages)
    end = time.time()
    time_generate_cut = end-start
    
    return cut, cut_value, time_sample, time_optimize, time_generate_cut, nqs_history[-1]


def vmc_solve_fast_precond(config, nqs=None,logger=None):
    

    if config['debug']:
        from nqs_sho import NQS, propose_update
        from wavefunction_sho import local_energy, log_psi, log_psi_vars
    else:
        from nqs import NQS, propose_update
        from wavefunction import local_energy, log_psi, log_psi_vars

    if nqs is None:
        nqs = NQS(config=config) # initialize nqs with the n and H from config file

    # start = time.time()
    time_sample=0.0
    time_optimize=0.0
    time_linear_operator=0.0
    for step in range(1, config['num_step']):
        start = time.time()
        operators, averages , nqs_history, start_index = metropolis_sampler_fast(step,
                                            local_energy,
                                            log_psi,
                                            log_psi_vars,
                                            nqs_init=nqs,
                                            config=config,
                                            propose_update=propose_update,
                                            logger=logger)

        end = time.time()
        time_sample += (end - start)
        start = time.time()
        # # dense reference for debugging
        # averages_copy =[ averages[0],mean_outdot(operators, nqs_history[start_index:]), averages[1],averages[2]]

        avec, vec_list = fisher_precond_vec( operators, nqs_history[start_index:], averages[0])
        # import pdb; pdb.set_trace()
        mat_vec = lambda x: fisher_mat_vec_fast_precond(x, avec, vec_list, averages[0], config['sr_reg'])
        # grads = averages[1] - averages[0] * averages[2]
        # mat_vec( np.multiply(avec,grads))
        end = time.time()
        time_linear_operator += end-start
        # print('time for constructing the operator ={:5.2e}'.format(time_linear_operator))
        nqs = nqs_history[-1]
        vars = stoch_reconfig_fast_precond(step, nqs, mat_vec, averages , avec, averages_copy, config,logger)
       
        nqs = NQS(config=config, state=nqs.state, vars=vars)
        end = time.time()
        time_optimize += (end - start)
        # print('Success at step ' + str(step))
        logger.write_step(step)


    
    nqs_local = nqs_history[start_index]
    cut_local, cut_value_local = procedure_cut(nqs_local, config)
    cut_list=[cut_local]
    cut_value_list=[cut_value_local]
    for nqs_index in range(start_index + 1,len(nqs_history)):
        nqs_local = nqs_history[nqs_index]
        cut_local, cut_value_local = procedure_cut(nqs_local, config)
        cut_list.append(cut_local)
        cut_value_list.append(cut_value_local)
        #import pdb; pdb.set_trace()
    
    ind_increase = np.argsort(np.array(cut_value_list))
    ind_best=ind_increase[-1]
    cut = cut_list[ind_best]
    cut_value = cut_value_list[ind_best]
    print_energy(config['num_step'], averages)
    return cut, cut_value

def fisher_mat_vec_fast(grads, grad_vec_operator, nqs_list, O_av, config_sr_reg):
    vec_list = [grad_vec_operator(nqs_tmp) for nqs_tmp in nqs_list ]
    OO_av_dot_grads = average([np.dot(vec,grads) * vec for vec in vec_list])
    return  OO_av_dot_grads - np.dot(O_av,grads) * O_av + config_sr_reg * grads


def fisher_precond_vec( grad_vec_operator, nqs_list, O_av):
    nsamp=len(nqs_list)
    vec_list = [grad_vec_operator(nqs_tmp) for nqs_tmp in nqs_list ]
    aux_vec = np.multiply(vec_list , vec_list) 
    aux_mat = [aux_vec[ind]/nsamp for ind in range(nsamp)]
    aux_mat.append( - np.multiply(O_av , O_av))
    avec=np.sum(aux_mat,axis=0)
    avec.flags.writeable = True
    avec[avec<1e-10] = 1.0
    return np.sqrt(avec), vec_list
def fisher_mat_vec_fast_precond(grads, avec, vec_list, O_av, config_sr_reg):
    grads2 = np.divide(grads, avec)

    OO_av_dot_grads = average([np.dot(vec,grads2) * vec for vec in vec_list]) - np.dot(O_av,grads2) * O_av 
    
    return   np.divide(OO_av_dot_grads, avec) + config_sr_reg * grads

def mean_outdot(op, history):
    return average([np.outer(op(nqs),op(nqs)) for nqs in history])

if __name__ == "__main__":
    main()
