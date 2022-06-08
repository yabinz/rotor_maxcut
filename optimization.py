import numpy as np
from scipy.sparse import linalg as sp_linalg
import time
from sampler import mean
from util import average

# for debugging
import copy


def gradients(step, averages, logger=None):
    O_av, OO_av, EO_av, E_av = averages
    grads = EO_av - O_av * E_av
    fisher = OO_av - np.outer(O_av, O_av)
    # if logger:
    #     print('average energy on step {}: {}'.format(step, E_av))
    #     logger.log_scalar('E_av', E_av, step)
    #     logger.log_scalar('grads_norm', np.linalg.norm(grads), step)
    if logger:
        logger.log_scalar('energy_avg', E_av)
        logger.log_scalar('grad_norm', np.linalg.norm(grads))
        # logger.log_scalar('b_norm', np.sqrt(np.sum(nqs.bs**2)))
        # logger.log_scalar('c_norm', np.sqrt(np.sum(nqs.cs**2)))
    return grads, fisher


def stoch_reconfig(step, nqs, averages, config, logger=None, TOL=1e-10, MaxIter =5e+3):
    grads, fisher = gradients(step, averages, logger)
    fisher_reg = fisher + config['sr_reg'] * np.eye(len(nqs.vars))
    # # direct solve 
    # lin_sol =  np.linalg.solve(fisher_reg, grads)

    # # iterative solve
    num_iters = 0
    def callback(xk):
         nonlocal num_iters
         num_iters+=1
    lin_sol, exit_info =  sp_linalg.minres(fisher_reg, grads,tol=TOL,maxiter=MaxIter,callback=callback)

    print('output_flag ={}'.format(exit_info))
    print('num of iterations = {}'.format(num_iters))

    vars_new = nqs.vars - config['lr'] * lin_sol
    if logger:
        # logger.log_scalar('energy_avg', E_av)
        # logger.log_scalar('grad_norm', np.linalg.norm(grads))
        logger.log_scalar('b_norm', np.sqrt(np.sum(nqs.bs**2)))
        logger.log_scalar('c_norm', np.sqrt(np.sum(nqs.cs**2)))
    return vars_new

def stoch_reconfig_precond(step, nqs, averages, config, logger=None, TOL=1e-10, MaxIter =5e+3):
    grads, fisher = gradients(step, averages, logger)
    # fisher_reg = fisher + config['sr_reg'] * np.eye(len(nqs.vars))
    # lin_sol =  np.linalg.solve(fisher_reg, grads)
    avec= np.ones((len(nqs.vars)))
    for ind_i in range(len(nqs.vars)):
        if fisher[ind_i][ind_i] >1e-10:
            avec[ind_i] = np.sqrt(fisher[ind_i][ind_i])
    fisher_precond = copy.deepcopy(fisher)
    for ind_i in range(len(nqs.vars)):
        for ind_j in range(len(nqs.vars)):
            fisher_precond[ind_i][ind_j] = fisher_precond[ind_i][ind_j]/avec[ind_i]/avec[ind_j]
    grads_precond = copy.deepcopy(grads)
    for ind_j in range(len(nqs.vars)):
        grads_precond[ind_j] = grads_precond[ind_j]/avec[ind_j]
    fisher_precond_reg = fisher_precond + config['sr_reg'] * np.eye(len(nqs.vars))
    num_iters = 0
    def callback(xk):
         nonlocal num_iters
         num_iters+=1
    lin_sol_precond, exit_info =  sp_linalg.minres(fisher_precond_reg, grads_precond,tol=TOL,maxiter=MaxIter,callback=callback)
    lin_sol = copy.deepcopy(lin_sol_precond)
    for ind_j in range(len(nqs.vars)):
            lin_sol[ind_j] = lin_sol[ind_j]/avec[ind_j]

    print('output_flag ={}'.format(exit_info))
    print('num of iterations = {}'.format(num_iters))

    vars_new = nqs.vars - config['lr'] * lin_sol
    if logger:
        # logger.log_scalar('energy_avg', E_av)
        # logger.log_scalar('grad_norm', np.linalg.norm(grads))
        logger.log_scalar('b_norm', np.sqrt(np.sum(nqs.bs**2)))
        logger.log_scalar('c_norm', np.sqrt(np.sum(nqs.cs**2)))
    if len(np.where(np.isnan(vars_new))[0])>0:
            import pdb; pdb.set_trace()
    return vars_new



def stoch_reconfig_fast(step, nqs, mat_vec, averages ,config,logger=None, TOL=1e-10, MaxIter =5e+3):
    
    # start = time.time()
    O_av,  EO_av, E_av = averages
    grads = EO_av - O_av * E_av
    # time_grads=time.time() - start
    ndim = len(grads)
    
    # start = time.time()
    fisher_LinearOperator = sp_linalg.LinearOperator((ndim,ndim), matvec=mat_vec)
    # time_fisher_op=time.time() - start
    # start = time.time()
    # print('tolerance set to {:5.2e}'.format(TOL))
    # print('solver type = {}'.format(scipy_solver_type))
    num_iters = 0
    def callback(xk):
         nonlocal num_iters
         num_iters+=1
    # if scipy_solver_type=="gmres":
    #     lin_sys_sol, exit_info = sp_linalg.gmres(fisher_LinearOperator, grads,tol=TOL,maxiter=MaxIter,callback=callback)
    # elif scipy_solver_type=="cg":
    #     lin_sys_sol, exit_info = sp_linalg.cg(fisher_LinearOperator, grads,tol=TOL,maxiter=MaxIter,callback=callback)
    # else: 
    #     lin_sys_sol, exit_info = sp_linalg.minres(fisher_LinearOperator, grads,tol=TOL,maxiter=MaxIter,callback=callback)
    lin_sys_sol, exit_info = sp_linalg.minres(fisher_LinearOperator, grads,tol=TOL,maxiter=MaxIter,callback=callback)

    # print('output_flag ={}'.format(exit_info))
    # print('num of iterations = {}'.format(num_iters))
    # print('relative residue={:5.2e}'.format(np.linalg.norm(fisher_operator.matvec(lin_sys_sol)-grads)/np.linalg.norm(grads)))

    # time_cg = time.time() - start
    # start = time.time()
    vars_new = nqs.vars - config['lr'] * lin_sys_sol
    # time_step = time.time() - start
    # end1 = time.time()
    # print('times: grads ={:5.2e}; operator={:5.2e}; iterative solve={:5.2e}; fwd step={:5.2e}.'.format( time_grads, time_fisher_op, time_cg, time_step))
    # print('total times = {:5.2e}.'.format(time_grads + time_fisher_op + time_cg + time_step))
    
    if logger:
        logger.log_scalar('minres_info',exit_info)
        logger.log_scalar('minres_iters', num_iters)
        logger.log_scalar('minres_absres', np.linalg.norm(fisher_LinearOperator.matvec(lin_sys_sol)-grads))
        logger.log_scalar('energy_avg', E_av)
        logger.log_scalar('grad_norm', np.linalg.norm(grads))
        logger.log_scalar('b_norm', np.sqrt(np.sum(nqs.bs**2)))
        logger.log_scalar('c_norm', np.sqrt(np.sum(nqs.cs**2)))
    return vars_new

def stoch_reconfig_fast_precond(step, nqs, mat_vec, averages, avec ,averages_copy, config,TOL=1e-12, MaxIter ="minres", 
logger=None, operators=None, nqs_history=None):
    
    # start = time.time()
    O_av,  EO_av, E_av = averages
    grads = EO_av - O_av * E_av
    # time_grads=time.time() - start
    ndim = len(grads)
    
    # start = time.time()
    fisher_LinearOperator = sp_linalg.LinearOperator((ndim,ndim), matvec=mat_vec)
    # time_fisher_op=time.time() - start
    # start = time.time()
    # print('tolerance set to {:5.2e}'.format(TOL))
    # print('solver type = {}'.format(scipy_solver_type))
    
    rhs_precond=np.divide(grads,avec)
    num_iters = 0
    # if scipy_solver_type=="gmres":
    #     lin_sys_sol_precond, exit_info = sp_linalg.gmres(fisher_operator, rhs_precond,tol=TOL,maxiter=MaxIter,callback=callback)
    # elif scipy_solver_type=="cg":
    #     lin_sys_sol_precond, exit_info = sp_linalg.cg(fisher_operator, rhs_precond,tol=TOL,maxiter=MaxIter,callback=callback)
    # else: 
    #     lin_sys_sol_precond, exit_info = sp_linalg.minres(fisher_operator, rhs_precond,tol=TOL,maxiter=MaxIter,callback=callback)
    
    lin_sys_sol_precond, exit_info = sp_linalg.minres(fisher_operator, rhs_precond,tol=TOL,maxiter=MaxIter,callback=callback)
    # lin_sys_sol_precond, cg_info = sp_linalg.minres(fisher_operator, np.divide(grads,avec),tol=1e-12)
    print('output_flag ={}'.format(cg_info))
    print('num of iterations = {}'.format(num_iters))
    # print('relative residue={:5.2e}'.format(np.linalg.norm(fisher_operator.matvec(lin_sys_sol_precond)-rhs_precond)/np.linalg.norm(rhs_precond)))
    
    # print('rel res with dense matrix = {:5.2e}'.format(np.linalg.norm(np.dot(fisher_precond_reg,lin_sys_sol_precond)-grads_precond)/np.linalg.norm(grads_precond)))
    # if np.isnan( np.linalg.norm(fisher_operator.matvec(lin_sys_sol_precond)-rhs_precond)/np.linalg.norm(rhs_precond) ):
    #     import pdb; pdb.set_trace()
    # lin_sys_sol = np.divide(lin_sys_sol_precond, avec)
    # # lin_sys_sol=np.linalg.solve(fisher_operator, grads) # will not work!
    # time_cg = time.time() - start
    # start = time.time()
    # vars_new = nqs.vars - config['lr'] * lin_sys_sol
    # time_step = time.time() - start
    # # end1 = time.time()
    # print('times: grads ={}; operator={}; iterative solve={}; fwd step={}.'.format( time_grads, time_fisher_op, time_cg, time_step))
    # print('total times = {}.'.format(time_grads + time_fisher_op + time_cg + time_step))
    # print('output_flag ={}'.format(cg_info))
    
    # # # # error checking for debugging
    # # tensor_vec_operator =  copy.deepcopy(operators)
    # # def tensor_from_vec(nqs_tmp):
    # #     return np.outer(tensor_vec_operator(nqs_tmp), tensor_vec_operator(nqs_tmp))
    
    # # # # import pdb; pdb.set_trace()
    # # operators_tmp = tensor_from_vec
    # # O_av, EO_av, E_av = averages
    # # OO_av = mean(operators_tmp, nqs_history) 
    # # fisher = OO_av - np.outer(O_av, O_av)
    # # fisher_reg = fisher + config['sr_reg'] * np.eye(len(nqs.vars))
    # # print(' sp_linalg.cg: abs res  = {}, rel res = {}'.format(np.linalg.norm(np.dot(fisher_reg,lin_sys_sol) - grads),
    # #                                                 np.linalg.norm(np.dot(fisher_reg,lin_sys_sol) - grads)/np.linalg.norm(grads)))
    # ## new lodger
    if logger:
        logger.log_scalar('energy_avg', averages_copy[-1])
        logger.log_scalar('grad_norm', np.linalg.norm(grads_ref))
        logger.log_scalar('b_norm', np.sqrt(np.sum(nqs.bs**2)))
        logger.log_scalar('c_norm', np.sqrt(np.sum(nqs.cs**2)))
    
    if len(np.where(np.isnan(vars_new_ref))[0])>0:
            import pdb; pdb.set_trace()
    return vars_new_ref
def stoch_reconfig_fast_debug(step, nqs, grad_vec_operator, averages ,nqs_list,config, logger=None):
    # grads, fisher = gradients(step, averages, logger)
    # start1 = time.time()
    # O_av, EO_av, E_av = averages
    # grads = EO_av - O_av * E_av
    start = time.time()
    grads = averages[1] - averages[0] * averages[2]
    time_grads=time.time() - start
    ndim = len(grads)
    
    # vars_new = nqs.vars - config['lr'] * np.linalg.solve(fisher_reg, grads)
    # mat_vec = lambda x: fisher_mat_vec_fast(x, operators, nqs_history, averages[0], config['sr_reg'])
    # mat_vec = lambda x: fisher_mat_vec(x, averages[1], nqs_history[start_index:], O_av, config['sr_reg'])
    start = time.time()
    vec_list = [grad_vec_operator(nqs_tmp)/ndim for nqs_tmp in nqs_list ]
    vec_list.append(-averages[0])
    nsamp=len(nqs_list)
    vec_list2 = [vec_list[isamp]*ndim for isamp in range(nsamp)]
    vec_list2.append(averages[0])
    umat=np.transpose(np.array(vec_list))
    vmat=np.array(vec_list2)
    
    time_fisher_op=time.time() - start
    start = time.time()
    # lin_sys_sol, cg_info = sp_linalg.cg(fisher_operator, grads,tol=1e-12)
    # lin_sys_sol=np.linalg.solve(fisher_operator, grads) # will not work!
    vec_y, res1, rank1, sv1 = np.linalg.lstsq(umat, grads)
    
    # import pdb; pdb.set_trace()
    # lin_sys_sol, res2, rank2, sv2 = np.linalg.lstsq(vmat, vec_y)
    # lin_sys_sol, res2, rank2, sv2 = np.linalg.lstsq(np.dot(np.transpose(vmat),vmat), 
    #     np.dot(np.transpose(vmat),vec_y))
    # tmpmat=np.dot(np.transpose(vmat),vmat)
    # tmpmat_reg=tmpmat +  config['sr_reg'] * np.eye(len(grads))
    # lin_sys_sol, res2, rank2, sv2 = np.linalg.lstsq(tmpmat_reg, 
    #     np.dot(np.transpose(vmat),vec_y))
    # # import pdb; pdb.set_trace()
    # print('unregulated mat-vec residue: {}'.format(np.linalg.norm(np.dot(tmpmat,lin_sys_sol)-np.dot(np.transpose(vmat),vec_y))))
    # print('  regulated mat-vec residue: {}'.format(np.linalg.norm(np.dot(tmpmat_reg,lin_sys_sol)-np.dot(np.transpose(vmat),vec_y))))
    # # svd approach
    uu,ss,vvh = np.linalg.svd(vmat)
    tmp=np.where(ss>1e-13)
    uu_trunc=uu[:, tmp[0]]
    ss_trunc=ss[ tmp[0]]
    vvh_trunc=vvh[tmp[0],:]
    vec_y_range=np.dot(uu_trunc,  (np.dot(vec_y,uu_trunc)))

    import pdb; pdb.set_trace()

    # vumat=np.dot(vmat,umat)
    # ddim=len(vmat)
    
    # woodsol= np.linalg.solve(vumat/config['sr_reg'] + np.eye(ddim), np.dot(vmat,grads)/config['sr_reg'])
    # lin_sys_sol= (grads - np.dot(umat,woodsol))/config['sr_reg']
    time_cg = time.time() - start
    # uvmat_reg=np.dot(umat,vmat) + config['sr_reg'] * np.eye(len(nqs.vars))
    # print('local residue: {}'.format(np.linalg.norm(np.dot(uvmat_reg,lin_sys_sol)-grads)))

    print('local residue: {}, and {}'.format(np.linalg.norm(np.dot(umat,vec_y)-grads), np.linalg.norm(np.dot(vmat,lin_sys_sol)-vec_y)))
    # import pdb; pdb.set_trace()
    start = time.time()
    vars_new = nqs.vars - config['lr'] * lin_sys_sol
    time_step = time.time() - start
    # end1 = time.time()
    print('times: grads ={}; operator={}; 2 lstsq solve={}; fwd step={}.'.format( time_grads, time_fisher_op, time_cg, time_step))
    print('total times = {}.'.format(time_grads + time_fisher_op + time_cg + time_step))
    # print(cg_info)
    
    # # error checking for debugging
    tensor_vec_operator =  copy.deepcopy(grad_vec_operator)
    def tensor_from_vec(nqs_tmp):
        return np.outer(tensor_vec_operator(nqs_tmp), tensor_vec_operator(nqs_tmp))
    
    # # import pdb; pdb.set_trace()
    operators_tmp = tensor_from_vec
    O_av, EO_av, E_av = averages
    OO_av = mean(operators_tmp, nqs_list) 
    fisher = OO_av - np.outer(O_av, O_av)
    fisher_reg = fisher + config['sr_reg'] * np.eye(len(nqs.vars))
    print(' 2 lstsq solves: abs res  = {}, rel res = {}'.format(np.linalg.norm(np.dot(fisher_reg,lin_sys_sol) - grads),
                                                    np.linalg.norm(np.dot(fisher_reg,lin_sys_sol) - grads)/np.linalg.norm(grads)))
    # print('               time = {}'. format(end1 - start1))

    # start2 = time.time()
    # # averages = [mean(op, nqs_history[start_index:]) for op in operators]
    # # averages = [mean(op, nqs_history) for op in operators_tmp]
    # OO_av = mean(operators_tmp, nqs_history) 
    # grads, fisher = gradients(step, [O_av,OO_av, EO_av, E_av], logger)
    # fisher_reg = fisher + config['sr_reg'] * np.eye(len(nqs.vars))
    # ref_sol = np.linalg.solve(fisher_reg, grads)
    # end2 = time.time()
    # print(' np.linalg.solve: abs res  = {}, rel res = {}'.format(np.linalg.norm(np.dot(fisher_reg,ref_sol) - grads),
    #                                                 np.linalg.norm(np.dot(fisher_reg,ref_sol) - grads)/np.linalg.norm(grads)))
    # print('                  time = {}'. format(end2 - start2))

    # import pdb; pdb.set_trace()
    return vars_new
def fisher_mat_vec(grads, OO_av, O_av, config_sr_reg):
    return  np.dot(OO_av,grads) -  np.dot(O_av,grads) * O_av  + config_sr_reg * grads

def fisher_mat_vec_fast(grads, grad_vec_operator, nqs_list, O_av, config_sr_reg):
    vec_list = [grad_vec_operator(nqs_tmp) for nqs_tmp in nqs_list ]
    OO_av_dot_grads = average([np.dot(vec,grads) * vec for vec in vec_list])
    return  OO_av_dot_grads - np.dot(O_av,grads) * O_av + config_sr_reg * grads

# # added this new function for printing
# def print_energy(step, averages):
#     O_av, OO_av, EO_av, E_av = averages
#     grads = EO_av - O_av * E_av
    
#     print('average energy on step {}: {}'.format(step, E_av))
#     print('grads norm on step {}: {}'.format(step, np.linalg.norm(grads)))
    