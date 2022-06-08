import argparse
import os


import time
from datetime import datetime
# from numpy.random import seed

# from logger_vmc import Logger
from new_logger import Logger
from util import read_config

from rotorvmc import vmc_solve,vmc_solve_fast
from cut import procedure_cut

# seed(1)

def main():
    # configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yaml')
    args = parser.parse_args()
    config = read_config(args.config_path)

    # # logging
    # logdir_suffix = datetime.now().replace(microsecond=0).isoformat()
    # if config['logdir_prefix'] != '':
    #     logdir_name = config['logdir_prefix'] + '_' + logdir_suffix
    # else:
    #     logdir_name = logdir_suffix
    # logdir = os.path.join(config['logdir_root'], logdir_name)
    # logger = Logger(logdir)
    
    ## new lodging 
    os.makedirs(config['logdir'], exist_ok=True)

    logger = Logger(config['logdir'])
    logger.set_variables(['acceptance_rate', 'b_norm', 'c_norm', 
    'minres_info', 'minres_iters','minres_absres',
    'energy_avg', 'energy_std', 'grad_norm','cut_value'])
    logger.write_header(config)

    # start = time.time()
    # nqs = vmc_solve(config, nqs=None,logger=logger)
    # end = time.time()

    # print('Calculation time for rotor solve', end - start)

    # start = time.time()
    # cut, cut_value = procedure_cut(nqs, config)
    # end = time.time()
    # print('Calculation time for procedure-cut from BMZ', end - start)
    # num_nodes=config['num_visible']
    # for ind in range(num_nodes):
    #     if cut[ind]>0:
    #         print("x_" + str(ind) + " = " + str(cut[ind]))


    # print("cut value = " + str(cut_value))

    cut, cut_value, time_sample, time_optimize, time_generate_cut, nqs_final = vmc_solve_fast(config, nqs=None,logger=logger)
    num_nodes=config['num_visible']
    for ind in range(num_nodes):
        if cut[ind]>0:
            print("x_" + str(ind) + " = " + str(cut[ind]))


    # print("cut value = " + str(cut_value))
    print("Rotor solver: cut value = " + str(cut_value))
    print('              time_sample={:5.2e}, time_optimize ={:5.2e}'.format(time_sample,time_optimize))
    print('              time_generate_final_cut={:5.2e}'.format(time_generate_cut))
    time_total = time_sample + time_optimize + time_generate_cut
    print('              Total time ={:5.2e}'.format(time_total))


if __name__ == "__main__":
    main()
