__author__ = "Saibal De"

from datetime import datetime
import time
import numpy as np
# np.random.seed(int(time.time())) # manually seeds according to time


class Logger(object):
    """Logging in plaintext."""

    def __init__(self, log_dir, config=None):
        """Create a log file inside log_dir."""
        if config is None:
            self.file_name = log_dir + "/" + datetime.now().replace(microsecond=0).isoformat() + "rd" + str(int(np.random.random()*1000000)) + ".csv"
        else:
            n=config["num_visible"]
            ne=len(config['edges'])
            self.file_name = log_dir + "/" + datetime.now().replace(microsecond=0).isoformat() + "_N{:d}".format(n) + "E{:d}".format(ne) +".csv"

        self.var_names = None
        self.var_values = None

    def set_variables(self, var_names):
        self.var_names = var_names
        self.var_values = {var: 0.0 for var in var_names}

    def log_scalar(self, var, value):
        self.var_values[var] = value

    def write_header(self, config=None):
        file_object = open(self.file_name, "w")
        if config is not None:
            graphparameter_log = "num_visible, num_hidden, num_edge, flag_initialization" + "\n"
            vmcparameter_log = "num_step, lr, sr_reg, metro_steps, bump_size, warm_steps," + "\n"
            file_object.write(graphparameter_log)
            line =  "{:e}".format(config['num_visible']) + ", {:e}".format(config['num_hidden']) +", {:e}".format(len(config['edges']))+ ", {:d}".format(config['flag_initialize']) + ",\n"
            file_object.write(line)
            file_object.write(vmcparameter_log)
            line =  "{:e}".format(config['num_step']) + ", {:e}".format(config['lr']) + ", {:e}".format(config['sr_reg']) + ", {:e}".format(config['metropolis_steps']) + ", {:e}".format(config['bump_size']) + ", {:e}".format(config['warm_steps']) +", \n" + "\n"
            file_object.write(line)

        header = "timestamp,step"
        for var in self.var_names:
            header += "," + var
        header += "\n"

        
        file_object.write(header)
        file_object.close()

    def write_step(self, step):
        line = datetime.now().isoformat() + "," + str(step)
        for var in self.var_names:
            line += "," + "{:e}".format(self.var_values[var])
        line += "\n"

        file_object = open(self.file_name, "a")
        file_object.write(line)
        file_object.close()
