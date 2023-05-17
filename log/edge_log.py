import time
import os

class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    def create_log_file(self, log_name):
        log_file = '{}.log'.format(log_name)
        final_log_file = os.path.join(self.log_dir, log_file)
        open(final_log_file, 'w').close()
    
    def write_log(self, log_name, log_str):
        log_file = '{}.log'.format(log_name)
        final_log_file = os.path.join(self.log_dir, log_file)
        with open(final_log_file, 'a+') as f:
            f.write('{} {} \n'.format(self.time_str, log_str))