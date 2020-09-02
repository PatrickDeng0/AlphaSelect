import sys, os

size, init_lr, select, start_bar, activations, modes, GPU = sys.argv[1:]

log_path = 'logs/'
os.makedirs(log_path, exist_ok=True)
log_path = log_path + '_'.join([size, init_lr, select, start_bar]) + '/'
os.makedirs(log_path, exist_ok=True)

cmd_line = '''\
python3 train.py %s %s %s %s %s %s %s > %strain.log\
''' % (size, init_lr, select, start_bar, activations, modes, GPU, log_path)
print(cmd_line)
os.system(cmd_line)

# nohup python3 main.py 2 3 0 15 rt btlc 0 > /dev/null 2>&1 &
