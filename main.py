import sys, os, pickle


size = sys.argv[1]
mini = sys.argv[2]
batch_size = sys.argv[3]
init_lr = sys.argv[4]
mode = sys.argv[5]
GPU = sys.argv[6]

log_path = 'logs/'
os.makedirs(log_path, exist_ok=True)
log_path = log_path + '_'.join(sys.argv[1:5]) + '/'
os.makedirs(log_path, exist_ok=True)

cmd_line = '''\
python train.py %s %s %s %s %s %s > %s%s_train.log\
''' % (size, mini, batch_size, init_lr, mode, GPU, log_path, mode)
print(cmd_line)
os.system(cmd_line)
