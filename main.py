import sys, os


size = sys.argv[1]
batch_size = sys.argv[2]
init_lr = sys.argv[3]
GPU = sys.argv[4]

log_path = 'logs/'
os.makedirs(log_path, exist_ok=True)
log_path = log_path + '_'.join(sys.argv[1:4]) + '/'
os.makedirs(log_path, exist_ok=True)

cmd_line = '''\
python train.py %s %s %s %s %s > %s%s_train.log\
''' % (size, batch_size, init_lr, 'lstm', GPU, log_path, 'lstm')
print(cmd_line)
os.system(cmd_line)

cmd_line = '''\
python train.py %s %s %s %s %s > %s%s_train.log\
''' % (size, batch_size, init_lr, 'cnn', GPU, log_path, 'cnn')
print(cmd_line)
os.system(cmd_line)
