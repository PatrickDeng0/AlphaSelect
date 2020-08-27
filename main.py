import sys, os

size = sys.argv[1]
batch_size = sys.argv[2]
init_lr = sys.argv[3]
intra = sys.argv[4]
start_bar = sys.argv[5]
GPU = sys.argv[6]

# LSTM is trained only if appointed
LSTM = True
if len(sys.argv) > 7 and sys.argv[7] == '0':
    LSTM = False

log_path = 'logs/'
os.makedirs(log_path, exist_ok=True)
log_path = log_path + '_'.join(sys.argv[1:6]) + '/'
os.makedirs(log_path, exist_ok=True)

cmd_line = '''\
python3 train.py %s %s %s %s %s %s %s > %s%s_train.log\
''' % (size, batch_size, init_lr, intra, start_bar, 'cnn', GPU, log_path, 'cnn')
print(cmd_line)
os.system(cmd_line)

if LSTM:
    cmd_line = '''\
    python3 train.py %s %s %s %s %s %s %s > %s%s_train.log\
    ''' % (size, batch_size, init_lr, intra, start_bar, 'lstm', GPU, log_path, 'lstm')
    print(cmd_line)
    os.system(cmd_line)

# nohup python3 main.py 12 10000 2 0 2 0 0 > /dev/null 2>&1 &
