import sys, os

size = sys.argv[1]
init_lr = sys.argv[2]
intra = sys.argv[3]
start_bar = sys.argv[4]
GPU = sys.argv[5]

# LSTM is trained only if appointed
LSTM = True
if len(sys.argv) > 6 and sys.argv[6] == '0':
    LSTM = False

log_path = 'logs/'
os.makedirs(log_path, exist_ok=True)
log_path = log_path + '_'.join(sys.argv[1:6]) + '/'
os.makedirs(log_path, exist_ok=True)

cmd_line = '''\
python3 train.py %s %s %s %s %s %s %s > %s%s_train.log\
''' % (size, init_lr, intra, start_bar, 's', 'cnn', GPU, log_path, 'cnn')
print(cmd_line)
os.system(cmd_line)

if LSTM:
    cmd_line = '''\
    python3 train.py %s %s %s %s %s %s %s > %s%s_train.log\
    ''' % (size, init_lr, intra, start_bar, 's', 'bilstm', GPU, log_path, 'lstm')
    print(cmd_line)
    os.system(cmd_line)

# nohup python3 main.py 10 3 1 2 0 1 > /dev/null 2>&1 &
# nohup python3 main.py 4 3 0 15 0 1 > /dev/null 2>&1 &
