import sys, os

size, select, start_bar, markets, activations, modes, GPU = sys.argv[1:]

log_path = 'logs/train_logs/'
os.makedirs(log_path, exist_ok=True)

log_name = '%s_%s_%s_train.log' % (size, select, start_bar)

cmd_line = '''\
python3 train.py %s %s %s %s %s %s %s > %s%s\
''' % (size, select, start_bar, markets, activations, modes, GPU, log_path, log_name)
print(cmd_line)
os.system(cmd_line)

# nohup python3 main.py 5 0 15 e r xc 0 > /dev/null 2>&1 &
# nohup python3 main.py 3 1 2 e r yc 0 > /dev/null 2>&1 &
# aipaas aistart -u dong -p qwer1234 --image=test --gpu=1 --pod_name=test4
