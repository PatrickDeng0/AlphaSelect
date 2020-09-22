import sys, os

size, select, start_bar, markets, activations, modes, optimizers, GPU = sys.argv[1:]

log_path = 'logs_3/train_logs/'
os.makedirs(log_path, exist_ok=True)

log_name = '%s_%s_%s_%s_train.log' % (size, select, start_bar, modes)

cmd_line = '''\
python3 train3.py %s %s %s %s %s %s %s %s > %s%s\
''' % (size, select, start_bar, markets, activations, modes, optimizers, GPU, log_path, log_name)
print(cmd_line)
os.system(cmd_line)

# nohup python3 main3.py 4 0 15 e r x a 0 > /dev/null 2>&1 &
# nohup python3 main3.py 7 1 2 e r y r 0 > /dev/null 2>&1 &

# nohup python3 main3.py 8 0 15 e r c r 0 > /dev/null 2>&1 &
# nohup python3 main3.py 5 1 2 e r c r 0 > /dev/null 2>&1 &

# nohup python3 main3.py 3 0 15 e t l r 0 > /dev/null 2>&1 &
# nohup python3 main3.py 5 1 2 e r l r 0 > /dev/null 2>&1 &

# aipaas aistart -u dong -p qwer1234 --image=test --gpu=1 --pod_name=test4
