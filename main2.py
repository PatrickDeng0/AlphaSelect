import sys, os

size, select, start_bar, market, activation, mode, optimizer, GPU = sys.argv[1:]

log_path = 'logs/train_logs/'
os.makedirs(log_path, exist_ok=True)

log_name = '%s_%s_%s_%s_%s_train.log' % (size, select, start_bar, mode, optimizer)

cmd_line = '''\
python3 train2.py %s %s %s %s %s %s %s %s > %s%s\
''' % (size, select, start_bar, market, activation, mode, optimizer, GPU, log_path, log_name)
print(cmd_line)
os.system(cmd_line)

# nohup python3 main2.py 2 0 15 e r x r 0 > /dev/null 2>&1 &
# nohup python3 main2.py 2 0 15 e r x n 0 > /dev/null 2>&1 &
# nohup python3 main2.py 2 0 15 e r x a 0 > /dev/null 2>&1 &

# nohup python3 main2.py 3 1 2 e r y r 0 > /dev/null 2>&1 &
# nohup python3 main2.py 3 1 2 e r y n 0 > /dev/null 2>&1 &
# nohup python3 main2.py 3 1 2 e r y a 0 > /dev/null 2>&1 &

# nohup python3 main2.py 3 1 2 e r c r 0 > /dev/null 2>&1 &
# nohup python3 main2.py 3 1 2 e r c n 0 > /dev/null 2>&1 &
# nohup python3 main2.py 3 1 2 e r c a 0 > /dev/null 2>&1 &

# nohup python3 main2.py 5 0 15 e r c r 0 > /dev/null 2>&1 &
# nohup python3 main2.py 5 0 15 e r c n 0 > /dev/null 2>&1 &
# nohup python3 main2.py 5 0 15 e r c a 0 > /dev/null 2>&1 &
