import pickle


sizes = ['2','4','6','8','10']
for size in sizes:
    print('For size', size)
    with open('data/size' + size + '.pkl', 'rb') as file:
        tickers, train_date, valid_date, test_date, train_data, valid_data, test_data = pickle.load(file)
    print('Train dates:', train_date[0], train_date[-1])
    print('Valid dates:', valid_date[0], valid_date[-1])
    print('Test dates:', test_date[0], test_date[-1])
    print('=================================================')
