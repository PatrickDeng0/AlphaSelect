import pandas as pd
import pickle, os, re


def collect(df_dict):
    mode_dict = {'0':'interday', '1':'intraday'}
    act_dict = {'r':'relu', 't':'tanh', 's':'sigmoid'}

    hy_params = os.listdir('logs/')
    for hy_param in hy_params:
        hypers = hy_param.split('_')
        if len(hypers) < 4:
            continue

        act_opti_hy_params = os.listdir('logs/'+hy_param+'/')
        for act_opti_hy_param in act_opti_hy_params:
            if len(act_opti_hy_param.split('.')) > 1:
                continue

            items = os.listdir('logs/' + hy_param + '/' + act_opti_hy_param + '/')
            activation, optimizer = act_opti_hy_param.split('_')[0], act_opti_hy_param.split('_')[1]
            for item in items:
                if item.endswith('.pkl'):
                    model_name = item.split('_')[0]
                    with open('logs/' + hy_param + '/' + act_opti_hy_param + '/' + item, 'rb') as file:
                        history, test_loss, test_metrics = pickle.load(file)
                    history_df = pd.DataFrame(history)
                    insert = [int(hypers[0]), mode_dict[hypers[1]], hypers[2],hypers[3], activation, optimizer,
                              history_df['IC'].values[-1], history_df['val_IC'].values[-1], test_metrics]
                    df_dict[model_name].loc[len(df_dict[model_name])] = insert
    return df_dict


if __name__ == '__main__':
    df_dict = {}
    models = ['x', 'y', 'cnn', 'lstm', 'bilstm', 'tcn']
    for model in models:
        df_dict[model] = pd.DataFrame(columns=['size', 'mode', 'bar', 'market', 'activation', 'optimizer',
                                               'train', 'valid', 'test'])
    result = collect(df_dict)
    for model, df in df_dict.items():
        df = df.sort_values(by=['mode', 'market', 'size', 'activation', 'optimizer'], ascending=True)
        df.index = range(len(df))
        df_dict[model] = df

    write = pd.ExcelWriter('logs/output.xls')
    for model, df in df_dict.items():
        df.to_excel(excel_writer=write, sheet_name=model, header=True, encoding="utf-8", index=False)
    write.save()
    write.close()
