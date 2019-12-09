
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


# get file path from the
def get_file_paths(directory):
    file_path = []
    file_name = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_path.append(filepath)
            file_name.append(filename)
    return file_path, file_name


if __name__ == '__main__':

    # read data
    root_path = 'dataport/'

    # save_root_path = 'processed_data/'
    sample_path, sample_name = get_file_paths(root_path)
    sample_path.sort()
    metadata = pd.read_csv(sample_path[0])
    df0 = pd.read_csv(sample_path[0])

    # df1 = pd.read_csv(sample_path[1])
    # time_tran = pd.to_datetime(df0['DATA_TIME'])
    # df0['DATA_TIME'] = time_tran.apply(lambda x: x.time())
    # df_combine = df0[['CONSNO', 'DATA_TIME', 'PAP_R']]
    # sample_path.pop(0)
    # cnt = 0
    # r = sample_path.pop(0)
    # for path in sample_path:
    #     df1 = pd.read_csv(path)
    #     print(path)
    #     time_tran = pd.to_datetime(df1['DATA_TIME'])
    #     df1['DATA_TIME'] = time_tran.apply(lambda x: x.time())
    #     df_combine = pd.merge(df_combine, df1[['CONSNO', 'DATA_TIME', 'PAP_R']], how='inner',
    #     on=['CONSNO', 'DATA_TIME'])
    #     cnt = cnt + 1
    # df_combine['CONSNO'].value_counts()
    # df_r = pd.read_csv(r)
    # df_r['CONSNO'].value_counts()