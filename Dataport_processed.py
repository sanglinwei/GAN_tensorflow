
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from keras.datasets import mnist

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
    root_path = 'dataport/metadata.csv'
    metadata = pd.read_csv(root_path)
    metadata_extract = metadata[['dataid', 'city', 'state']]
    metadata_extract.to_csv('processed_data/metadata_extract.csv')



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

    # # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    X_train = pd.DataFrame(X_train.reshape([-1, 28*28]))

    df1 = pd.read_csv('./dataport/load_profile.csv')
    df1.dropna()
    idx = int(df1.shape[0] / (28 * 28)) * 28 * 28
    np1 = df1[0:idx].to_numpy()[:, 1]
    for i in range(df1.shape[1] - 2):
        np1 = np.concatenate((np1, df1[0:idx].to_numpy()[:, i + 2]), axis=0)
    np2 = np1.reshape((-1, 28, 28))
    load_data = np.expand_dims(np2, axis=3)

    # scale to -1 - 1
    scale = np1.max() - np1.min()
    scaled_load_data = (load_data - np1.min()) / scale * 2 - 1
    load_data_pd = pd.DataFrame(scaled_load_data.reshape([-1, 28*28]))





