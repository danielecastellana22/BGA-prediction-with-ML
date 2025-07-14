from argparse import ArgumentParser
import os
import os.path as osp
import numpy as np
from sklearn.model_selection import StratifiedKFold


def store_splits(splits_folder, y, n_outer_splits=5, n_inner_splits=5):
    outer_CV = StratifiedKFold(n_outer_splits, shuffle=True)
    inner_CV = StratifiedKFold(n_inner_splits, shuffle=True)

    x = np.zeros(y.shape[0])
    for i, (train_val_index, test_index) in enumerate(outer_CV.split(x, y)):

        for j, (train_index, val_index) in enumerate(inner_CV.split(x[train_val_index], y[train_val_index])):
            dir_path = osp.join(splits_folder, f'{i}_{j}')
            os.makedirs(dir_path)

            train_index = train_val_index[train_index]
            val_index = train_val_index[val_index]

            assert list(sorted(train_index.tolist() + val_index.tolist() + test_index.tolist())) == list(range(y.shape[0]))
            assert np.unique(y[train_index]).tolist() == np.unique(y[val_index]).tolist() == np.unique(y[test_index]).tolist()

            with open(osp.join(dir_path, 'train_idx.txt'), 'w') as f:
                f.write('\n'.join(map(str, train_index.tolist())))

            with open(osp.join(dir_path, 'val_idx.txt'), 'w') as f:
                f.write('\n'.join(map(str, val_index.tolist())))

            with open(osp.join(dir_path, 'test_idx.txt'), 'w') as f:
                f.write('\n'.join(map(str, test_index.tolist())))


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('data_path', type=str, help='Path dei dati da splittare.')
    args = argparser.parse_args()

    y = np.load(osp.join(args.data_path, 'y.npy'), allow_pickle=True)

    n_outer_splits = 5
    n_inner_splits = 5

    d_name = osp.split(args.data_path)[-1]
    output_folder = osp.join('splits', d_name, f'{n_outer_splits}_{n_inner_splits}')
    os.makedirs(output_folder)

    store_splits(output_folder, y, n_outer_splits, n_inner_splits)



