import pandas as pd
from itertools import combinations_with_replacement
from argparse import ArgumentParser


def main(csv_file):
    tab = pd.read_csv(csv_file, index_col=None, header=None, sep=";")
    tt = tab.transpose()

    # change column names
    tt.iloc[0, 0] = 'Inter-continental Region'
    tt.iloc[0, 1] = 'ID'
    tt.iloc[0, 2] = 'Detailed-continental Region'


    # change headers
    new_header = tt.iloc[0]  # grab the first row for the header
    tt = tt[1:]  # take the data less the header row
    tt.columns = new_header

    # set row index
    tt.set_index('ID', inplace=True)

    map_dict = {}
    for k, (i, j) in enumerate(combinations_with_replacement(range(4), 2)):
        map_dict[f'{i}|{j}'] = k
        map_dict[f'{j}|{i}'] = k

    # When arg is a dictionary, values in Series that are not in the dictionary (as keys) are converted to NaN.
    for c in tt.columns:
        if c not in ['ID', 'Inter-continental Region', 'Detailed-continental Region']:
            tt[c] = tt[c].map(map_dict, na_action='ignore').astype("Int64")

    # remove - and space in region names
    tt['Inter-continental Region'] = tt['Inter-continental Region'].apply(lambda x: x.replace('-', '.').replace(' ', ''))
    tt['Detailed-continental Region'] = tt['Detailed-continental Region'].apply(lambda x: x.lower())

    tt.to_pickle('data/loaded_data.pkl')


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('csv_file', type=str, help='CSV file which contains the data.')
    args = argparser.parse_args()
    main(args.csv_file)
