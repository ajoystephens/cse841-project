import numpy as np
import pandas as pd
import raha
import pickle
from argparse import ArgumentParser

argParser = ArgumentParser()
argParser.add_argument("-d", "--dataset", default='compas', help="dataset")

args = argParser.parse_args()


path = f'datasets/{args.dataset}/raha-baran-results-{args.dataset}/error-correction/correction.dataset'
# path = f'datasets/compas/raha-baran-results-compas/error-correction/correction.dataset'

# retrieve raha data object
objects = []
with (open(path, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
data = objects[0]

# extract desired data
actual_dict = data.get_actual_errors_dictionary()
correction_dict = data.corrected_cells

# data.create_repaired_dataset(correction_dict)
# corrected_df = data.repaired_dataframe.astype(float)

path = f'datasets/{args.dataset}/dirty.csv'
# path = f'datasets/compas/dirty.csv'
dirty_df = pd.read_csv(path, sep=",", header=0, dtype=float)

corrected_df = dirty_df.copy()
for cell in correction_dict:
    val = correction_dict[cell]
    if val == '': val='nan'
    dirty_df.iloc[cell] = float(val)

# save corrected dataset
corrected_df.to_csv("datasets/compas/corrected.csv", sep=",", header=True, index=False)
# np.savetxt("datasets/compas/corrected.csv", corrected_df.to_numpy(), delimiter=",")

# save other info
data_dict = {
    'actual_errors':actual_dict,
    'corrected_errors':correction_dict,
}
file = open('datasets/compas/errors.p', 'wb')
pickle.dump(data_dict,file)
file.close()
