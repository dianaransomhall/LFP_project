# example of using ChemicalDataset class
import pickle
import os
from datasets.chemical_dataset import ChemicalDataset


file="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_per50.pkl"

with open( file , 'rb') as f:  # Python 3: open(..., 'rb')
    per50 = pickle.load(f)



per50_dataset = ChemicalDataset( chemical_name='per50')
for name, df in per50.items():
    print('hi')
    per50_dataset.add_dataframe(name, df)

#os.mkdir(os.path.abspath('./saved_data'))

per50_dataset.save(os.path.abspath('./saved_data'))







