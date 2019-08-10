# example of using ChemicalDataset class
import pickle
import os
from datasets.chemical_dataset import ChemicalDataset



#missing "011315_26011_LIN_FPSPK"


#file="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_per50.pkl"
file="/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_pbic.pkl"

with open( file , 'rb') as f:  # Python 3: open(..., 'rb')
    bic = pickle.load(f)


lin1=ChemicalDataset("lin1")
for df_name in temp:
    lin1.add_dataframe(df_name, df)


for df_name in lin1._dataframes.keys():
    df = lin1.get_dataframe(df_name)
    lin2.add_dataframe(df_name, df)

for name, df in info["lin1"]:
    lin1.add_dataframe(name, df)


per50_dataset = ChemicalDataset( chemical_name='per50')
for name, df in per50.items():
    per50_dataset.add_dataframe(name, df)

#os.mkdir(os.path.abspath('./saved_data'))

per50_dataset.save(os.path.abspath('./saved_data'))


# dataframes = list(per50_dataset._dataframes.values())
# list(per50_dataset._dataframes.values())
# list(per50_dataset._dataframes.values())[0]
# list(per50_dataset._dataframes.items())[1][0], to get key and value


