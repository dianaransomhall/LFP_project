import neuroshare as ns
import os
import statistics
import pickle
import numpy as np
import shelve
import scipy as sp
import math
import decimal
from scipy import fftpack, signal, stats
from scipy.fftpack import rfft, irfft, fftfreq
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from datasets.chemical_dataset import ChemicalDataset
from data.data_functions import get_files

import os

#save



#load data


# chem = "dom"
file_want='20608_DA_FPSPK'
# dom_spk['20608_DA_FPSPK'].keys()
# dom_spk['20608_DA_FPSPK']['14']

#load lfp
files = get_files(chem="dom")

with open("/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_dom.pkl", 'rb') as f:  # Python 3: open(..., 'rb')
    dom = pickle.load(f)

dom_dataset = ChemicalDataset(chemical_name='dom')
for name, df in dom.items():
    dom_dataset.add_dataframe(name, df)

os.mkdir(os.path.abspath('./saved_data'))

dom_dataset.save(os.path.abspath('./saved_data'))