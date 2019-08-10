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
from data.data_functions import *
import json
import os

#save



#load data
info=ChemicalDataset("bic")


# chem = "dom"
file_want='20608_DA_FPSPK'
# dom_spk['20608_DA_FPSPK'].keys()
# dom_spk['20608_DA_FPSPK']['14']

#load lfp

with open("/Volumes/BACKUP/EXTRAP/LFP_processed/pickle_dom.pkl", 'rb') as f:  # Python 3: open(..., 'rb')
    dom = pickle.load(f)

# to get chemicals
chem="bic"
# bic_spk = get_spikes(chem)
directory ='/Users/dh2744/Dropbox/Documents/Software/Python/LFP/LFP_project/saved_data'
file = os.path.join(directory,'{}'.format(chem),  '{}_spk.json'.format(chem) )

with open( file, 'w') as json_file:
    json.dump(bic_spk, json_file)

# to load spk train
bic_spk = dict()
directory ='/Users/dh2744/Dropbox/Documents/Software/Python/LFP/LFP_project/saved_data'
file = os.path.join(directory,'{}'.format(chem),  '{}_spk.json'.format(chem) )

with open( file, 'r') as json_file:
    bic_spk = json.load(json_file)

bic_spk = json.load( file )


def get_spikes(chem):
    chem_dict = dict()
    files = get_files(chem)
    for f in files:
        print(f)
        title = '_'.join((f.split(".mcd")[0].split("_")[(len(f.split(".mcd")[0].split("_")) - 3):]))
        fd = ns.File(f)
        chNamesList = get_chNamesList(fd)
        spikes = dict()

        # fd.entities[0].get_data(200) gives you the 200th spike and the LFP or shape of spike,
        for numCh in range(0, 60):

            cur_entity = fd.entities[numCh]
            cur_spk_train = []
            for curSp in range(0, cur_entity.item_count):
                # curSp = 0
                cur_spk_train.append(cur_entity.get_data(curSp)[1])

            spikes[chNamesList[numCh]] = cur_spk_train

        chem_dict[title] = spikes
    return chem_dict
# end of get_spikes

