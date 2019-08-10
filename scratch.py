import numpy as np
import pandas as pd
import os


load_dir = '/Users/dh2744/Dropbox/Documents/Software/Python/Taxi_homework/minibook-2nd-data'
name1='data/nyc_data'
name2='data/nyc_fare'

with open(os.path.join(load_dir, '{}.csv'.format(name1)), 'r') as f:
    data = pd.DataFrame( pd.read_csv(f ) )










