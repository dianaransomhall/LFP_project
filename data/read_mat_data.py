# Read .mat files into python
# April 18, 2019
# Diana Hall

import os

mat_file_dir= "/Volumes/BACKUP/EXTRAP/matlab_Cina_Herr_Diana/finshed_data"

#check all .mat
x2 = [x.split(".") for x in os.listdir(mat_file_dir) ];




def MCD_read(MCDFilePath):

# import necessary libraries

import neuroshare as ns

import numpy as np

#open file using the neuroshare bindings

fd = ns.File(MCDFilePath)

#create index

indx = 0

#create empty dictionary

data = dict()

#loop through data and find all analog entities

for entity in fd.list_entities():

analog = 2

#if entity is analog

if entity.entity_type == analog:

#store entity in temporary variable

dummie = fd.entities[indx]

#get data from temporary variable

data1,time,count=dummie.get_data()

#create channel names

channelName = entity.label[0:4]+entity.label[23:]

#store data with name in the dictionary

data[channelName] = np.array(data1)

#increase index

indx = indx + 1

#return dictionary

return data