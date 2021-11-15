import os
import numpy as np
import nibabel as nib
import bisect
import time
import pickle
import pandas as pd
import copy

import matplotlib.pyplot as plt

import csv

# labels for the run number, total of three runs
p = ["_run1", "_run2", "_run3"] #

# an array with labels for each subject (total of 100 subjects)
subjects = ['100307 F26-30/', '102109 M26-30/', '102513 M26-30/', '102614 M22-25/', '102715 M26-30/', '111211 F26-30/', '114116 F26-30/', '115724 F22-25/', '116423 M26-30/', '117021 F26-30/', 
'118831 M22-25/', '120010 F26-30/', '120414 F26-30/', '122418 F26-30/', '125222 M26-30/', '125424 F31-35/', '126426 F31-35/', '130720 M31-35/', '134627 M26-30/', '135124 F31-35/', 
'135629 M22-25/', '136126 M26-30/', '136631 M22-25/', '137431 F31-35/', '137532 M31-35/', '138130 F31-35/', '138332 M26-30/', '139435 F26-30/', '143224 F26-30/', '143830 F26-30/', 
'144933 M26-30/', '146735 F31-35/', '146836 M26-30/', '151121 M31-35/', '151325 M22-25/', '151931 M31-35/', '152225 M31-35/', '152427 F31-35/', '153126 F26-30/', '153934 M26-30/', 
'154330 M26-30/', '161832 M26-30/', '165436 M26-30/', '165941 F26-30/', '167440 F26-30/', '168947 F26-30/', '169545 M22-25/', '172635 M31-35/', '175136 F22-25/', '176845 F26-30/', 
'177140 F31-35/', '180230 F26-30/', '180533 F31-35/', '183741 M31-35/', '185083 M31-35/', '186040 M26-30/', '186545 F31-35/', '186848 F26-30/', '188145 F31-35/', '189652 M26-30/', 
'191235 M22-25/', '192237 F31-35/', '193845 M22-25/', '194443 M26-30/', '196851 M26-30/', '196952 F26-30/', '198047 F26-30/', '199352 F22-25/', '200513 M22-25/', '204218 M26-30/', 
'206323 F31-35/', '206525 M26-30/', '206727 M22-25/', '206828 M22-25/', '206929 M31-35/', '208630 M31-35/', '210112 M22-25/', '211619 M26-30/', '211821 M26-30/', '213017 M22-25/', 
'213522 M26-30/', '219231 M26-30/', '227533 F31-35/', '238033 M31-35/', '248238 F31-35/', '255740 F26-30/', '257946 F26-30/', '274542 F26-30/', '281135 M26-30/', '286347 F26-30/', 
'299760 M26-30/', '300719 M26-30/', '314225 M22-25/', '325129 M31-35/', '329844 M26-30/', '342129 M26-30/', '349244 M26-30/', '350330 F22-25/', '360030 F31-35/', '362034 F31-35/'] 

# this function takes in a run number and a subject ID as input and uses these parameters to retrieve the endpoints and reduced mask matrices generated for that subject in All_Regions.py
# it then creates a new connectivity matrix based on the endpoints
def analyze(run, subject):

    # retrieving the endpoints file for this subject
    endpoints_title = '/Volumes/Toshiba4TB/' + subject + 'endpoints_maxangle=45_gfa=0.25_subject=' + subject.split(None, 1)[0] + run + '.pkl'

    # retrieving the reduced label mask for this subject
    reduced_label_title = '/Volumes/Toshiba4TB/' + subject + 'labels_maxangle=45_gfa=0.25_subject=' + subject.split(None, 1)[0] + run + '.pkl'

    with open(endpoints_title, 'rb') as f:
        endpoints = pickle.load(f)

    with open(reduced_label_title, 'rb') as g:
        label = pickle.load(g)

    endpoints_label_map = np.zeros([endpoints.shape[0], endpoints.shape[1]])

    print("Creating endpoints_label_map")

    for i in range(endpoints_label_map.shape[0]): 
        if i == endpoints_label_map.shape[0]//2:
            print("Creating endoints_label_map Halfway done")
        
        # np.round rounds an array to a given number of decimals, default is zero (basically making everything a whole number)
        # not sure why we're adding np.array([-90,126,72]) and multiplying by np.array([-1/1.25,1/1.25,1/1.25])
        a = np.array(np.round((endpoints[i][0] + np.array([-90,126,72])) * np.array([-1/1.25,1/1.25,1/1.25])), dtype = 'int32')
        b = np.array(np.round((endpoints[i][1] + np.array([-90,126,72])) * np.array([-1/1.25,1/1.25,1/1.25])), dtype = 'int32')
        
        # assigning the correct label to each element under the ith key using the reduced label mask
        endpoints_label_map[i][0] = label[a[0],a[1],a[2]]
        endpoints_label_map[i][1] = label[b[0],b[1],b[2]]

    print("Done creating endpoints_label_map")

    # path directory for this subject
    aseg_dir = '/Volumes/Toshiba4TB/' + subject

    # path extension for structural data
    aseg_stat_file = "Structural/aseg.stats"

    # splitting the headers string in the aseg file into individual column headers
    headers = "Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange".split(' ') # for aseg

    # reading the aseg stats csv file which contains information on the subcortical regions of interest
    aseg_stat = pd.read_csv(aseg_dir + aseg_stat_file, sep = '\s+', comment = '#', names = headers)
                    
    # path extension for cortex stats data                
    cortex_stats = 'Structural/cortex.stats' 

    # splitting the headers string in the cortex stats file into individual column headers
    headers = "No. Labelname R G B A".split(" ")

    # reading the cortex stats csv file
    cortex_stat = pd.read_csv(aseg_dir + cortex_stats, sep = '\s+', comment = '#', names = headers)

    # creating a dictionary that maps the subcortical regions ID to its name
    id_to_name = {i:j for i,j in zip(aseg_stat['SegId'], aseg_stat['StructName']) if i != j}

    # adding the cortical structure IDs and names to the dictionary
    for i,j in zip(cortex_stat['No.'], cortex_stat['Labelname']):
        id_to_name[i] = j

    # sorting the dictionary by ID
    dict_items = id_to_name.items()
    id_to_name = sorted(dict_items)
    id_to_name = dict(id_to_name)

    label_map = {}

    # creating a new dictionary called 'label_map' with the same keys as 'id_to_name' and setting everything equal to 0
    for i in id_to_name.keys():
        label_map[i] = {}
        for j in id_to_name.keys():
            label_map[i][j] = 0

    # increasing the value of each endpoint label by 1 
    for e in endpoints_label_map:
        if e[0] not in [2,41,7,46,0]: # 41 = Right Cerebral White Matter
            if e[1] not in [2,41,7,46,0]:
                label_map[e[0]][e[1]] += 1

    new_label_map = copy.deepcopy(label_map)
    for i in id_to_name.keys():
        for j in id_to_name.keys():
            new_label_map[i][j] += label_map[j][i]

    return new_label_map

# creating a bi-directional map file for each run (300 files total)
for subject in subjects:
    for run in p:
        label_map = analyze(run, subject)
        #title = '/Users/salmasalhi/my_code/HumanBrain_v2/' + subject + 'bi-directional_label_map-shorder=6-maxangle=45-gfa=0.25-subject=' + subject.split(None, 1)[0] + '_less_steps.pkl'
        title = '/Volumes/Toshiba4TB/' + subject + 'bi-directional_maps_subject=' + subject.split(None, 1)[0] + run +'.pkl'  
        with open (title, 'wb') as l:
            pickle.dump(label_map, l)
        print('Done ', run)
    print('Done subject ', subject.split(None, 1)[0])

