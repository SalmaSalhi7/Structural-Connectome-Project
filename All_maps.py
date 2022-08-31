import os
from re import M
import numpy as np
import nibabel as nib
import bisect
import time
import pickle

# this library is for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm, colors
import csv


all_labels = np.array([4,5,8,10,11,12,13,14,15,16,17,18,24,26,28,30,31,43,44,47,49,50,51,52,53,54,58,60,62,63,85,251,252,253,254,255,1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,
1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035])

p = ['_run1'] # , '_run2', '_run3'

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

aseg_dir = '/Volumes/Toshiba4TB/' + subjects[0] 

# path extension for the subcortical stats file
aseg_stat_file = "Structural/aseg.stats"

# splitting the header in the subcortical stats file by column name
headers = "Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange".split(' ')

# reading the data from the subcortical stats file
aseg_stat = pd.read_csv(aseg_dir + aseg_stat_file, sep = '\s+', comment = '#', names = headers)

# creating a dictionary that matches structure ID with structure name
id_to_name = {i:j for i,j in zip(aseg_stat['SegId'], aseg_stat['StructName']) if i != j}

# path extension for the cortex stats file
cortex_stats = 'Structural/cortex.stats' 

# splitting the header in the cortex stats file by column name
headers = "No. Labelname R G B A".split(" ")

# reading the data from the cortex stats file
cortex_stat = pd.read_csv(aseg_dir + cortex_stats, sep = '\s+', comment = '#', names = headers)

# adding the cortical structures to the dictionary 
for i,j in zip(cortex_stat['No.'], cortex_stat['Labelname']):
    id_to_name[i] = j

# this function takes in a particular run for a subject, retrieves the corresponding bi-directional map file, and then uses it to create a dataframe that contains the connectivity matrix. Unwanted structures are removed and the final connectivity matrix is returned
def analyze(run, subject):

    # path extension for the subject's bi-directional map file for this particular run
    data_file = '/Volumes/Toshiba4TB/' + subject + 'bi-directional_maps_originaltest_subject=' + subject.split(None, 1)[0] + run + '.pkl'
    with open (data_file, 'rb') as f:
        data = pickle.load(f)

    # creating a master dataframe that is the size of the number of structures
    Master = pd.DataFrame(data[all_labels[0]].values(), columns = [id_to_name[all_labels[0]]])

    # this loop populates the dataframe with the data from the bi-directional file, excluding all unwanted structures 
    for i in range(1, len(all_labels)):
        if all_labels[i] not in [2,7,41,46,72,77,78,79,80,81,82]:
            Next = pd.DataFrame(data[all_labels[i]].values(), columns = [id_to_name[all_labels[i]]])
            Master = pd.merge(Master, Next, how = 'outer', left_index = True, right_index = True)

    # a dataframe for structure IDs
    labels_id = pd.DataFrame([i for i in data[16].keys()],  columns = ['id'])

    # a dataframe for structure names
    labels_name = pd.DataFrame([id_to_name[i] for i in data[16].keys()],  columns = ['name']) 

    pd.set_option("display.max_rows", None, "display.max_columns", None)
   
    # adding the list of names to the master dataframe that contains the data
    Master = labels_name.join(Master)

    # adding the list of ids to the master dataframe that contains the data
    Master = labels_id.join(Master)

    # a list of IDs to remove (corresponding to unwanted structures)
    remove = [2,7,41,46,72,77,78,79,80,81,82]
    msk = Master['id'] == 0

    # iterating through the list of unwanted IDs and removing them from the dataframe
    for i in remove:
        msk += Master['id'] == i
    
    # returning the final connectivity matrix
    return Master[~msk], id_to_name

# this loop iterates through all the subjects and calls the analyze function to generate connectivity files for each subject's three runs, then averages them to obtain one final connectivity matrix for each subject
for subject in subjects:

    # creating an empty connectivity array with the correct size, to be populated later
    connectivity = np.zeros((len(all_labels), len(all_labels))) # shape = (108,108)

    # iterating through each run 
    for run in p:

        # calling the analyze function to generate a connectivity matrix dataframe 
        MPM, id_to_name = analyze(run, subject)

        # making sure all unwanted labels are removed 
        for x in all_labels:
            if x not in [2,7,41,46,72,77,78,79,80,81,82]:
                MPM[id_to_name[x]][MPM['name'] == id_to_name[x]] = 0
            
        # converting the dataframe to a csv file and saving it to the hard drive
        MPM.to_csv('/Volumes/Toshiba4TB/' + subject + 'Connectivity_originaltest' + subject.split(None, 1)[0] + run + '.csv')

        print('Done creating csv file for ', run)

        # generating an array from the csv connectivity file
        l = np.genfromtxt('/Volumes/Toshiba4TB/' + subject + 'Connectivity_originaltest' + subject.split(None, 1)[0] + run + '.csv', delimiter=',', dtype = int) # shape = (108,109)
        
        # deleting the columns with the ID and names (so that we just have numbers)
        l = np.delete(l,0,axis=0)
        l = np.delete(l,0,axis=1)
        l = np.delete(l,0,axis=1)
        l = np.delete(l,0,axis=1)

        # adding this matrix to the matrices from the other runs
        connectivity += l 

    # generating an average matrix from all three runs
    average = connectivity/3

    # adding the names back to the matrix
    names = np.empty(len(id_to_name.values()), dtype=str)
    for name in id_to_name.values():
        n = str(name)
        names = np.append(names,n)

# creating the average connectivity matrix across all 100 subjects
average = np.zeros((108,108), dtype=np.float64)
for subject in subjects:
    connectivity = np.genfromtxt('/Volumes/Toshiba4TB/' + subject + 'AVERAGE_CONNECTIVITY_' + subject.split(None, 1)[0] + '.csv', delimiter = ',')
    average += connectivity

average = average/len(subjects)

np.savetxt('/Volumes/Toshiba4TB/100_SUBJECTS_AVERAGE.csv', average, delimiter=',')

   

    



    




    

    


   
    

