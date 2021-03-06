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

p = ['_run1', '_run2', '_run3']

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

# aseg_dir = '/Users/salmasalhi/my_code/HumanBrain_v2/102109 M26-30/' 
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
    data_file = '/Volumes/Toshiba4TB/' + subject + 'bi-directional_maps_subject=' + subject.split(None, 1)[0] + run + '.pkl'
    with open (data_file, 'rb') as f:
        data = pickle.load(f)

    # creating a master dataframe that is the size of the number of structures
    Master = pd.DataFrame(data[all_labels[0]].values(), columns = [id_to_name[all_labels[0]]])

    # this loop populates the dataframe with the data from the bi-directional file, excluding all unwanted structures 
    for i in range(1, len(all_labels)):
        if all_labels[i] not in [2,7,41,46,72,77,78,79,80,81,82]:
            Next = pd.DataFrame(data[all_labels[i]].values(), columns = [id_to_name[all_labels[i]]])
            Master = pd.merge(Master, Next, how = 'outer', left_on = id_to_name[all_labels[0]], right_on = id_to_name[all_labels[i]], left_index = True, right_index = True)

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

# this function generates a connectivity matrix only for the top 20 most connected structures (the top 20 structures with the greatest node weight)
def analyze_top(run, subject, top):

    # path extension for the bi-directional map file
    data_file = data_file = '/Volumes/Toshiba4TB/' + subject + 'bi-directional_maps_subject=' + subject.split(None, 1)[0] + run + '.pkl'
    with open (data_file, 'rb') as f:
        data = pickle.load(f)

    # creating an empty master dataframe with a size of 20
    Master = pd.DataFrame(data[top[1]].values(), columns = [id_to_name[top[1]]])

    # this loop populates the dataframe with the data from the bi-directional file that matches the labels in the top array 
    for i in range(1, len(top)):
        Next = pd.DataFrame(data[top[i]].values(), columns = [id_to_name[top[i]]])
        Master = pd.merge(Master, Next, how = 'outer', left_on = id_to_name[top[0]], right_on = id_to_name[top[i]], left_index = True, right_index = True)

    # a dataframe for structure IDs
    labels_id = pd.DataFrame([i for i in data[16].keys()],  columns = ['id'])

    # a dataframe for structure names
    labels_name = pd.DataFrame([id_to_name[i] for i in data[16].keys()],  columns = ['name']) 

    pd.set_option("display.max_rows", None, "display.max_columns", None)
   
    # adding the structure names to the master dataframe
    Master = labels_name.join(Master)

    # adding the structure IDs to the master dataframe
    Master = labels_id.join(Master)

    # removing unwanted structures
    remove = [2,7,41,46,8,47,77,78,79,80,81,82]
    msk = Master['id'] == 0
    for i in remove:
        msk += Master['id'] == i

    # creating a mask for all rows that contain data corresponding to structures NOT in the top array
    for label in all_labels:
        if label not in top:
            msk += Master['id'] == label

    # returning the inverse of the mask (ie only the part of the dataframe that contains the top array information)
    return Master[~msk]

# this function finds the top 20 structures with the largest node weight and returns a list containing the names of those structures
def find_top(subject, average, id_to_name):

    # creating an array that is the same size as the average matrix which will be populated with the summed connectivity of each row in the connectivity matrix
    sums = np.zeros((len(average)))
    i = 0
    
    # iterating through each row in the connectivity matrix and summing the connectivities along the row, then adding this sum to the sum array
    for row in average:
        sum = 0
        for col in row:
            sum += col 
        sums[i] = sum
        i += 1

    # creating a list of structure names from the id_to_name dictionary generated in the analyze function
    names = []
    for key in id_to_name.keys():
        if key in all_labels:
            names.append(id_to_name[key])

    # creating a dictionary that matches each sum to the corresponding structure
    sums_dict = {i:j for i,j in zip(sums, names)}

    # sorting the sums array from lowest to highest
    sums = np.sort(sums)

    # reversing the sorted sums array
    sorted_sums = sums[::-1]

    # creating a new array that contains only the top 20 sums
    sorted_top_sums = sorted_sums[:20]
    
    # creating an array of size 20 that will be populated with the names of the top 20 structures
    top_names = np.array((len(sorted_top_sums)))

    # iterating through each sum in the top 20 sums and picking out the structure name it corresponds to, then saving the name into the top_names array
    for sum in sorted_top_sums:
        if sum in sums_dict.keys():
            top_names = np.append(top_names, sums_dict[sum])

    # deleting an extra row in top_names
    top_names = np.delete(top_names, 0)

    # finding the IDs that correspond to the top 20 structures by iterating through the keys in id_to_name and matching the dictionary value they correspond to to the names in top_names
    top_labels = []
    for key in id_to_name.keys():
        if id_to_name[key] in top_names:
            top_labels.append(key)

    # saving the list of 20 names as a txt file
    np.savetxt('/Volumes/Toshiba4TB/' + subject + 'TOP_20_NODES_NAMES_' + subject.split(None, 1)[0] + '.txt', top_names, delimiter=' ', fmt='%s')

    # saving the list of 20 IDs (labels) as a txt file
    np.savetxt('/Volumes/Toshiba4TB/' + subject + 'TOP_20_LABELS_STREAMLINE_COUNT' + subject.split(None, 1)[0] + '.txt', top_labels, delimiter=' ', fmt='%s')

    # creating an empty array that will be populated with the symmetrical connectivity matrix corresponding only to the top 20 structures
    first_average = []

    # copying the average connectivity matrix into first_average
    for row in average:
        first_average.append(row[0])

    # creating a dictionary that matches structure names to the corresponding connectivity row
    name_to_connectivity = {i:j for i,j in zip(id_to_name.values(), first_average)}

    # iterating through each name in the dictionary and deleting rows that are not one of the top 20 structures, to create a 20 by 104 (I think?) matrix
    for key in list(name_to_connectivity):
        if key not in top_names:
            name_to_connectivity.pop(key)

    # copying the average connectivity into a new array
    average = np.array(average, dtype = object)

    # inserting the names column into the average connectivity matrix 
    average = np.insert(average, 0, names, axis=1)
    names.insert(0, '')
    top_rows = [names]
    i = 0

    # iterating through each row in the average connectivity matrix and appending rows of connectivity data to the top_rows array only if the structure name that corresponds to the data is in top_names
    for row in average:
        if row[0] in top_names:
            top_rows.append(row)

    # transposing the top_rows array so that we can make the matrix symmetric
    top_columns = list(map(list, zip(*top_rows)))

    # iterating through the transposed array and again removing data that does not correspond to one of the top 20 structures, so that we have a 20 by 20 matrix in the end
    final_top = [top_columns[0]]
    for row in top_columns:
        if row[0] in top_names:
            final_top.append(row)

    final_top = np.array(final_top)
    
    # returing the final top 20 connectivity matrix 
    return final_top

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
        MPM.to_csv('/Volumes/Toshiba4TB/' + subject + 'Connectivity_' + subject.split(None, 1)[0] + run + '.csv')

        print('Done creating csv file for ', run)

        # generating an array from the csv connectivity file
        l = np.genfromtxt('/Volumes/Toshiba4TB/' + subject + 'Connectivity_' + subject.split(None, 1)[0] + run + '.csv', delimiter=',', dtype = int) # shape = (108,109)
        
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

    # saving the average connectivity matrix for this subject as a csv file
    np.savetxt('/Volumes/Toshiba4TB/' + subject + 'AVERAGE_CONNECTIVITY_' + subject.split(None, 1)[0] + '.csv', average, delimiter=',')

    # finding the top 20 structures (node weight)
    #final_top = find_top(subject, average, id_to_name)
    #top_dataframe = analyze_top(subject, final_top)

    # saving the top 20 structures into a csv
    #top_dataframe.to_csv('/Volumes/Toshiba4TB/' + subject + 'TOP_20_REGIONS_STREAMLINE_COUNT_' + subject.split(None, 1)[0] + '.csv', delimiter=',', dtype=int)

    #with open('/Volumes/Toshiba4TB/' + subject + 'TOP_20_REGIONS_STREAMLINE_COUNT_' + subject.split(None, 1)[0] + '.csv', 'w') as out:
        #writer = csv.writer(out)
        #writer.writerows(final_top)

# creating the average connectivity matrix across all 100 subjects
average = np.zeros((108,108), dtype=np.float64)
for subject in subjects:
    connectivity = np.genfromtxt('/Volumes/Toshiba4TB/' + subject + 'AVERAGE_CONNECTIVITY_' + subject.split(None, 1)[0] + '.csv', delimiter = ',')
    average += connectivity

average = average/len(subjects)

np.savetxt('/Volumes/Toshiba4TB/100_SUBJECTS_AVERAGE_TEST.csv', average, delimiter=',')

   

    



    




    

    


   
    

