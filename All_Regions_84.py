import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import bisect
import pickle 

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.streamline import Streamlines
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import CsaOdfModel
from dipy.direction import ProbabilisticDirectionGetter
from dipy.data import small_sphere

all_labels = np.array([8,10,11,12,13,17,18,26,47,49,50,51,52,53,54,58,1001,1002,1003,1005,1006,1007,1008,1009,1010,1011,1012,
1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,2001,2002,2003,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,
2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035])

print(all_labels.size)

# this function takes in the paths for both the diffusion and structural images, the run number, and the subject path extension as parameters. It loads both images, and uses them to generate reference coordinates and create a 3D label mask. 
# all relevant labels are added to the mask (excluding hypointensities and white matter labels), and then this mask is used to generate the seeds from which the tracking will begin. The result of this tracking creates two files, one for the 
# endpoints of the streamlines and the other for the 3D array with labels in the corresponding positions. 
def Analyze(img_d_path,img_s_path,gtab, run, subject):

    # loading the diffusion image
    img_d = nib.load(img_d_path)

    # loading the structural image 
    img_s = nib.load(img_s_path)

    # extracting the shape of the diffusion image in 3D, normally image is (145,174,145,288)
    img_d_shape_3D = [j for i,j in enumerate(img_d.dataobj._shape) if i<3]

    # extracting the shape of the structural image in 3D, which is (260,311,260)
    img_s_shape_3D = img_s.dataobj._shape

    # creating the affine matrix for the diffusion image. an affine matrix is used to transform between voxel and reference coordinates
    img_d_affine = img_d.affine

    # creating the affine matrix for the structural image
    img_s_affine = img_s.affine

    # retrieving the data from the structural image. this is a 4D array 
    img_s_data = img_s.get_fdata()

    # retrieving the data from the diffusion image. this is a 3D array 
    img_d_data = img_d.get_fdata()

    # making a set of voxel coordinates in the i dimension with the same number of coordinates as the shape of the 3D structural image
    Vox_coord_s_i = np.arange(img_s_shape_3D[0])

    # making a set of voxel coordinates in the j dimension with the same number of coordinates as the shape of the 3D structural image
    Vox_coord_s_j = np.arange(img_s_shape_3D[1])

    # making a set of voxel coordinates in the k dimension with the same number of coordinates as the shape of the 3D structural image
    Vox_coord_s_k = np.arange(img_s_shape_3D[2])

    # using the affine matrix to transform into a list of REFERENCE coordinates in the i dimension
    Ref_coord_s_i = Vox_coord_s_i * img_s_affine[0,0] + img_s_affine[0,3]

    # using the affine matrix to transform into a list of REFERENCE coordinates in the j dimension
    Ref_coord_s_j = Vox_coord_s_j * img_s_affine[1,1] + img_s_affine[1,3]

    # using the affine matrix to transform into a list of REFERENCE coordinates in the k dimension
    Ref_coord_s_k = Vox_coord_s_k * img_s_affine[2,2] + img_s_affine[2,3]

    # making an empty array with the same shape as the 
    reduced_size_label = np.zeros(img_d_shape_3D)

    # iterating through all 3 dimensions, calculating the reference coordinate at each index, and then finding the new index based on the list of reference coordinates calculated above. this new index will be used to find the corresponding label at that 
    # coordinate and place it into our 3D list of labels (reduced_size_label)
    for i in range(img_d_shape_3D[0]):
        for j in range(img_d_shape_3D[1]):
            for k in range(img_d_shape_3D[2]):

                # convert to reference coordinates in the i dimension
                ref_coord_i = i * img_d_affine[0,0] + img_d_affine[0,3]

                # converting to reference coordinates in the j dimension
                ref_coord_j = j * img_d_affine[1,1] + img_d_affine[1,3]

                # converting to reference coordinates in the k dimension
                ref_coord_k = k * img_d_affine[2,2] + img_d_affine[2,3]

                # finding the index in the i dimension
                min_i_ind = bisect.bisect_left(np.sort(Ref_coord_s_i),ref_coord_i)

                # finding the index in the j dimension
                min_j_ind = bisect.bisect_left(Ref_coord_s_j,ref_coord_j)

                # finding the index in the k dimension
                min_k_ind = bisect.bisect_left(Ref_coord_s_k,ref_coord_k)

                # saving the correct label at these new indices
                reduced_size_label[i][j][k] = img_s_data[260-1-min_i_ind,min_j_ind,min_k_ind]


    print("Label image reduction successful")

    # making a mask for the left cerebral white matter (the label for this region is 2)
    left_cerebral_wm = reduced_size_label == 2 

    # making a mask for the right cerebral white matter (the label for this region is 41)
    right_cerebral_wm = reduced_size_label == 41 

    # making a mask for the total cerebral white matter by adding the left and right masks 
    cerebral_wm = left_cerebral_wm + right_cerebral_wm

    # making a mask for the left cerebellum white matter (the label for this region is 7)
    left_cerebellum_wm = reduced_size_label == 7 

    # making a mask for the right cerebellum white matter (the label for this region is 46)
    right_cerebellum_wm = reduced_size_label == 46 

    # making a mask for the total cerebellum white matter by adding the right and left masks
    cerebellum_wm = left_cerebellum_wm + right_cerebellum_wm
    
    # creating an empty matrix with the same shape as reduced_size_label to represent the left cortex
    left_cortex = np.zeros(reduced_size_label.shape)

    # populating the left cortex mask with all left cortical regions using corresponding labels
    for i in [1001,1002,1003,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035]:
        left_cortex += reduced_size_label == i

    # creating an empty matrix with the same shape as reduced_size_label to represent the right cortex    
    right_cortex = np.zeros(reduced_size_label.shape)

    # populating the right cortex mask with all right cortical regions using corresponding labels
    for i in [2001,2002,2003,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035]:
        right_cortex += reduced_size_label == i

    # creating an empty matrix to represent the mask for all subcortical structures (excluding the corpus callosum and hypointensities) 
    extra = np.zeros(reduced_size_label.shape)

    # populating the subcortical mask
    for i in [8,10,11,12,13,17,18,26,47,49,50,51,52,53,54,58]: 
        extra += reduced_size_label == i

    # adding all masks to create an overarching white matter mask to be used in fibre tracking
    wm = cerebral_wm + cerebellum_wm + extra + left_cortex + right_cortex

    # creating a set of seeds from the white matter mask and the diffusion affine matrix as a starting point for the fibre tracking
    seeds = utils.seeds_from_mask(wm, density = 1, affine = img_d_affine)

    # creating a response function to guide the fibre tracking, using the gradient table generated from the bvals and bvecs files, as well as the diffusion data
    response, ratio = auto_response_ssst(gtab, img_d_data, roi_radii = 10, fa_thr = 0.7) 

    # printing the response function. Ensure last element of the arry is roughly twice as large as the first two, and that the first two are the same
    print(response)

    # creating a constrained spherical deconvolution model from the gradient table and the response function
    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)

    # creating a fit of the model using the diffusion data and the mask
    csd_fit = csd_model.fit(img_d_data, mask = wm)

    print("CSD model complete")

    # creating a CSA model from the gradient table
    csa_model = CsaOdfModel(gtab, sh_order=6)

    # creating a numpy array to represent the gfa using the CSA model, the diffusion data, and the mask
    gfa = csa_model.fit(img_d_data, mask = wm).gfa 

    # generating a classifier (to be used in calculating the streamlines) from the gfa array
    classifier = ThresholdStoppingCriterion(gfa, .25)

    # fibre orientation distribution
    fod = csd_fit.odf(small_sphere)

    # creating a pmf from the fod 
    pmf = fod.clip(min = 0)

    # creating a probabilistic direction getter that will enable probabilistic calculation of the direction of the fibre tracking
    prob_dg = ProbabilisticDirectionGetter.from_pmf(pmf, max_angle = 45. ,sphere = small_sphere)

    # generating a list of streamlines using the probabilistic direction getter, the classifier, the seeds, and the diffusion affine image
    streamlines_generator = LocalTracking(prob_dg, classifier, seeds, img_d_affine, step_size = 0.5) 

    # making a numpy array of the streamlines
    astreamlines = np.array(list(streamlines_generator))

    print('streamlines shape: ', astreamlines.shape)

    labels = np.array(reduced_size_label, dtype=int)

    # making a numpy array of ednpoints based on the streamlines. 
    endpoints = np.array([st[::len(st) - 1] for st in astreamlines if len(st) > 1])

    endpoints_title = '/Volumes/Extreme SSD/Brain/Toshiba4TB/' + subject + 'endpoints_maxangle=45_gfa=0.25_subject=' + subject.split(None, 1)[0] + run + '.pkl'
    reduced_label_title = '/Volumes/Extreme SSD/Brain/Toshiba4TB/' + subject + 'labels_maxangle=45_gfa=0.25_subject=' + subject.split(None, 1)[0] + run + '.pkl'
    
    with open(endpoints_title,'wb') as f:
        pickle.dump(endpoints,f)
    with open(reduced_label_title,"wb") as g:
        pickle.dump(reduced_size_label,g)

# an array with labels for each run (each subject will be run three times)
p = ["_84run1, _84run2, _84run3"] 

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


"""Importing diffusion data"""

# path for the data (located on external hard drive)
path = '/Volumes/Extreme SSD/Brain/Toshiba4TB/'

# path extension for diffusion data
datafile_d = 'Diffusion/data.nii.gz'

"""Importing structural data"""

# path extension for structural data
datafile_aparc_aseg = "Structural/aparc+aseg.nii.gz"

# iterating through each subject in the list of subjects and generating three runs for each
for subject in subjects:

    bvals = np.genfromtxt(path + subject + 'Diffusion/bvals', dtype = 'int')
    bvecs = np.genfromtxt(path + subject + 'Diffusion/bvecs')
    bvecs = np.array([[i,j,k] for i,j,k in zip(*bvecs)], dtype = 'float64')
    gtab = gradient_table(bvals,bvecs = bvecs)

    assert gtab.bvecs.shape == bvecs.shape
    assert gtab.bvals.shape == bvals.shape

    # analyzing three times per subject (three runs)
    for run in p:
        Analyze(path + subject + datafile_d, path + subject + datafile_aparc_aseg, gtab, run, subject)
        print('Done ', run)
        

    print('Done subject ', subject.split(None, 1)[0])