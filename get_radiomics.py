import os
import h5py
import numpy as np
import SimpleITK as sitk
import csv
import re
from radiomics import featureextractor

# this directory is only for testing
base_volume_dir = './test_dir'
def load_data(h5_path):
    with h5py.File(h5_path, 'r') as file:
        mask = file['mask'][:]    
        image = file['image'][:]  
        image = image.astype(np.float32)
    return image,mask

# Load all data from the directory
def load_3D(volume_dir):
    all_masks = []
    all_images = []
    
    filenames = sorted(os.listdir(volume_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # volume_dir
    for filename in filenames:
        # filename=volume_dir+"/"+filename
        filename=os.path.join(volume_dir,filename)
        #print(filename)
        if filename.endswith('.h5'):
            #print(filename)
            # load
            #image, mask = load_data(os.path.join(volume_dir, filename))
            image, mask = load_data(filename)
            all_masks.append(mask)
            all_images.append(image)

    # put together
    stacked_masks = np.stack(all_masks)
    stacked_images = np.stack(all_images)
    return  stacked_images,stacked_masks


# since stacked_masks.shape=(155, 240, 240, 3)，reduce dimension to  (155, 240, 240) by sum
def load_and_adjust(volume_dir):
    stacked_images,stacked_masks=load_3D(volume_dir)
    stacked_images=stacked_images[:, :, :, 0]
    sum_masks = np.sum(stacked_masks, axis=-1)
    return stacked_images,sum_masks


# get features for one volume and return 
def get_radiomics(volume_dir=base_volume_dir):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    #volume_dir = './archive/BraTS2020_training_data/content/data/volume_7'
    stacked_images,stacked_masks=load_and_adjust(volume_dir)
    # features = extract_radiomics(stacked_images,stacked_masks,)
    # features = extractor.execute(sitk.GetImageFromArray(stacked_images,True), sitk.GetImageFromArray(stacked_masks,True))
    sitk.GetImageFromArray(stacked_images,True)
    features = extractor.execute(sitk.GetImageFromArray(stacked_images,False), sitk.GetImageFromArray(stacked_masks,False),label_channel=1)
    return features


# Input parameter is a directory contains multi subfolders and return the volumes number as a list
def get_volumes(directory=base_volume_dir):
    items = os.listdir(directory)
    print
    # to get all the ids from the directory
    volume_numbers = [int(re.search(r'\d+', item).group()) for item in items if re.search(r'\d+', item)]
    return sorted(volume_numbers)




# col_list should be a list of top 10 features,this function gets all the readiomic features on directory
def get_all_radiomics(directory=base_volume_dir, col_list=None):
    extractor = featureextractor.RadiomicsFeatureExtractor()
    # csv_file_path = base_volume_dir+'/radiomic_features.csv'
    csv_file_path = os.path.join(directory , 'radiomic_features.csv')
    id_list=get_volumes(directory)
    with open(csv_file_path, mode='w', newline='') as csv_file:
        
        csv_writer = csv.writer(csv_file)
        # head line for the csv file
        feature_names = list(col_list)
        csv_writer.writerow(['volume_id'] + feature_names)  

        for volume_id in id_list:
            #volume_dir = f'./archive/BraTS2020_training_data/content/data/volume_{volume_id}'
            volume_dir = os.path.join(directory, f'volume_{volume_id}')
            # load data
            stacked_images, stacked_masks = load_and_adjust(volume_dir)

            # features = extract_radiomics(sitk.GetImageFromArray(stacked_images, True), sitk.GetImageFromArray(stacked_masks, True))
            features = extractor.execute(sitk.GetImageFromArray(stacked_images,False), sitk.GetImageFromArray(stacked_masks,False),label_channel=1)       
            csv_writer.writerow([f'volume_{volume_id}'] + [features[feature_name] for feature_name in feature_names])




# test code
# since we still dont have the feature list result, we need to get them as list.

# features=get_radiomics('test_dir/volume_1')

# feature_names = ['original_shape_Sphericity',
#         'original_shape_SurfaceVolumeRatio',
#         'original_shape_Flatness',
#         'original_shape_Maximum3DDiameter',
#         'original_shape_Elongation',
#         'original_shape_LeastAxisLength',
#         'original_shape_Maximum2DDiameterSlice',
#         'original_shape_MajorAxisLength',
#         'original_shape_MeshVolume',
#         'original_shape_SurfaceArea',
#         'original_firstorder_Mean',
#         'original_firstorder_RootMeanSquared',
#         'original_firstorder_90Percentile',
#         'original_firstorder_Median',
#         'original_firstorder_InterquartileRange',
#         'original_firstorder_RobustMeanAbsoluteDeviation',
#         'original_firstorder_Maximum',
#         'original_firstorder_MeanAbsoluteDeviation',
#         'original_firstorder_Range',
#         'original_firstorder_10Percentile',
#         'original_glszm_GrayLevelNonUniformity',
#         'original_firstorder_Variance',
#         'original_glszm_ZonePercentage',
#         'original_glszm_SizeZoneNonUniformity',
#         'original_glszm_ZoneEntropy',
#         'original_gldm_DependenceNonUniformityNormalized',
#         'original_gldm_LargeDependenceEmphasis',
#         'original_gldm_DependenceEntropy',
#         'original_glszm_SizeZoneNonUniformityNormalized',
#         'original_glrlm_RunPercentage']
# get_all_radiomics(directory="./test2", col_list=feature_names)