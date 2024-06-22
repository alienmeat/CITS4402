import csv
import h5py
import numpy as np
import os
from scipy.ndimage import label, binary_erosion
from numpy.linalg import svd
import re

# Define the base directory where all volumes are stored
base_volume_dir = './test_dir'

def load_data(h5_path):
    with h5py.File(h5_path, 'r') as file:
        image = file['image'][:]
        mask = file['mask'][:]
        if image.ndim == 4:  
            image = np.mean(image, axis=-1)  # Convert to grayscale by averaging channels
        if mask.ndim == 4:  
            mask = mask[..., 0]  # Take the first channel assuming mask is binary
    return image, mask

def max_tumor_area(masks):
    return max(np.sum(mask) for mask in masks)


def max_tumor_diameter_pca(masks):
    diameters = []
    for mask in masks:
        labeled_mask, num_features = label(mask)
        if num_features == 0:
            continue

        # Identify the largest component
        component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_features + 1)]
        largest_component_label = np.argmax(component_sizes) + 1
        largest_component = labeled_mask == largest_component_label

        # Extract positions of the largest component
        positions = np.column_stack(np.where(largest_component))
        if positions.shape[0] > 1:  # Ensure there are enough points for PCA
            mean_centered = positions - np.mean(positions, axis=0)
            _, s, vh = svd(mean_centered, full_matrices=False)
            for axis, variance in zip(vh, s):
                if axis[1] != 0:
                    slope = axis[0] / axis[1]
                else:
                    slope = np.inf  # Vertical line
                if np.isfinite(slope):
                    intercept = np.mean(positions[:, 0]) - slope * np.mean(positions[:, 1])
                    projected_distances = (positions[:, 0] - slope * positions[:, 1] - intercept) / np.sqrt(
                        1 + slope ** 2)
                else:
                    projected_distances = positions[:, 1]  # Handle vertical projection
                max_distance = np.ptp(projected_distances)
                diameters.append(max_distance)
    return max(diameters) if diameters else 0


def max_tumor_diameter_simple(masks):
    max_diameters = []
    for mask in masks:
        labeled_mask, num_features = label(mask)
        if num_features == 0:
            continue

        # Find the largest connected component
        max_component = None
        max_size = 0
        for i in range(1, num_features + 1):
            component = (labeled_mask == i)
            size = np.sum(component)
            if size > max_size:
                max_size = size
                max_component = component

        # Calculate the diameter of the largest component
        positions = np.column_stack(np.where(max_component))
        x_min, x_max = positions[:, 1].min(), positions[:, 1].max()
        y_min, y_max = positions[:, 0].min(), positions[:, 0].max()
        max_diameters.append(max(x_max - x_min, y_max - y_min))

    return max(max_diameters) if max_diameters else 0





def max_tumor_diameter_by_angle(masks):
    max_diameters = []
    for mask in masks:
        labeled_mask, num_features = label(mask)
        if num_features == 0:
            continue

        # Extract the largest component
        max_component = None
        max_size = 0
        for i in range(1, num_features + 1):
            component = (labeled_mask == i)
            size = np.sum(component)
            if size > max_size:
                max_size = size
                max_component = component

        positions = np.column_stack(np.where(max_component))
        max_diameter_for_mask = 0
        for angle in range(0, 180, 1):
            radians = np.radians(angle)
            projections = positions[:, 1] * np.cos(radians) + positions[:, 0] * np.sin(radians)
            diameter = projections.max() - projections.min()
            max_diameter_for_mask = max(max_diameter_for_mask, diameter)

        max_diameters.append(max_diameter_for_mask)

    return max(max_diameters) if max_diameters else 0


def outer_layer_involvement(images, masks, thickness=5):
    involvements = []
    for image, mask in zip(images, masks):
        # Ensure mask is single-channel (if it's not already)
        if mask.ndim == 3:
            mask = np.any(mask, axis=-1)  # Assuming the mask is binary, reduce across the channel dimension

        # Create a single-channel brain mask from the first channel if not already single-channel
        if image.ndim == 3:
            image = image[..., 0]

        # Define the brain tissue based on intensity and create the eroded mask
        brain_mask = image > np.mean(image)  # Simple thresholding
        eroded_brain_mask = binary_erosion(brain_mask, iterations=thickness)
        outer_layer_mask = brain_mask & ~eroded_brain_mask

        # Calculate the overlap of tumor and outer layer
        overlap = outer_layer_mask & mask
        if np.any(outer_layer_mask):
            involvement = np.sum(overlap) / np.sum(outer_layer_mask)
            involvements.append(involvement)
        else:
            involvements.append(0)

    # Compute the mean involvement over all images
    return np.mean(involvements) * 100


def process_volume(volume_dir, csv_writer):
    filenames = sorted(os.listdir(volume_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # print(filenames)
    masks = []
    images = []
    for filename in filenames:
        if filename.endswith('.h5'):
            image, mask = load_data(os.path.join(volume_dir, filename))
            if np.any(mask):  # Only consider non-zero slices
                masks.append(mask)
                images.append(image)

    if masks:
        max_area = max_tumor_area(masks)
        max_diameter_pca = max_tumor_diameter_pca(masks)
        max_diameter_simple = max_tumor_diameter_simple(masks)
        max_diameter_angle = max_tumor_diameter_by_angle(masks)
        avg_involvement = outer_layer_involvement(images, masks)


        csv_writer.writerow([volume_dir.split('_')[-1], max_area, max_diameter_angle, avg_involvement])
        #  print(f"Processed {volume_dir}: Max Area {max_area}, Max Diameter PCA {max_diameter_pca}, Max Diameter Simple {max_diameter_simple}, Avg Involvement {avg_involvement:.2f}%")
        print(f"Processed {volume_dir}: Max Diameter PCA {max_diameter_pca}, Max Diameter Simple {max_diameter_simple}, Max Diameter Angle {max_diameter_angle}")



def get_volumes(directory=base_volume_dir):
    items = os.listdir(directory)
    volume_numbers = [int(re.search(r'\d+', item).group()) for item in items if re.search(r'\d+', item)]
    return sorted(volume_numbers)


# get all volomes's features on dir
def get_all(dir=base_volume_dir):
    # csv_file = "conventional_features.csv"
    csv_file = os.path.join(dir , 'conventional_features.csv')
    volume_id_list=get_volumes(dir)
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Volume_ID", "Max_Tumor_Area", "Max_Tumor_Diameter",  "Avg_Outer_Layer_Involvement"])
        for i in volume_id_list: 
            # volume_path = os.path.join(base_volume_dir, f'volume_{i}')
            volume_path = os.path.join(dir, f'volume_{i}')
            process_volume(volume_path, writer)
    print("Data processing complete. Results saved to:", csv_file)


#print(process_volume(volume_dir=base_volume_dir))

# csv_file = "conventional_3.csv"
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Volume_ID", "Max_Tumor_Area", "Max_Tumor_Diameter",  "Avg_Outer_Layer_Involvement"])
#     for i in range(1, 370):  # Assuming volumes are numbered from 1 to 369
#         volume_path = os.path.join(base_volume_dir, f'volume_{i}')
#         print(volume_path)
#         process_volume(volume_path, writer)

