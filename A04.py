"""
Author: John Pertell
Date:   11.20.23
Desc:   This program is meant for use when computing LBP labels/images. This
        program provides four useful functions to grab LBP labels, get LBP images
        and get histogram features.
"""

import numpy as np
import cv2
import General_A04 as g4

# This function computes one LBP label given a 
# subimage
def getOneLBPLabel(subimage, label_type):
    if label_type == g4.LBP_LABEL_TYPES.UNIFORM:
        
        # Computing the neighbor values given a 3x3 subimage
        top_left = subimage[0,0]
        top_center = subimage[0,1]
        top_right = subimage[0,2]

        mid_left = subimage[1,0]
        center = subimage[1,1]
        mid_right = subimage[1,2]

        bot_left = subimage[2,0]
        bot_center = subimage[2,1]
        bot_right = subimage[2,2]

        # Compute label based off neighbors
        label = [top_center, top_right, mid_right, 
                 bot_right, bot_center, bot_left, 
                 mid_left, top_left]

        bin_label = [0] * len(label)
        # Make the lbp label binary using center value as threshold
        for i in range(len(label)):
            if np.all(label[i] > center):
                bin_label[i] = 1
            else:
                bin_label[i] = 0

        # Counting for bit changes
        m = 0
        for i in range(1, len(bin_label)):
            if bin_label[i] != bin_label[i-1]:
                m +=1

        if m <= 2:
            uniform_label = sum(bin_label) # Count of 1's
        elif m > 2:
            uniform_label = 9

        return uniform_label
    else:
        print("label type is not supported")
        print("leaving function...")
        return None

# This fucntion computes the LBP image given an image
def getLBPImage(image, label_type):
    
    # Pad image with 0's to prevent out of bound erros
    padded_image = cv2.copyMakeBorder(image, 
                                      1, 1, 
                                      1, 1,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)
    lbp_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Traversing the image ->find subimage->get label
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            
            # Find subimage given coordinates
            top_left = padded_image[r, c]
            top_center = padded_image[r, c+1]
            top_right = padded_image[r, c+2]

            mid_left = padded_image[r+1, c]
            center = padded_image[r+1,c+1]
            mid_right = padded_image[r+1, c+2]

            bot_left = padded_image[r+2, c]
            bot_center = padded_image[r+2, c+1]
            bot_right = padded_image[r+2, c+2]

            subimage = np.array([[top_left, top_center, top_right],
                                 [mid_left, center, mid_right],
                                 [bot_left, bot_center, bot_right]])
            
            # Assign pixels to labels
            lbp_image[r,c] = getOneLBPLabel(subimage, label_type)
    return lbp_image

# Compute histogram given a subimage
def getOneRegionLBPFeatures(subImage, label_type):
    if label_type == g4.LBP_LABEL_TYPES.UNIFORM:
        hist = np.zeros(10, dtype="float32")

        for row in subImage:
            for pixel_val in row:
                hist[pixel_val] += 1
        
        total_count = np.sum(hist)
        normalized_hist = hist / total_count
        
        return normalized_hist
    else:
        print("label type is not supported")
        print("leaving function...")
        return None

# Compute each individual histogram given a regionSideCnt
# and a featureImage.
def getLBPFeatures(featureImage, regionSideCnt, label_type):

    if label_type == g4.LBP_LABEL_TYPES.UNIFORM:

        # Using integeer division to cut out each region
        height = featureImage.shape[0] // regionSideCnt
        width = featureImage.shape[1] // regionSideCnt

        image_hist = []

        # Traversing the rows and columns of the regions
        for i in range(regionSideCnt):
            for j in range(regionSideCnt):

                # Determine where the rows/columns start and end
                start_row = i * height
                end_row = start_row + height
                start_col = j * width
                end_col = start_col + width

                # Grab subimage of start->end row/col
                sub_image = featureImage[start_row:end_row, start_col:end_col]
                
                # Computing individual histogram given the new subimage in the region
                hist = getOneRegionLBPFeatures(sub_image, g4.LBP_LABEL_TYPES.UNIFORM)

                image_hist.append(hist)

        image_hist = np.array(image_hist)
        image_hist = np.reshape(image_hist, (image_hist.shape[0]*image_hist.shape[1],))
    
    else:
        print("label type is not supported")
        print("leaving function...")
        return None

    return image_hist
