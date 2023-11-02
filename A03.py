"""
Author: John Pertell
Date:   11.02.2023
Desc:   This program is made to detect white blood cells from a BCCD dataset.
        The function find_WBC returns a list of bounding boxes around the cell.
"""

import numpy as np
import cv2

# Code taken from Thresh.py, slightly modified
def get_connected_image(thresh_image):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN,
                                    element,iterations=1)
    centers = [
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,0],
            [0,255,255],
            [255,255,255]
        ]     
    
    num_components, label_image = cv2.connectedComponents(thresh_image,
                                                          connectivity=8,
                                                          ltype=cv2.CV_32S)
    
    output = np.zeros(thresh_image.shape + (3,), dtype="uint8")
    
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            label = label_image[row,col]
            if label > 0:
                label -= 1
                label %= len(centers)
                output[row,col] = centers[label]
                
    return num_components, output, label_image

# Code taken from Thresh.py, slightly modified
def do_kmeans(image):
    image_shape = image.shape
    image = np.reshape(image, (-1, 3)).astype("float32")
    _, bestLabels, centers = cv2.kmeans(image,
                                            K=5,
                                            bestLabels=None,
                                            criteria=(
                                                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                10, 1.0 
                                            ),
                                            attempts=10,
                                            flags=cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    output = centers[bestLabels.flatten()]
    output = np.reshape(output, image_shape)
    return output

# This function will take a grayscale image and seperate
# the foreground from the background via a threshold value
def binary_threshold(gray_image, thresh=237, invert_image=True):

    # An inverted image will swap light/dark pixels
    if invert_image: inverted_bin_image = 255 - gray_image

    # Iterate through the image's pixels, if the pixels are
    # less than the thresh, set to black(background), 
    # else set to white(foreground)
    for row in range(inverted_bin_image.shape[0]):
        for col in range(inverted_bin_image.shape[1]):
            if inverted_bin_image[row, col] <= thresh:
                inverted_bin_image[row, col] = 0
            else:
                inverted_bin_image[row, col] = 255
    return inverted_bin_image
    

# This function prepares an image for white blood cell retrieval.
# It utilizes the features of other functions in this script.
def prepare_image(image):
    
    # Grab the kmeans image, convert it to gray scale
    kmean_image = do_kmeans(image)
    gray_image = cv2.cvtColor(kmean_image, cv2.COLOR_BGR2GRAY)

    # Transform the image via a histogram, then threshold to seperate
    # foreground from background.
    gray_image = cv2.equalizeHist(gray_image)
    binary_image = binary_threshold(gray_image)
    
    # Do Morphological Operations to seperate/fill blobs
    kernel = np.ones((5, 5), np.uint8)
    morph_iter =  4 
    binary_image = cv2.erode(binary_image, kernel, iterations=morph_iter) 
    binary_image = cv2.dilate(binary_image, kernel, iterations=morph_iter+3)

    num_components, component_image, label_image = get_connected_image(binary_image)
    return component_image, num_components, label_image

# This function checks if the provided boxes are overlapped(sensitivity determined by threshold)
# if it is deemed neccessary, this function will combine the bounding boxes.
def merge_boxes(boxes, overlap_threshold=0.7):
    merged_boxes = []

    # Iterating through all boxes
    while len(boxes) > 0:
        current_box = boxes[0]
        x1, y1, x2, y2 = current_box
        remaining_boxes = []

        # Grabbing min/max values from boxes
        for box in boxes[1:]:
            x1_new, y1_new, x2_new, y2_new = box
            x_min = min(x1, x1_new)
            y_min = min(y1, y1_new)
            x_max = max(x2, x2_new)
            y_max = max(y2, y2_new)
            
            # Determines intersection info(iou)
            intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2_new - x1_new) * (y2_new - y1_new)
            iou = intersection_area / (area1 + area2 - intersection_area)

            # Should the merge be completed? Is it within the threshold?
            if iou >= overlap_threshold:
                x1, y1, x2, y2 = min(x1, x1_new), min(y1, y1_new), max(x2, x2_new), max(y2, y2_new)
            else:
                remaining_boxes.append(box)

        merged_boxes.append((x1, y1, x2, y2))
        boxes = remaining_boxes

    # Return the new list of merged bounding_boxes
    return merged_boxes


# This function will return a list of bounding boxes that is around a white blood cell
# in a BCCD image.
def find_WBC(image):
    # Prepare the image and get the declare bounding_boxes
    _, num_components, label_image  = prepare_image(image)
    bounding_boxes  = []
    
    # Find the coordinates of the white pixels in the image
    for label in range(1, num_components):
        white_pixel_coordinates = np.column_stack(np.where(label_image == label))
        if len(white_pixel_coordinates) == 0:
            continue

        # Let's scale the box size. This is because we are
        # searching for the purple dots in the cell. In reality
        # the cell is bigger, so I found that a slight increase 
        # in box size helps improve numbers.
        box_scaler = 3
        x_min = np.min(white_pixel_coordinates[:, 0]) - box_scaler
        y_min = np.min(white_pixel_coordinates[:, 1]) - box_scaler
        x_max = np.max(white_pixel_coordinates[:, 0]) + box_scaler
        y_max = np.max(white_pixel_coordinates[:, 1]) + box_scaler

        # Basically if the box area is smaller then min_area
        # it is probably noise, safe enough to ignore it.
        area =  ((x_max - x_min) * (y_max - y_min))
        constrain_area = True
        if constrain_area:
            min_area = 4000 # 3700
            if area >= min_area: 
                box = (x_min, y_min, x_max, y_max)
                bounding_boxes.append(box)
        
    # Merging bounding boxe, return the merged list
    bounding_boxes = merge_boxes(bounding_boxes)
    return bounding_boxes


# This main function is mostly used for debugging. It is meant to take a bccd image
# and draw a bounding box around a white blood cell. This script is meant for use with
# Train_A03.py, where it calls find_WBC() and calculates accuracy. 
def main():
        filename = "C:\\Users\\Johnny\\Documents\\Suny_Poly\\Coursework\\FALL2023\\CS470\\CS_470_PERTELJ\\CS_470_PERTELJ\\bccd.jpg"
        #filename = "C:\\Users\\Johnny\\Documents\\Suny_Poly\\Coursework\\FALL2023\\CS470\\CS_470_PERTELJ\\CS_470_PERTELJ\\assign03\\output_wbc\\TRAIN_123.png"
        image = cv2.imread(filename)
        print("Loading image:", filename)

        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        bounding_boxes = find_WBC(image)
        print(bounding_boxes)
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)
            print("BOX: " ,box, " added to image!")
        cv2.imshow("PYTHON", image)
        
        cv2.waitKey(-1)
        cv2.destroyAllWindows()


if __name__ == "__main__": 
    main()