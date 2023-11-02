import numpy as np
import cv2

# Code taken from Thresh.py
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
    #print("Number of connected components:", num_components)
    
    output = np.zeros(thresh_image.shape + (3,), dtype="uint8")
    
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            label = label_image[row,col]
            if label > 0:
                label -= 1
                label %= len(centers)
                output[row,col] = centers[label]
                
    return num_components, output, label_image

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

def binary_threshold(gray_image, thresh=237):
    inverted_bin_image = 255 - gray_image
    for row in range(inverted_bin_image.shape[0]):
        for col in range(inverted_bin_image.shape[1]):
            if inverted_bin_image[row, col] <= thresh:
                inverted_bin_image[row, col] = 0
            else:
                inverted_bin_image[row, col] = 255
    return inverted_bin_image
    

def prepare_image(image):
    
    kmean_image = do_kmeans(image)
    gray_image = cv2.cvtColor(kmean_image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.equalizeHist(gray_image)
    binary_image = binary_threshold(gray_image)
    
    # Do Morphological Operations to seperate image
    kernel = np.ones((5, 5), np.uint8)
    morph_iter =  4 
    binary_image = cv2.erode(binary_image, kernel, iterations=morph_iter) 
    binary_image = cv2.dilate(binary_image, kernel, iterations=morph_iter+3)

    num_components, component_image, label_image = get_connected_image(binary_image)
    return component_image, num_components, label_image

def find_WBC(image):
    prepared_image = np.copy(image)
    _, num_components, label_image  = prepare_image(prepared_image)
    bounding_boxes  = []
    
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
        # it is probably noise, safe enough to ignore it
        area =  ((x_max - x_min) * (y_max - y_min))
        constrain_area = True
        if constrain_area:
            min_area = 4000 # 3700
            if area >= min_area: 
                box = (x_min, y_min, x_max, y_max)
                bounding_boxes.append(box)
        
    # Merging bounding boxes
    bounding_boxes = merge_boxes(bounding_boxes)
    return bounding_boxes

def merge_boxes(boxes, overlap_threshold=0.7):
    merged_boxes = []

    while len(boxes) > 0:
        current_box = boxes[0]
        x1, y1, x2, y2 = current_box
        remaining_boxes = []

        for box in boxes[1:]:
            x1_new, y1_new, x2_new, y2_new = box
            x_min = min(x1, x1_new)
            y_min = min(y1, y1_new)
            x_max = max(x2, x2_new)
            y_max = max(y2, y2_new)
            
            intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x2_new - x1_new) * (y2_new - y1_new)
            
            iou = intersection_area / (area1 + area2 - intersection_area)

            if iou >= overlap_threshold:
                #print("BRB, gotta merge")
                x1, y1, x2, y2 = min(x1, x1_new), min(y1, y1_new), max(x2, x2_new), max(y2, y2_new)
            else:
                remaining_boxes.append(box)

        merged_boxes.append((x1, y1, x2, y2))
        boxes = remaining_boxes

    return merged_boxes

def main():
        filename = "C:\\Users\\Johnny\\Documents\\Suny_Poly\\Coursework\\FALL2023\\CS470\\CS_470_PERTELJ\\CS_470_PERTELJ\\bccd.jpg"
        #filename = "C:\\Users\\Johnny\\Documents\\Suny_Poly\\Coursework\\FALL2023\\CS470\\CS_470_PERTELJ\\CS_470_PERTELJ\\assign03\\output_wbc\\TRAIN_123.png"
        image = cv2.imread(filename)
        print("Loading image:", filename)

        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        #prepared_image, _, _ = prepare_image(image)
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