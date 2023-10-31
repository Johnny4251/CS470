import numpy as np
import cv2

def prepare_image(image):
    
    kimage = np.copy(image)
    image_shape = kimage.shape
    kimage = np.reshape(kimage, (-1, 3)).astype("float32")
    _, bestLabels, centers = cv2.kmeans(kimage,
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
    
    gray_image = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)
    
    #just for testing.. 
    do_thresh = True
    thresh = 245
    gray_image = 255 - gray_image
    if do_thresh:
        for row in range(gray_image.shape[0]):
            for col in range(gray_image.shape[1]):
                if gray_image[row, col] <= thresh:
                    gray_image[row, col] = 0
                else:
                    gray_image[row, col] = 255
    
    kernel = np.ones((10, 5), np.uint8)
    gray_image = cv2.erode(gray_image, kernel, iterations=3)
    gray_image = cv2.dilate(gray_image, kernel, iterations=5)

    return gray_image

def find_WBC(image):
    prepared_image = np.copy(image)
    new_image = prepare_image(prepared_image)

    white_pixel_coordinates = np.column_stack(np.where(new_image == 255))
    if len(white_pixel_coordinates) == 0:
        print("NO WHITE PIXELS FOUND!")
        return []
    
    #print("White Pixel Count: " + str(len(white_pixel_coordinates)))
    x_min = np.min(white_pixel_coordinates[:, 0])
    y_min = np.min(white_pixel_coordinates[:, 1])
    x_max = np.max(white_pixel_coordinates[:, 0])
    y_max = np.max(white_pixel_coordinates[:, 1])

    bounding_boxes = [(x_min, y_min, x_max, y_max)]
    #print(bounding_boxes)
    return bounding_boxes


def main():
        filename = "C:\\Users\\Johnny\\Documents\\Suny_Poly\\Coursework\\FALL2023\\CS470\\CS_470_PERTELJ\\CS_470_PERTELJ\\bccd.jpg"
        filename = "C:\\Users\\Johnny\\Documents\\Suny_Poly\\Coursework\\FALL2023\\CS470\\CS_470_PERTELJ\\CS_470_PERTELJ\\assign03\\output_wbc\\TEST_044.png"
        image = cv2.imread(filename)
        print("Loading image:", filename)

        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        prepared_image = prepare_image(image)
        cv2.imshow("Prepared Image", prepared_image)

        bounding_boxes = find_WBC(image)
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(image, (y_min, x_min), (y_max, x_max), (0, 0, 255), 2)
        cv2.imshow("PYTHON", image)
        
        cv2.waitKey(-1)
        cv2.destroyAllWindows()


if __name__ == "__main__": 
    main()