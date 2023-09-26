"""
Author: John Pertell
Date:   10.4.23
Desc:   This Python application uses OpenCV for histogram equalization on images.
        The app contains functions to create an unnormalized histogram, a normalized histogram, 
        a cumulative distribution function (CDF), and perform image transformation.
"""

import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

def create_unnormalized_hist(image):
    """
    Calculate an unnormalized histogram for a grayscale image.

    Param:
    - image: A grayscale image represented as an array of shape (height, width).

    Returns:
    - hist: An unnormalized histogram as a numpy array of size 256.
    """
    
    hist = np.zeros(256, dtype="float32")   # Initializing the histogram

    # Count pixel occurrences in the histogram
    for row in image:
        for pixel_val in row:
            hist[pixel_val] += 1

    return hist


def normalize_hist(hist):
    """
    Calculate the normalized histogram given an unnormalized histogram.

    Param:
    - hist: A histogram calculated from a grayscale image

    Returns:
    - normalized_hist: A normalized histogram calculated from hist
    """

    # Normalize the histogram by dividing each bin count by the total pixel count
    total_count = np.sum(hist)
    normalized_hist = hist / total_count
    
    return normalized_hist

"""
This function takes in a normalized hist and then calculates the cdf based off of the input.
It then returns the CDF found as a numpy array.
"""
def create_cdf(nhist):
    """
    Calculate the cdf from a normalized histogram.

    Param: 
    - nhist: A normalized histogram calculated from normaized_hist()

    Returns:
    - cdf: The cdf of nhist
    """

    # Initalize the cdf numpy array
    cdf = np.zeros(256, dtype="float32")
    
    # Calculate the CDF from the normalized histogram
    cdf[0] = nhist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + nhist[i]

    return cdf

def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    """
    Gets the hist/equalize/transform from an image. This function also 
    provides the option to perform stretching. 

    Param: 
    - image: The image to perform histogram calculations on.
    - do_stretching: Gives the option to the end user to perform stretching on the histogram.

    Returns:
    - int_transform: The transform of the cdf.
    """

    # Create histogram, normalize it, get the CDF
    hist = create_unnormalized_hist(image)
    nhist = normalize_hist(hist)
    cdf = create_cdf(nhist)

    # If histogram stretching is enabled, adjust the CDF values for stretching
    if do_stretching:
        min_val = cdf[0]
        for i in range(len(cdf)):
            cdf[i] = cdf[i] - min_val

        max_val = cdf[255]
        for i in range(len(cdf)):
            cdf[i] = cdf[i] / max_val

    int_transform = cdf * 255.0
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]

    return int_transform

def do_histogram_equalize(image, do_stretching):
    """
    Creates, normalizes, equalizes, a histogram and provides histogram stretching if asked for. 
    This function will return an output image from the calculations.

    Param: 
    - image: The image to perform histogram calculations on.
    - do_stretching: Gives the option to the end user to perform stretching on the histogram.

    Returns:
    - output: The output image after histogram calculations and transformation applied.
    """

    # Initalize the output image and grab the transformation function
    output = np.copy(image)
    transform_func = get_hist_equalize_transform(image, do_stretching)

    # Traversing image, grabbing value and transforming pixels
    height, width = image.shape
    for rows in range(height):
        for columns in range(width):
            value = image[rows, columns]
            new_value = transform_func[value]
            output[rows, columns] = new_value
    return output


def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    # Setting up and launching a GUI enviroment for image histogram equalization
    demo = gr.Interface(fn=intensity_callback,
                inputs=["image", "checkbox"],
                outputs=["image"])
    demo.launch()
    

if __name__ == "__main__":
    main()