import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

"""
This function will take in an image as a parameter and return a
unormalized histogram of said image. This is assuming the image
is a gray scale image of shape(height, width) & dtype=uint8
"""
def create_unnormalized_hist(image):
    hist = np.zeros(256, dtype="float32")

    for row in image:
        for pixel_val in row:
            hist[pixel_val] += 1

    return hist

def normalize_hist(hist):
    total_count = np.sum(hist)
    normalized_hist = hist / total_count
    
    return normalized_hist

def create_cdf(nhist):
    cdf = np.zeros(256, dtype="float32")
    
    cdf[0] = nhist[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + nhist[i]

    return cdf

def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    hist = create_unnormalized_hist(image)
    nhist = normalize_hist(hist)
    cdf = create_cdf(nhist)
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

    output = np.copy(image)
    transform_func = get_hist_equalize_transform(image, do_stretching)

    height, width = image.shape
    for rows in range(height):
        for columns in range(width):
            value = image[rows, columns]
            new_value = value * transform_func[value]
            output[rows, columns] = new_value
    return output


def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():

    test_image_path = "assign01\images\image01.png"
    image = cv2.imread(test_image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = do_histogram_equalize(gray, do_stretching=True)
    #hist = create_unnormalized(image)
    #normalized_hist = normalize_hist(hist)
    #cdf = create_cdf(normalized_hist)

    cv2.imshow("WINDOW", output)

    waitKey = cv2.waitKey()

    # Plot the histogram
    #plt.bar(range(256), cdf, width=1.0, color='b')

    # Add labels and title
    #plt.xlabel('Pixel Value')
    #plt.ylabel('Count')
    #plt.title('Unnormalized Histogram')

    # Show the plot
    #plt.show()

    """
    demo = gr.Interface(fn=intensity_callback,
                inputs=["image", "checkbox"],
                outputs=["image"])
    demo.launch()
    """
    

if __name__ == "__main__":
    main()