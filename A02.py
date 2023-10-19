"""
Author: John Pertell
Date:   10.18.23
Desc:   This Python application uses OpenCV for image filtering. The function
        read_kernel_file takes in a kernel file and returns the corresponding
        numpy array. The apply_filter function returns calculates an output image
        based on an input and a kernel. It then returns the new filtered image.
"""
import numpy as np
import cv2
import gradio as gr

# Calculate the kernel based on a kernel file(filepath)
# Return the new kernel
def read_kernel_file(filepath):
    # Read file and tokenize using ' ' as a delimiter 
    with open(filepath, 'r') as f:
        file_content = f.read()
    tokens = file_content.split()

    # The row & column count is determined as the
    # first 2 numbers of the array. 
    row_count = int(tokens[0])
    column_count = int(tokens[1])

    # Create a numpy array of zeros w/shape determined
    # by first two values of filter file
    kernel = np.zeros((row_count, column_count))

    # Calculate the kernel, which is based on the token
    # values. This goes into corresponding index on kernel array
    i = 2 # starting at 2
    for r in range(row_count):
        for c in range(column_count):
            kernel[r][c] = tokens[i]
            i = i+1

    print("Filter: " + file_content)
    print("Kernel Size: " + str(row_count) + "x" + str(column_count))
    print(kernel)
    
    # Return the calculated kernel
    return kernel

# Apply filter based on image and kernel values
# Return a new filtered image
def apply_filter(image, kernel, alpha=0.125, beta=127, convert_uint8=True):

    # Casting to float
    image = image.astype("float64")
    kernel = kernel.astype("float64")

    # Flipping so we perform convolution
    kernel = cv2.flip(kernel, -1)
    
    # Creating a padding_image 
    padding_height, padding_width = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded_image = cv2.copyMakeBorder(image, 
                                      padding_height, padding_height, 
                                      padding_width, padding_width,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)

    # define the output as a numpy array
    output = np.zeros((image.shape[0], image.shape[1]), dtype="float64")

    # Traversing the image, find pixels and perform
    # calculations based on kernel values
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            subImage = padded_image[ row : (row + kernel.shape[0]), col : (col + kernel.shape[1]) ]
            filtervals = subImage * kernel
            value = np.sum(filtervals)
            output[row,col] = value

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    return output # filtered image

def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)
    return output_img

def main():

    demo = gr.Interface(fn=filtering_callback,
                        inputs=["image",
                            "file",
                            gr.Number(value=0.125),
                            gr.Number(value=127)],
                        outputs=["image"])
    demo.launch() 
    

if __name__ == "__main__":
    main()
