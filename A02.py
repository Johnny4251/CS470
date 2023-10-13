import numpy as np
import cv2
import gradio as gr


def read_kernel_file(filepath):
    with open(filepath, 'r') as f:
        file_content = f.read()

    tokens = file_content.split()

    row_count = int(tokens[0])
    column_count = int(tokens[1])

    kernel = np.zeros((row_count, column_count))

    i = 2
    for r in range(row_count):
        for c in range(column_count):
            kernel[r][c] = tokens[i]
            i = i+1

    print("Filter: " + file_content)
    print("Kernel Size: " + str(row_count) + "x" + str(column_count))
    print(kernel)
    
    return kernel

def apply_filter(image, kernel, alpha=0.125, beta=127, convert_uint8=True):
    image = image.astype("float64")
    kernel = kernel.astype("float64")

    kernel = cv2.flip(kernel, -1)
    
    padding_height, padding_width = kernel.shape[0] // 2, kernel.shape[1] // 2

    padded_image = cv2.copyMakeBorder(image, 
                                      padding_height, padding_height, 
                                      padding_width, padding_width,
                                      cv2.BORDER_CONSTANT)

    output = np.zeros((image.shape[0], image.shape[1]), dtype="float64")

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            subImage = padded_image[ row : (row + kernel.shape[0]), col : (col + kernel.shape[1]) ]
            filtervals = subImage * kernel
            value = np.sum(filtervals)
            output[row,col] = value

    if convert_uint8:
        output = cv2.convertScaleAbs(output, alpha=alpha, beta=beta)

    return output

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
