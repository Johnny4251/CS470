import numpy as np
import cv2
import gradio as gr


def read_kernel_file(filepath):
    with open(filepath, 'r') as f:
        file_content = f.read()

    tokens = file_content.split()

    row_count = int(tokens[0])
    column_count = int(tokens[1])

    kernel_values = np.zeros((row_count, column_count))

    i = 2
    for r in range(row_count):
        for c in range(column_count):
            kernel_values[r][c] = tokens[i]
            i = i+1

    print("Filter: " + file_content)
    print("Kernel Size: " + str(row_count) + "x" + str(column_count))
    print(kernel_values)
    
    return kernel_values

def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    pass

def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val)
    return output_img

def main():

    read_kernel_file("C:\\Users\Johnny\\Documents\\Suny_Poly\\Coursework\\FALL2023\CS470\\CS_470_PERTELJ\\CS_470_PERTELJ\\assign02\\filters\\Filter_001.txt")

    """
    demo = gr.Interface(fn=filtering_callback,
    inputs=["image",
    "file",
    gr.Number(value=0.125),
    gr.Number(value=127)],
    outputs=["image"])
    demo.launch() 
    """
    

# Later, at the bottom
if __name__ == "__main__":
    main()
