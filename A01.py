import cv2
import numpy as np
import gradio as gr

def create_unnormalized(image):
    pass

def normalize_hist(hist):
    pass

def create_cdf(nhist):
    pass

def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    pass

def do_histogram_equalize(image, do_stretching):
    return image
    pass


def intensity_callback(input_img, do_stretching):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = do_histogram_equalize(input_img, do_stretching)
    return output_img

def main():
    demo = gr.Interface(fn=intensity_callback,
                inputs=["image", "checkbox"],
                outputs=["image"])
    demo.launch()

if __name__ == "__main__":
    main()