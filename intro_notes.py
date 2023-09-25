import cv2
import numpy as np

def image_creation() :

    # Shape of image is: (480, 640, 3)
    image = np.zeros((480, 640, 3), dtype="uint8")
    #another_image = np.copy(image)

    # Note this will drop channel dimension...
    # Shape of gray is (480, 640)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # To add the dimension back
    gray = np.expand_dims(gray, axis=-1)

    # To remove the dimension again
    gray = np.squeeze(gray, axis=-1)

    # To convert back to color, (information is still lost)
    new_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def windows_display():

    """ 
    important functions:
        cv2.imshow - open/display image
            - cv2.imshow(windowName, image)
        cv2.namedWindow - if you want to create a window ahead of time
            - cv2.namedWindow(windowName)
            - windowName must match the name given in imshow
        cv2.waitKey - in order for the program to process window events
            cv2.waitKey(time)
                - Process window events
                - Returns by user (-1 if nothing)
                - If time is > 0 -> wait(time) if milliseconds
                - If time is <= 0 -> wait forever
        cv2.destroyWindow - destroys window given by windowName
            - cv2.destroyWindow(windowName)
            - also cv2.destroyWindows()
    """

    image = np.zeros((480, 640, 3), dtype="uint8")
    image[:,:] = 255
    image[10:40,50:100] = (0,255,0)

    windowName = "Testing Window"
    cv2.namedWindow(windowName)

    cv2.imshow(windowName, image)
    cv2.waitKey(-1)

    print("goodbye") 
    cv2.destroyAllWindows()

#windows_display()

def processing():
    """
    8-bit unsigned integers only contain values 0->255
        -> Typically anything >255 will overflow!
    Usually we do calculations with float/double arrays
    To convert back to 8bit use cv2.convertScaleAbs
        - This takes absolute value
        - Clamps values between [0,255]
        - Converts type to uint8
    
    Example:
        # image is float32 array
        uintImage = cv2.convertScaleAbs(image)
        #uintImage is now uint8
    To convert to a different type in general:
        floatImage = uintImage.astype("float32")
    """
    filename = "seniorphoto.jpg"
    output = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow("SeniorPhoto")

    minValX = 100
    minValY = 200

    output = np.where(output<=128, 0, output)
    output = np.where(output>128, 255, output)

    cv2.imshow("SeniorPhoto", output)
    cv2.waitKey(-1)
    
    
    cv2.destroyAllWindows()

def resizing():
    #cv2.resize(src, dsize, fx, fy,interpolation)
    image = cv2.imread("myeyes.png")

    small_image = cv2.resize(image, dsize=(0,0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
    resized_small_image = cv2.resize(small_image, dsize=(0,0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

    cv2.imshow("original", image)
    cv2.imshow("new", resized_small_image)

    cv2.waitKey(-1)
    cv2.destroyAllWindows()

resizing()
