import numpy as np
import cv2
import General_A04 as g4

def getOneLBPLabel(subimage, label_type):
    if label_type == g4.LBP_LABEL_TYPES.UNIFORM:
        
        # Computing the neighbor values given a 3x3 subimage

        top_left = subimage[0,0]
        top_center = subimage[0,1]
        top_right = subimage[0,2]

        mid_left = subimage[1,0]
        center = subimage[1,1]
        mid_right = subimage[1,2]

        bot_left = subimage[2,0]
        bot_center = subimage[2,1]
        bot_right = subimage[2,2]

        # Compute label based off neighbors
        label = [top_left, top_center, top_right, 
                 mid_left, mid_right, 
                 bot_left, bot_center, bot_right]

        bin_label = [0] * len(label)
        # Make the lbp label binary using center value as threshold
        for i in range(len(label)):
            if np.all(label[i] > center):
                bin_label[i] = 1
            else:
                bin_label[i] = 0

        m = 0
        for i in range(len(bin_label)):
            print("Curr: ",bin_label[i], "Prev: ", bin_label[i-1])
            if bin_label[i] != bin_label[i-1]:
                print("INCREMENTING M")
                m +=1

        uniform_label = 0
        if m <= 2:
            uniform_label = sum(bin_label)
        elif m > 2:
            uniform_label = 9

        #print("SUBIMAGE: ", subimage)
        #return bin_label, m, uniform_label
        return uniform_label
    else:
        print("other label types are not supported")
        print("leaving function...")

def getLBPImage(image, label_type):
    
    padded_image = cv2.copyMakeBorder(image, 
                                      1, 1, 
                                      1, 1,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=0)
    lbp_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            
            top_left = padded_image[r-1, c-1]
            top_center = padded_image[r-1, c]
            top_right = padded_image[r-1, c+1]

            mid_left = padded_image[r, c-1]
            center_pixel = padded_image[r,c]
            mid_right = padded_image[r, c+1]

            bot_left = padded_image[r+1, c-1]
            bot_center = padded_image[r+1, c]
            bot_right = padded_image[r+1, c+1]

            subimage = np.array([[top_left, top_center, top_right],
                        [mid_left, center_pixel, mid_right],
                        [bot_left, bot_center, bot_right]])
            label = getOneLBPLabel(subimage, label_type)
            #print(label)
            lbp_image[r,c] = label
    return lbp_image

def main():
    subimage = np.array([[0, 1, 1],
                         [0, 0, 1],
                         [0, 0, 0]])
    lbp = getOneLBPLabel(subimage, g4.LBP_LABEL_TYPES.UNIFORM)
    print(lbp)
#    bin_label, m, uniform_label = getOneLBPLabel(subimage, g4.LBP_LABEL_TYPES.UNIFORM)
#    print("bin_label: ", bin_label)
 #   print("m: ", m)
 #   print("U_label: ", uniform_label)

    """
    image = cv2.imread("assign04\images\Image1.png")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = getLBPImage(image, g4.LBP_LABEL_TYPES.UNIFORM)

    print("READY!")
    cv2.imshow("window", lbp_image)
    cv2.waitKey(-1)
    
    cv2.destroyAllWindows()
    """

if __name__ == "__main__":
    main()