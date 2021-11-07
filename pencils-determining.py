import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from scipy.ndimage import label
from skimage.filters import threshold_otsu, gaussian
from skimage import color
from skimage.exposure import adjust_sigmoid
from skimage.transform import resize

def check(image, y, x): 
    if not 0 <= x < image.shape[1]: 
        return False 
    if not 0 <= y < image.shape[0]: 
        return False 
    if image[y, x] != 0: 
        return True 
    return False 
 
def neighbours4(image, y, x): 
    left = y, x - 1 
    top = y - 1, x 
    right = y, x + 1 
    down = y + 1, x 
    if not check(image, *left): 
        left = None, None 
    if not check(image, *top): 
        top = None, None 
    if not check(image, *right): 
        right = None, None 
    if not check(image, *down): 
        down = None, None 
    return left, top, right, down 
 
def area(LB, label = 1): 
    pxs = np.where(LB == label) 
    return len(pxs[0]) 
 
def centroid(LB, label = 1): 
    pxs = np.where(LB == label) 
    return pxs[0].mean(), pxs[1].mean() # y first, x second 
 
def get_boundaries(LB, label = 1): 
    pxs = np.where(LB == label) 
    boundaries = [] 
    for y, x in zip(*pxs): 
        for yn, xn, in neighbours4(LB, y, x): 
            if yn == None: 
                boundaries.append((y,x)) 
                break 
            elif xn == None: 
                boundaries.append((y,x)) 
                break 
            elif LB[yn, xn] != label: 
                boundaries.append((y,x)) 
                break 
    return boundaries
 
def perimeter(LB, label = 1): 
    return len(get_boundaries(LB, label)) 

if __name__ == "__main__":
    path = './imgs/images/'
    files = [file for file in listdir(path)]
    pencils = 0

    for file in files:
        image = plt.imread(path + file)
        gray = color.rgb2gray(image)
        gray = resize(gray, (gray.shape[0]//10, gray.shape[1]//10))
        gray = adjust_sigmoid(gray, cutoff=10, gain=3)
        gray = gaussian(gray, sigma=3)

        binary = gray.copy()
        binary[gray >= threshold_otsu(gray)] = 0
        binary[binary > 0] = 1

        labeled = label(binary)[0]
        #lc = labeled.copy()

        for i in range(1, np.max(labeled) + 1):
            figure_area = area(labeled, i)
            figure_perimeter = perimeter(labeled, i)
            area_perimeter_div = figure_area / figure_perimeter
            if 6 < area_perimeter_div < 10 and 3000 < figure_area < 5000:
                #labeled[labeled == i] = 255
                pencils += 1
            #print(f"{i} -> area : {figure_area}, per : {figure_perimeter}, a/p : {area_perimeter_div}")

        # plt.subplot(221)
        # plt.imshow(image)
        # plt.subplot(222)
        # plt.imshow(gray, cmap='gray')
        # plt.subplot(223)
        # plt.imshow(lc, cmap='gray')
        # plt.subplot(224)
        # plt.imshow(labeled, cmap='gray')
        # plt.show()

    print(f"There are {pencils} pencils on the images")