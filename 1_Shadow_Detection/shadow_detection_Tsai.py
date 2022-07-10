"""
Paper:
V. J. D. Tsai
A comparative study on shadow compensation of color aerial images in invariant color models
2006

Code:
Nassim REFES
https://github.com/NassimREFES/shadow-detection-algorithms
2022

"""

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt
import math
import copy
import sys
import os
from osgeo import gdal

# --------------------------------------------- #

def bgr_to_ycbcr(img):
    Y = []
    Cb = []
    Cr = []

    rows = img.shape[0]
    cols = img.shape[1]

    b, g, r = cv2.split(img)

    for row in range(rows):
        Y.append([])
        Cb.append([])
        Cr.append([])
        for col in range(cols):
            Y[row].append(r[row][col]*0.257 + g[row][col]*0.504 + b[row][col]*0.098 + 16)
            Cb[row].append(r[row][col]*(-0.148) + g[row][col]*(-0.291) + b[row][col]*0.439 + 128)
            Cr[row].append(r[row][col]*0.439 + g[row][col]*(-0.368) + b[row][col]*(-0.071) + 128)

    return cv2.merge((np.array(np.round(Y), dtype=np.uint8), 
                      np.array(np.round(Cb), dtype=np.uint8), 
                      np.array(np.round(Cr), dtype=np.uint8)))
    return img

# --------------------------------------------- #

def bgr_to_yiq(img):
    Y = []
    I = []
    Q = []

    rows = img.shape[0]
    cols = img.shape[1]

    b, g, r = cv2.split(img)

    for row in range(rows):
        Y.append([])
        I.append([])
        Q.append([])
        for col in range(cols):
            Y[row].append(r[row][col]*0.299 + g[row][col]*0.587 + b[row][col]*0.114)
            I[row].append(r[row][col]*0.596 + g[row][col]*(-0.275) + b[row][col]*(-0.321))
            Q[row].append(r[row][col]*0.212 + g[row][col]*(-0.523) + b[row][col]*0.311)

    return cv2.merge((np.array(np.round(Y), dtype=np.uint8), 
                      np.array(np.round(I), dtype=np.uint8), 
                      np.array(np.round(Q), dtype=np.uint8)))
    return img

# --------------------------------------------- #

def bgr_to_hcv(img):
    V = []
    C = []
    H = []

    rows = img.shape[0]
    cols = img.shape[1]

    b, g, r = cv2.split(img)

    for row in range(rows):
        V.append([])
        C.append([])
        H.append([])
        for col in range(cols):
            if r[row][col] == 0 and g[row][col] == 0 and b[row][col] == 0:
                V[row].append(0)
                H[row].append(0)
                C[row].append(0)
            else:
                V[row].append((1.0/3.0) * (r[row][col] + g[row][col] + b[row][col]))
                if V[row][col] == g[row][col]:
                    H[row].append(0)
                else:
                    H[row].append(math.degrees(math.atan((r[row][col] - b[row][col]) / (math.sqrt(3)*(V[row][col]-g[row][col])))))
                if math.fabs(math.cos(H[row][col])) <= 0.2:
                    C[row].append((r[row][col] - b[row][col]) / (math.sqrt(3)*math.degrees(math.sin(H[row][col]))))
                else:
                    C[row].append((V[row][col]-g[row][col]) / math.degrees(math.cos(H[row][col])))

    return cv2.merge((np.array(np.round(H), dtype=np.uint8), 
                      np.array(np.round(C), dtype=np.uint8), 
                      np.array(np.round(V), dtype=np.uint8)))
    return img

# --------------------------------------------- #

def bgr_to_hsv(img):
    V = []
    S = []
    H = []

    rows = img.shape[0]
    cols = img.shape[1]

    b, g, r = cv2.split(img)

    for row in range(rows):
        V.append([])
        S.append([])
        H.append([])
        for col in range(cols):
            V[row].append((1.0/3.0) * r[row][col] + (1.0/3.0) * g[row][col] + (1.0/3.0) * b[row][col])
            
            if (r[row][col] + g[row][col] + b[row][col]) == 0:
                S[row].append(0)
            else:
                S[row].append(1 - 3.0/(r[row][col] + g[row][col] + b[row][col])*min((r[row][col], g[row][col], b[row][col])))
            
            numer = (1.0/2.0)*(r[row][col]-g[row][col]) + (1.0/2.0)*(r[row][col]-b[row][col])
            denom = math.sqrt((r[row][col]-g[row][col])**2 + (r[row][col]-b[row][col])*(g[row][col]-b[row][col]))

            if denom == 0: 
                H[row].append(0)
            else:
                d = math.degrees(math.acos(numer / denom))
                if b[row][col] <= g[row][col]:
                    H[row].append(d)
                else:
                    H[row].append(360.0 - d)

    return cv2.merge((np.array(np.round(H), dtype=np.uint8), 
                      np.array(np.round(S), dtype=np.uint8), 
                      np.array(np.round(V), dtype=np.uint8)))
    return img

# --------------------------------------------- #

def bgr_to_hsl(img):
    L = []
    V1 = None
    V2 = None
    S = []
    H = []

    rows = img.shape[0]
    cols = img.shape[1]

    b, g, r = cv2.split(img)

    for row in range(rows):
        L.append([])
        S.append([])
        H.append([])
        for col in range(cols):
            L[row].append(r[row][col]*(1.0/3.0) + g[row][col]*(1.0/3.0) + b[row][col]*(1.0/3.0))
            
            x = math.sqrt(6)
            V1 = r[row][col]*(-x)/6.0 + g[row][col]*(-x)/6.0 + b[row][col]*x/3.0
            V2 = r[row][col]*1.0/x + g[row][col]*(-2.0)/x + b[row][col]*0
            S[row].append(math.sqrt(V1**2 + V2**2))
            
            if not (V1 == 0):
                H[row].append(math.degrees(math.atan(V2 / V1)))
            else:
                H[row].append(0)

    return cv2.merge((np.array(np.round(H), dtype=np.uint8), 
                      np.array(np.round(S), dtype=np.uint8), 
                      np.array(np.round(L), dtype=np.uint8)))
    return img

# --------------------------------------------- #

def ratio_spectral_map(img, _type):
    img = img * 1.0 / 255.0

    R = []
    rows = img.shape[0]
    cols = img.shape[1]

    x, y, z = cv2.split(img)

    for row in range(rows):
        R.append([])
        for col in range(cols):
            if _type in ["HSV", "HSL", "HCV"]:
                Le = z[row][col]
                He = x[row][col]
            elif _type in ["YIQ", "YCbCr"]:
                Le = x[row][col]
                He = z[row][col]
            R[row].append( (He + 1) / (Le + 1) )
    return np.array(R, dtype=np.float64)

# --------------------------------------------- #

class OTSU_Algorithm:
    rows = 0
    cols = 0

    uniq_freq = {} # histogramme (uniq , freq)

    seuil = 0
    intra_classe = float('inf')

    def __init__(self, img):
        for i in range(256):
            self.uniq_freq[i] = 0

        self.rows = img.shape[0]
        self.cols = img.shape[1]

        for row in range(self.rows):
            for col in range(self.cols):
                self.uniq_freq[int(img[row][col])] += 1

        self.uniq_freq = sorted(self.uniq_freq.items())
        self.__calcule_seuil()

    def __moyenne(self, unique_frequence, begin_, end_):
        S, T = 0.0, 0.0
        for i, uf in enumerate(unique_frequence):
            if end_ <= i:
                break
            if begin_ <= i:
                S += float(uf[0]) * uf[1]
                T += uf[1]
        return float(S)/T if T > 0 else 0

    def __variance(self, unique_frequence, begin_, end_, moyenne):
        V = 0.0
        S = 0.0
        for i, uf in enumerate(unique_frequence):
            if end_ <= i:
                break
            if begin_ <= i:
                V += float( (float(uf[0])-moyenne)**2 ) * uf[1]
                S += uf[1]
        return float(V) / S if S > 0 else 0

    def __prob(self, unique_frequence, begin_, end_):
        S = 0.0
        for i, uf in enumerate(unique_frequence):
            if end_ <= i:
                break
            if begin_ <= i:
                S += uf[1]
        return S

    def __calcule_seuil(self):
        for i in range(1, len(self.uniq_freq)):
            # classe 1
            u1 = self.__moyenne(self.uniq_freq, 0, i)
            v1 = self.__variance(self.uniq_freq, 0, i, u1)
            p1 = self.__prob(self.uniq_freq, 0, i)

            # classe 2
            u2 = self.__moyenne(self.uniq_freq, i, len(self.uniq_freq))
            v2 = self.__variance(self.uniq_freq, i, len(self.uniq_freq), u2)
            p2 = self.__prob(self.uniq_freq, i, len(self.uniq_freq))

            ic = p1*v1 + p2*v2
            if ic < self.intra_classe:
                self.intra_classe = ic
                self.seuil = self.uniq_freq[i][0]

        return (self.intra_classe, self.seuil)

# --------------------------------------------- #

def segmentation_seuillage(img, T, _min=0, _max=255, inv=False):
    rows = img.shape[0]
    cols = img.shape[1]
    img_seg = np.zeros((rows, cols), dtype=np.uint8)

    r1 = _min
    r2 = _max

    if inv:
        r1 = _max
        r2 = _min
        
    for row in range(rows):
        for col in range(cols):
            if img[row][col] < T:
                img_seg[row][col] = r1
            else:
                img_seg[row][col] = r2   

    return img_seg

# --------------------------------------------- #

def _method(color_inv, _path, kernel_size):
    img = cv2.imread(_path)
    img = img.astype(np.float64)
    res = None
    if color_inv == "HSL":
        res = bgr_to_hsl(img)
    elif color_inv == "HSV":
        res = bgr_to_hsv(img)
    elif color_inv == "HCV":
        res = bgr_to_hcv(img)
    elif color_inv == "YIQ":
        res = bgr_to_yiq(img)
    elif color_inv == "YCbCr":
        res = bgr_to_ycbcr(img)

    ratio = ratio_spectral_map(res, color_inv)

    normlize_ratio = np.zeros((img.shape[0], img.shape[1]))

    cv2.normalize(ratio, normlize_ratio, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    normlize_ratio = np.array(np.array(normlize_ratio)*255, dtype=np.uint8)

    otsu = OTSU_Algorithm(normlize_ratio)
    xotsu, res_xotsu = cv2.threshold(normlize_ratio, 0, 255, type=cv2.THRESH_OTSU)
    #print(color_inv)
    #print(xotsu)
    #print(otsu.seuil)
    #return
    img_seg = segmentation_seuillage(normlize_ratio, otsu.seuil, inv=False)



    
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    #cv2.morphologyEx(img_seg, cv2.MORPH_CLOSE, kernel, img_seg)
    #cv2.morphologyEx(img_seg, cv2.MORPH_OPEN, kernel, img_seg)

    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img_seg, kernel, iterations=2)
    #dilation = cv2.dilate(erosion, kernel, iterations=1)


##    res_str = {
##        0: str(color_inv+"/"+color_inv+"-resultat.jpg"),
##        1: str(color_inv+"/"+color_inv+"-norm_ratio.jpg"),
##        2: str(color_inv+"/"+color_inv+"-segmented.jpg"),
##    }
##
##    if not os.path.exists(color_inv):
##        os.makedirs(color_inv)
##
##    cv2.imwrite(color_inv+"/original.jpg", img)
##    cv2.imwrite(res_str[0], res)
##    cv2.imwrite(res_str[1], normlize_ratio)
##    cv2.imwrite(res_str[2], img_seg)

##    while True:
##        cv2.imshow('img', img)
##        cv2.imshow('res', res)
##        cv2.imshow('ratio', ratio)
##        cv2.imshow('normlize ratio', normlize_ratio)
##        cv2.imshow('normlize ratio seg', img_seg)
##
##        #cv2.imwrite('x.jpg', res)
##
##        k = cv2.waitKey(33)
##        if k==27:
##            break
##        elif k==-1:
##            continue
##        else:
##            print(k)

    return erosion

# --------------------------------------------- #

def main(input_image, folder_path, base_name):
    shadows = _method("HSL", input_image, kernel_size = 2)
##    shadows = _method("HSV", input_image, kernel_size = 2)
##    shadows = _method("HCV", input_image, kernel_size = 2)
##    shadows = _method("YIQ", input_image, kernel_size = 2)
##    shadows = _method("YCbCr", input_image, kernel_size = 2)

    cv2.imwrite(os.path.join(folder_path, base_name + '_Tsai.tif'), shadows)

    dataset = gdal.Open( input_image )
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is not None and geotransform is not None:
        dataset2 = gdal.Open( os.path.join(folder_path, base_name + '_Tsai.tif'), gdal.GA_Update )
        if geotransform is not None and geotransform != (0,1,0,0,0,1):
            dataset2.SetGeoTransform( geotransform )
        if projection is not None and projection != '':
            dataset2.SetProjection( projection )
        gcp_count = dataset.GetGCPCount()
        if gcp_count != 0:
            dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )

    # --------------------------------------------- #

##    img = cv2.imread(file_path)
##    #img = img * 1.0/255
##    res = bgr_to_hsl(img)
##
##    ratio = ratio_spectral_map(res, "HSL")
##    
##    normlize_ratio = np.zeros((img.shape[0], img.shape[1]))
##
##    #rows = img.shape[0]
##    #cols = img.shape[1]
##    #for row in range(rows):
##    #    for col in range(cols):
##    #        normlize_ratio[row][col] = ratio[0][row][col] / ratio[2] * 255
##    #normlize_ratio = np.array(normlize_ratio, dtype=np.uint8)
##
##    #np.set_printoptions(threshold=numpy.nan)
##    cv2.normalize(ratio, normlize_ratio, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
##    
##    # normalize to [0 255] for detection
##    normlize_ratio = np.array(np.array(normlize_ratio)*255, dtype=np.float64)
##    #print(normlize_ratio)
##
##    otsu = OTSU_Algorithm(normlize_ratio)
##    #print(otsu.seuil)
##    img_seg = segmentation_seuillage(normlize_ratio, otsu.seuil, inv=False)
##
##    #plt.imshow(res)
##    #plt.show()
##    while True:
##        cv2.imshow('img', img)
##        cv2.imshow('res', res)
##        cv2.imshow('ratio', ratio)
##        cv2.imshow('normlize ratio', normlize_ratio)
##        cv2.imshow('normlize ratio seg', img_seg)
##
##        #cv2.imwrite('x.jpg', res)
##
##        k = cv2.waitKey(33)
##        if k==27:
##            break
##        elif k==-1:
##            continue
##        else:
##            print(k)


if __name__ == '__main__':
    main(input_image, folder_path, base_name)
