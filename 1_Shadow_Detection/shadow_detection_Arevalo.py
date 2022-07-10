# -*- coding: utf-8 -*-

"""
Paper:
V. Arévalo, J. González, and G. Ambrosio
Shadow detection in colour high-resolution satellite images
2008

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
from time import sleep
from threading import Thread
from multiprocessing import Process, Lock, Queue


np.seterr(divide='ignore', invalid='ignore')

lock = Lock()
g_seeds = [[]]

# --------------------------------------------- #

def bgr_to_c1c2c3(img):
    C1 = []
    C2 = []
    C3 = []

    rows = img.shape[0]
    cols = img.shape[1]

    b, g, r = cv2.split(img)

    for row in range(rows):
        C1.append([])
        C2.append([])
        C3.append([])
        for col in range(cols):
            C1[row].append(math.degrees( math.atan( float(r[row][col]) / max(g[row][col], b[row][col]) ))) if max(g[row][col], b[row][col]) != 0 else C1[row].append(0)
            C2[row].append(math.degrees( math.atan( float(g[row][col]) / max(r[row][col], b[row][col]) ))) if max(r[row][col], b[row][col]) != 0 else C2[row].append(0)
            C3[row].append(math.degrees( math.atan( float(b[row][col]) / max(r[row][col], g[row][col]) ))) if max(r[row][col], g[row][col]) != 0 else C3[row].append(0)

    """while True:
        cv2.imshow('C1', np.array(np.round(C1), dtype=np.uint8))
        cv2.imshow('C2', np.array(np.round(C2), dtype=np.uint8))
        cv2.imshow('C3', np.array(np.round(C3), dtype=np.uint8))
        k = cv2.waitKey(33)
        if k==27:
            break
        elif k==-1:
            continue
        else:
            print (k)"""

    return cv2.merge((np.array(np.round(C1), dtype=np.uint8), 
                      np.array(np.round(C2), dtype=np.uint8), 
                      np.array(np.round(C3), dtype=np.uint8)))
    return img

"""
[summary] fenetre
pos = [x, y]
[ pos[0]+debut : pos[0]+fin+1, pos[1]+debut : pos[1]+fin+1 ]
"""

class Seed:
    def __init__(self, elem, pos, val):
        self.pos_noyau = pos # position (x, y) du centre de la fenetre
        self.val_noyau = val

        rows = elem.shape[0]
        cols = elem.shape[1]

        # Self.hist { val : freq, [pos1(x, y), pos2(x, y), pos3(x, y), …] } 
        self.hist = {}

        if rows == 1 and cols == 1: # 1 element
            self.add(pos, elem[0, 0])
        else: # 5x5 window
            i = -2
            for row in range(rows):
                j = -2
                for col in range(cols):
                    self.add((pos[0]+i, pos[1]+j), elem[row, col])
                    #if elem[row, col] in self.hist.keys():
                    #    self.hist[elem[row, col]].append((pos[0]+i, pos[1]+j))
                    #else:
                    #    self.hist[elem[row, col]] = [ (pos[0]+i, pos[1]+j), ]
                    j += 1
                i += 1

        self.__maj_uf()

        # distribution gaussienne associe
        # standard distribution ( valeur par defaut )
        self.moyenne = 0
        self.ecart_type = 1
        self.variance = self.ecart_type**2

        self.moyenne = self.calcule_moyenne()
        self.variance = self.calcule_variance()
        self.ecart_type = self.calcule_ecart_type()

    # mise a jour de l'uniq et freq de l hist
    def __maj_uf(self):
        self.__uniq = np.array(list(self.hist.keys()), dtype=object)
        self.__freq = [len(v) for v in np.array(list(self.hist.values()), dtype=object)]

    def __maj_calcule(self):
        self.moyenne = self.calcule_moyenne()
        self.variance = self.calcule_variance()
        self.ecart_type = self.calcule_ecart_type()

    def calcule_moyenne(self):
        return float(sum(self.__uniq*self.__freq)) / sum(self.__freq)

    def calcule_variance(self):
        return float(sum((( self.__uniq - self.moyenne ) ** 2) * self.__freq)) / sum(self.__freq)

    def calcule_ecart_type(self):
        return np.sqrt(float(sum((( self.__uniq - self.moyenne ) ** 2) * self.__freq)) / sum(self.__freq))

    def add(self, pos, val):
        if val in self.hist.keys():
            if pos not in self.hist[val]:
                self.hist[val].append((pos[0], pos[1]))
        else:
            self.hist[val] = [ (pos[0], pos[1]), ]

class Region(Seed):
    def __init__(self, elem, pos, val):
        Seed.__init__(self, elem, pos, val)

        self.D0 = float(3)
        self.Te = float(0.30)
        self.Tv = float(0.65)
        self.Ts = float(0.002)

        self.region = [self.pos_noyau, ]

    def add_pixel(self, c3, v, s, grad_v, noyau, pos_pixel):
        fin = int(noyau[0]/2)
        debut = -(int(noyau[1]/2))

        row = pos_pixel[0]
        col = pos_pixel[1]

        if grad_v[row, col] < self.Te:
            if np.abs(float(c3[row, col]) - self.moyenne)/float(self.ecart_type) < self.D0:
                if row >= 2 and col >= 2:
                    seed_v = v[row+debut:row+fin+1, col+debut:col+fin+1]
                    seed_s = s[row+debut:row+fin+1, col+debut:col+fin+1]

                    if sum(sum(seed_v))/(noyau[0]*noyau[1]) < self.Tv and sum(sum(seed_s))/(noyau[0]*noyau[1]) > self.Ts:
                        self.region.append(pos_pixel)
                        self.add(pos_pixel, c3[row, col])
                        return True
        return False
    

# seeds utilisé remplace cette contrainte
# un emplacement deja cultivé ne peu etre dans d autre region
def recherche_dans_regions(regions, pos):
    for region in regions:
        if pos in region.region:
            return True
    return False

def croissance_region(seeds, region, c3, v, s, grad_v, noyau):
    global g_seeds
    global lock

    fin = int(noyau[0]/2)
    debut = -(int(noyau[1]/2))

    # commence par la seed window element
    region_tmp = [] #[j for i in region.hist.values() for j in i if j != region.pos_noyau]
    region_tmp_s = {}

    for i in region.hist.values(): 
        for j in i:
            if j != region.pos_noyau:
                region_tmp.append(j)
                ss = str(j[0])+'+'+str(j[1])
                try:
                    region_tmp_s[ss]
                except:
                    region_tmp_s[ss] = 1

    # les autres element autour
    normlized_v = v * 1.0 / 255.0
    normlized_s = s * 1.0 / 255.0
    normlized_grad_v = grad_v * 1.0 / 255.0

    #for r in region_tmp:
    while len(region_tmp):
        r = region_tmp[0]
        
        lock.acquire()
        
        if not g_seeds[r[0], r[1]] :
            if region.add_pixel(c3, normlized_v, normlized_s, normlized_grad_v, noyau, r):
                region._Seed__maj_uf()
                region._Seed__maj_calcule()
                
                seeds[r[0], r[1]] = 255 #1
                g_seeds[r[0], r[1]] = 255 #1

                row = r[0]
                col = r[1]
                for i in range(debut, fin+1):
                    for j in range(debut, fin+1):
                        if i == 0 and j == 0:
                            pass # centre
                        else:
                            if row+i >= 2 and row+i < c3.shape[0] and col+j >= 2 and col+j < c3.shape[1]:
                                ss = str(row+i)+'+'+str(col+j)
                                try:
                                    region_tmp_s[ss]
                                except:
                                    region_tmp_s[ss] = 1
                                    region_tmp.append((row+i,col+j))
                                    
        lock.release()
        ss = str(r[0])+'+'+str(r[1])
        del region_tmp[0]
        del region_tmp_s[ss]

    return seeds, region
            
def selection_seeds(c3, c3_moy, v, s, noyau, ML):
    global g_seeds

    rows = c3.shape[0]
    cols = c3.shape[1]

    normlized_v = v * 1.0 / 255.0
    normlized_s = s * 1.0 / 255.0

    regions = []
    seeds_window = np.zeros((rows, cols), dtype=np.uint8)
    seeds = np.zeros((rows, cols), dtype=np.uint8)
    g_seeds = np.zeros((rows, cols), dtype=np.uint8)
    
    fin = int(noyau[0]/2)
    debut = -(int(noyau[1]/2))

    Tv = float(0.35)
    Ts = float(0.02)

    #centre du noyau
    dx = np.abs(debut)
    dy = np.abs(fin)

    for row in range(dx, rows-dx):
        for col in range(dy, cols-dy):
            seed = c3[row+debut:row+fin+1, col+debut:col+fin+1]
            seed_v = normlized_v[row+debut:row+fin+1, col+debut:col+fin+1]
            seed_s = normlized_s[row+debut:row+fin+1, col+debut:col+fin+1]

            if sum(sum(seed_v))/(noyau[0]*noyau[1]) < Tv and sum(sum(seed_s))/(noyau[0]*noyau[1]) > Ts:
                if ML[row][col]: # est un maxima local
                    # Aucun des pixels de la fenetre ne doit 
                    # appartenir a une autre fenetre de depart 
                    sw = seeds_window[row+debut:row+fin+1, col+debut:col+fin+1]
                    if not (1 in sw):
                        pos = (row, col)
                        selected_region = Region(seed, pos, seed[dx, dy])
                        regions.append(selected_region) # seed window selected as region
                        seeds_window[row+debut:row+fin+1, col+debut:col+fin+1] = 1
                        seeds[row, col] = 255 #1 # pour sequentiel croissance region algo
                        g_seeds[row, col] = 255 #1 # pour parallel croissance region algo
    #while True:
    #    cv2.imshow('seeds window', seeds_window*255)
    #    k = cv2.waitKey(33)
    #    if k==27:
    #        break
    #    elif k==-1:
    #        continue
    #    else:
    #        print (k)
    return [regions, seeds, seeds_window]         

def get_maxima_locaux(Gm, Gv, Gh):
    rows = Gm.shape[0]
    cols = Gm.shape[1]

    ML = np.ones((rows, cols), dtype=np.uint8)

    for row in range(1, rows-1):
        for col in range(1, cols-1):
            if Gv[row][col] >= Gh[row][col]:
                Gm1 = float(Gh[row-1][col]) / Gv[row-1][col] + ( float(Gv[row-1][col+1]-Gh[row-1][col+1]) / Gv[row-1][col+1] )
                Gm2 = float(Gh[row+1][col-1]) / Gv[row+1][col-1] + ( float(Gv[row+1][col]-Gh[row+1][col]) / Gv[row+1][col] )
            if Gh[row][col] > Gv[row][col]:
                Gm1 = float(Gv[row][col+1]) / Gh[row][col+1] + ( float(Gh[row-1][col+1]-Gv[row-1][col+1]) / Gh[row-1][col+1] )
                Gm2 = float(Gv[row][col-1]) / Gh[row][col-1] + ( float(Gh[row+1][col-1]-Gv[row+1][col-1]) / Gh[row+1][col-1] )
            if Gm[row][col] > Gm1 and Gm[row][col] > Gm2:
                ML[row][col] = Gm[row][col]
    
    return ML

def main(input_image, folder_path, base_name):
    global g_seeds
    
    img = cv2.imread(input_image)

    # -----------------------------------------
    # --------- Pre processing stage ----------
    # -----------------------------------------

    res = bgr_to_c1c2c3(img)
    bgr_to_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    # C3 lisse par un noyau 3x3
    noyau = np.ones((3,3), dtype=np.float32) / 9
    c1, c2, c3 = cv2.split(res)
    c3_lisse = cv2.filter2D(c3, -1, noyau)

    # gradient Sobel 3x3
    h, s, v = cv2.split(bgr_to_hsv)

    # gradients horizontaux de V
    v_sobelx = cv2.Sobel(v, cv2.CV_64F, dx=1, dy=0, ksize=3) 

    # gradients verticaux de V
    v_sobely = cv2.Sobel(v, cv2.CV_64F, dx=0, dy=1, ksize=3)

    # amplitude de Gx | Gy
    v_sobelxy = np.round(np.sqrt(v_sobelx**2 + v_sobely**2))
    v_sobelxy = v_sobelxy.astype(np.uint8)

    ML = get_maxima_locaux(v_sobelxy, v_sobely, v_sobelx)

    # -----------------------------------------
    # ------- Shadow detection stage ----------
    # -----------------------------------------

    # la moyenne de l ensemble de l image c3
    c3_lisse = c3_lisse.astype(np.float64)
    rows = c3_lisse.shape[0]
    cols = c3_lisse.shape[1]
    c3_moy = sum(sum(c3_lisse))
    c3_moy = c3_moy / (rows*cols)
    c3_lisse = c3_lisse.astype(np.uint8)

    # la selection des seeds
    regions, seeds, seeds_window = selection_seeds(c3_lisse, c3_moy, v, s, [5, 5], ML)

##    while True:
##        cv2.imshow('img', img)
##        cv2.imshow('c1c2c3', res)
##        cv2.imshow('hsv', bgr_to_hsv)
##        cv2.imshow('c3 lisse', c3_lisse)
##        cv2.imshow('Gx', v_sobelx)
##        cv2.imshow('Gy', v_sobely)
##        cv2.imshow('Gx & Gy', v_sobelxy)
##        cv2.imshow('seeds', seeds)
##        cv2.imshow('seeds', seeds_window)
##
##        k = cv2.waitKey(33)
##        if k==27:
##            break
##        elif k==-1:
##            continue
##        else:
##            print(k)

##    # parallele croissance region
##    my_process = []
##    for i in range(len(regions)):
##        j = i + 32 # crée 32 processus en meme temp
##        if j < len(regions):
##            while i < j:
##                my_process.append(Thread(target=croissance_region, args=(g_seeds, regions[i], c3_lisse, v, s, v_sobelxy, [5, 5])))
##                i = i + 1
##            for p in my_process:
##                p.start()
##            for p in my_process:
##                p.join()
##            del my_process[:]
##        else:
##            my_process.append(Thread(target=croissance_region, args=(g_seeds, regions[i], c3_lisse, v, s, v_sobelxy, [5, 5])))
##            i = i + 1
##    for p in my_process:
##        p.start()
##    for p in my_process:
##        p.join()
##    del my_process[:]

    # sequentielle croissance region
##    for i in range(len(regions)):
##        seeds, regions[i] = croissance_region(seeds, regions[i], c3_lisse, v, s, v_sobelxy, [5, 5])


    # -----------------------------------------
    # ------------- Fill Gap ------------------
    # -----------------------------------------

    # dilatation suivi d erosion
    #elem_struct = np.ones((2,2),np.uint8)
    elem_struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    fermeture_morph = cv2.morphologyEx(seeds, cv2.MORPH_CLOSE, elem_struct)


    # # ------------------------------------------------------------------------------------------------


##    img = img.astype(np.uint8)
##
##    cv2.imwrite("original.jpg", img)
##    cv2.imwrite('c1c2c3.jpg', res)
##    cv2.imwrite('hsv.jpg', bgr_to_hsv)
##    cv2.imwrite('C3 lisse.jpg', c3_lisse)
##    cv2.imwrite('gradient_V.jpg', v_sobelxy)
##    cv2.imwrite('seeds_croissance_region.jpg', g_seeds)
##    cv2.imwrite('seeds.jpg', seeds)
##    cv2.imwrite('seeds_window.jpg', seeds_window*255)
##    cv2.imwrite('gradient_V.jpg', v_sobelxy)
##    cv2.imwrite('fermeture_croissance_region.jpg', fermeture_morph)
##
##    while True:
##        cv2.imshow('img', img)
##        cv2.imshow('c1c2c3', res)
##        cv2.imshow('hsv', bgr_to_hsv)
##        cv2.imshow('c3 lisse', c3_lisse)
##        cv2.imshow('Gx', v_sobelx)
##        cv2.imshow('Gy', v_sobely)
##        cv2.imshow('Gx & Gy', v_sobelxy)
##        cv2.imshow('seeds', seeds)
##        #cv2.imshow('seeds_croissance_region', g_seeds)
##        cv2.imshow('fermeture seeds', fermeture_morph)
##        k = cv2.waitKey(33)
##        if k==27:
##            break
##        elif k==-1:
##            continue
##        else:
##            print(k)

    cv2.imwrite(os.path.join(folder_path, base_name + '_Arevalo.tif'), fermeture_morph) #seeds
    
    dataset = gdal.Open( input_image )
    projection   = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()

    if projection is not None and geotransform is not None:
        dataset2 = gdal.Open( os.path.join(folder_path, base_name + '_Arevalo.tif'), gdal.GA_Update )
        if geotransform is not None and geotransform != (0,1,0,0,0,1):
            dataset2.SetGeoTransform( geotransform )
        if projection is not None and projection != '':
            dataset2.SetProjection( projection )
        gcp_count = dataset.GetGCPCount()
        if gcp_count != 0:
            dataset2.SetGCPs( dataset.GetGCPs(), dataset.GetGCPProjection() )


if __name__ == '__main__':
    main(input_image, folder_path, base_name)
