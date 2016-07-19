import numpy as np
from math import ceil, floor

class NearestNeighboorInterpolator2D:
    def __init__(self, source_shape, target_shape):
        h_src, w_src = source_shape
        h_target, w_target = target_shape
        self.target_shape = target_shape

        self.x_array = np.empty(w_target, dtype=np.int32)
        self.y_array = np.empty(h_target, dtype=np.int32)

        for y in range(0, h_target):
            percentage_y = y / (h_target - 1)
            coord_old_space_y = percentage_y * (h_src - 1)
            y0 = floor(coord_old_space_y)
            y1 =  ceil(coord_old_space_y)
            yold = y0 if coord_old_space_y - y0 <= y1 - coord_old_space_y else y1
            self.y_array[y] = yold

        for x in range(0, w_target):
            percentage_x = x / (w_target - 1)
            coord_old_space_x = percentage_x * (w_src - 1)
            x0 = np.floor(coord_old_space_x)
            x1 =  np.ceil(coord_old_space_x)
            xold = x0 if coord_old_space_x - x0 <= x1 - coord_old_space_x else x1
            self.x_array[x] = xold
    
    def interpolate(self, source):
        return source[self.y_array][:,self.x_array][:]



class BilinearInterpolator2D:
    def __init__(self, source_shape, target_shape):
        h_src, w_src = source_shape
        h_target, w_target = target_shape
        self.target_shape = target_shape

        self.x0_array = np.empty(w_target, dtype=np.int32)
        self.y0_array = np.empty(h_target, dtype=np.int32)
        self.x1_array = np.empty(w_target, dtype=np.int32)
        self.y1_array = np.empty(h_target, dtype=np.int32)
        self.px_array = np.empty(w_target, dtype=np.float64)
        self.py_array = np.empty(h_target, dtype=np.float64)

        for y in range(0, h_target):
            percentage_y = y / (h_target - 1)
            coord_old_space_y = percentage_y * (h_src - 1)
            y0 = floor(coord_old_space_y)
            y1 =  ceil(coord_old_space_y)
            py = y1 - coord_old_space_y
            self.y0_array[y] = y0
            self.y1_array[y] = y1
            self.py_array[y] = py

        for x in range(0, w_target):
            percentage_x = x / (w_target - 1)
            coord_old_space_x = percentage_x * (w_src - 1)
            x0 = np.floor(coord_old_space_x)
            x1 =  np.ceil(coord_old_space_x)
            px = x1 - coord_old_space_x 
            self.x0_array[x] = x0
            self.x1_array[x] = x1
            self.px_array[x] = px

        self.pxd_array = np.ones(w_target) - self.px_array
        self.pyh_array = np.ones(h_target) - self.py_array

#    def interpolate(self, source):
#        x_haut_gauche = np.zeros(self.target_shape, dtype=np.float64)
#        x_haut_droite = np.zeros(self.target_shape, dtype=np.float64)
#        x_bas_gauche  = np.zeros(self.target_shape, dtype=np.float64)
#        x_bas_droite  = np.zeros(self.target_shape, dtype=np.float64) 
#
#        x_haut_gauche = source[self.y1_array][:,self.x0_array]
#        x_haut_droite = source[self.y1_array][:,self.x1_array]
#        x_bas_gauche  = source[self.y0_array][:,self.x0_array]
#        x_bas_droite  = source[self.y0_array][:,self.x1_array]
#
#        x_haut_gauche = np.multiply(x_haut_gauche, self.px_array)  
#        x_haut_droite = self.pxd_array 
#        x_bas_gauche  = self.px_array  
#        x_bas_droite  = self.pxd_array 
#
#        x_haut_gauche += x_haut_droite
#        x_haut_gauche *= self.pyh_array[:, np.newaxis]
#        x_bas_gauche  += x_bas_gauche
#        x_bas_gauche  *= self.py_array[np.newaxis, :]
#
#        return x_haut_gauche + x_bas_gauche 
