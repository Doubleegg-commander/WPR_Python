import numpy as np
import xarray as xr
import os
import warnings
import pandas as pd

'''
基本函数
'''
def get_array_mode(arr):
    mode=pd.Series(arr.reshape(-1)).mode()[0]

    return mode

'''
双权重估计
'''
class bi_weight_estimation():
    def __init__(self,vertical_velocity_arr):
        self.w_speed=vertical_velocity_arr

    def biweight_mean(self):
        mean_arr=np.ones(self.w_speed.shape)


        return mean_arr

    def biweight_std(self):

        return std_arr

    def biweight_skewness(self):

        return skeness_arr