import numpy as np
import math
import pandas as pd
import xarray as xr
import netCDF4 as nc
import datetime
from .RWP_READ import *
import warnings
from scipy.special import gamma

warnings.filterwarnings('ignore')

'''
基础函数
'''
def Get_model(data_structure):
    file_dims=list(data_structure.dims)
    dim_model=[]

    for dim in file_dims:
        dim_model.append(dim[-1:])
    dim_model=list(set(dim_model))

    return dim_model

def angle_to_radian(angle):
    radian=angle*math.pi/180

    return radian

'''
再处理产品
'''
def get_radfile_model_range_resolution(radar_data,model_name):
    Level_arr=np.array(radar_data['Level_'+model_name[0]])
    range_resolution=(Level_arr[-1]-Level_arr[0])/(Level_arr.shape[0])
    
    return range_resolution

def get_radarfile_vertical_resolution(radar_info,model_name):
    model_info=radar_info.loc[model_name]
    model_pulse_width=float(model_info['脉冲宽度'])
    vertical_resolution=((model_pulse_width)*(10**(-6))*(3*(10)**8))/2
    
    return vertical_resolution

def get_direction_ind(obs_para_frame,standard_seris=['E','S','W','N','R','L']):
    beam_series_sign=list(obs_para_frame['波束顺序标志'][:])    
    series_Arr=np.zeros([len(beam_series_sign),len(standard_seris)])
    for ind in range(len(beam_series_sign)):
        model_beam_series_sign=beam_series_sign[ind]
        for dirt_ind in range(len(standard_seris)):
            dirt=standard_seris[dirt_ind]
            series_Arr[ind,dirt_ind]=model_beam_series_sign.find(dirt)

    return series_Arr     

def muti_beam_calc_wind_product(velocity_x,velocity_y,velocity_w,off_zenith_angle):
    U_Arr=(velocity_x-velocity_w*(math.cos(angle_to_radian(off_zenith_angle))))/(math.sin(angle_to_radian(off_zenith_angle)))
    V_Arr=(velocity_y-velocity_w*(math.cos(angle_to_radian(off_zenith_angle))))/(math.sin(angle_to_radian(off_zenith_angle)))
    W_Arr=velocity_w

    wind_direction=np.zeros(velocity_x.shape)
    for t_ind in range(velocity_x.shape[0]):
        if V_Arr[t_ind]!=0:
            wind_direction[t_ind]=180+(np.arctan(U_Arr[t_ind]/V_Arr[t_ind])/math.pi)*180
        else:
            if U_Arr[t_ind]>0:
                wind_direction[t_ind]=270
            elif U_Arr[t_ind]<0:
                wind_direction[t_ind]=90
    
    horizontal_wind_speed=U_Arr*np.sin(angle_to_radian(wind_direction-180))+V_Arr*np.cos(angle_to_radian(wind_direction-180))

    return U_Arr,V_Arr,W_Arr,wind_direction,horizontal_wind_speed

def get_horizontal_wind_speed(radian_speed,off_zenith_angle,beam_series=None,method=1):
    if method==1:
        horizontal_wind_speed=radian_speed/(math.sin(angle_to_radian(off_zenith_angle)))

        return horizontal_wind_speed
    elif method==3:
        if (beam_series[0]!=-1)&(beam_series[3]!=-1):
            north_ind=int(beam_series[3])
            east_ind=int(beam_series[0])
            centre_ind=int(beam_series[4])

            north_rad_speed=radian_speed[north_ind,:]
            east_rad_speed=radian_speed[east_ind,:]
            centre_rad_speed=radian_speed[centre_ind,:]
            
            velocity_rad_x=east_rad_speed
            velocity_rad_y=north_rad_speed
            velocity_rad_w=centre_rad_speed
            
            U_Arr,V_Arr,W_Arr,wind_direction,horizontal_wind_speed=muti_beam_calc_wind_product(velocity_rad_x,velocity_rad_y,velocity_rad_w,off_zenith_angle)

            return U_Arr,V_Arr,W_Arr,wind_direction,horizontal_wind_speed
        else:
            raise Exception('The Effective Beam Number Is Insufficient')
    elif method==5:
        nan_in_need=np.count_nonzero(beam_series[:-1]<0)
        if nan_in_need==0:
            north_ind=int(beam_series[3])
            south_ind=int(beam_series[1])
            west_ind=int(beam_series[2])
            east_ind=int(beam_series[0])
            centre_ind=int(beam_series[4])

            north_rad_speed=radian_speed[north_ind,:]
            south_rad_spped=radian_speed[south_ind,:]
            west_rad_speed=radian_speed[west_ind,:]
            east_rad_speed=radian_speed[east_ind,:]
            centre_rad_speed=radian_speed[centre_ind,:]

            velocity_rad_x=0.5*(east_rad_speed-west_rad_speed)
            velocity_rad_y=0.5*(north_rad_speed-south_rad_spped)
            velocity_rad_w=centre_rad_speed

            U_Arr,V_Arr,W_Arr,wind_direction,horizontal_wind_speed=muti_beam_calc_wind_product(velocity_rad_x,velocity_rad_y,velocity_rad_w,off_zenith_angle)

            return U_Arr,V_Arr,W_Arr,wind_direction,horizontal_wind_speed
        else:
            raise Exception('The Effective Beam Number Is Insufficient')

def get_wind_shear(horizontal_wind_speed,level_data,beams='Single'):
    wind_shear=np.ones(horizontal_wind_speed.shape)
    wind_shear*=-999.0
    
    for lev_ind in range(level_data.shape[0]-1):
        if beams=='Muti':
            wind_shear[:,lev_ind]=(horizontal_wind_speed[:,lev_ind+1]-horizontal_wind_speed[:,lev_ind])/(level_data[lev_ind+1]-level_data[lev_ind])
        elif beams=='Single':
            wind_shear[lev_ind]=(horizontal_wind_speed[lev_ind+1]-horizontal_wind_speed[lev_ind])/(level_data[lev_ind+1]-level_data[lev_ind])

    return wind_shear

def get_zenith_arr(obs_para_frame):
    model_num=len(obs_para_frame)
    
    zenith_arr=np.zeros([model_num,6])
    for model_ind in range(model_num):
        model_frame=obs_para_frame.iloc[model_ind,:]
        east_zenith=np.float(model_frame['东波束-铅垂线夹角'])
        south_zenith=np.float(model_frame['南波束-铅垂线夹角'])
        west_zenith=np.float(model_frame['西波束-铅垂线夹角'])
        north_zenith=np.float(model_frame['北波束-铅垂线夹角'])
        centre_row_zenith=np.float(model_frame['中波束(行)-铅垂线夹角'])
        centre_col_zenith=np.float(model_frame['中波束(列)-铅垂线夹角'])
        
        model_zenith_arr=np.array([east_zenith,south_zenith,west_zenith,north_zenith,centre_row_zenith,centre_col_zenith])
        zenith_arr[model_ind,:]=model_zenith_arr

    return zenith_arr
    
def calc_wind_product(Rad_file,method=5):
    if RWP_READ.Read_RWP_Data(Rad_file).datatype!='RAD':
        raise Exception('Not a RAD Data File')
    file_data=RWP_READ.Read_RWP_Data(Rad_file).Get_data()
    radar_info,obs_info=RWP_READ.Read_RWP_Data(Rad_file).Get_Radar_RAD_Model_info()
    
    model_name=list(radar_info.index)
    beam_series_arr=get_direction_ind(obs_info)
    wind_data_structure=xr.Dataset()
    if method==1:
        for dim in list(file_data.dims):
            wind_data_structure[dim]=file_data[dim]

        for model_ind in range(len(model_name)):
            model=model_name[model_ind]

            radar_model_info=radar_info.loc[model]
            model_data_dims=tuple(['Beam_num_'+model[0],'Level_'+model[0]])

            off_zenith_angle=np.float(radar_model_info['东波束-铅垂线夹角'])
            horizontal_wind_speed=get_horizontal_wind_speed(np.array(file_data['Radial_velocity_'+model[0]]),off_zenith_angle,method=method)

            levels=np.array(file_data['Level_'+model[0]])
            wind_shear=get_wind_shear(horizontal_wind_speed,levels,beams='Muti')
            wind_data_structure['Wind_Shear_'+model[0]]=(model_data_dims,wind_shear)

        return wind_data_structure
    elif (method==3)|(method==5):
        useful_dims=remove_str('Beam',list(file_data.dims))
        for dim in useful_dims:
            wind_data_structure[dim]=file_data[dim]
        
        for model_ind in range(len(model_name)):
            model=model_name[model_ind]
            beam_series=beam_series_arr[model_ind]
            radar_model_info=radar_info.loc[model]
            model_data_dims=tuple(['Level_'+model[0]])

            off_zenith_angle=np.float(radar_model_info['东波束-铅垂线夹角'])
            U_Arr,V_Arr,W_Arr,wind_direction,horizontal_wind_speed=get_horizontal_wind_speed(np.array(file_data['Radial_velocity_'+model[0]]),off_zenith_angle,beam_series.astype(np.int),method=method)

            levels=np.array(file_data['Level_'+model[0]])
            wind_shear=get_wind_shear(horizontal_wind_speed,levels)
            wind_data_structure['Wind_Shear_'+model[0]]=(model_data_dims,wind_shear)
            wind_data_structure['U_'+model[0]]=(model_data_dims,U_Arr)
            wind_data_structure['V_'+model[0]]=(model_data_dims,V_Arr)
            wind_data_structure['W_'+model[0]]=(model_data_dims,W_Arr)
            wind_data_structure['Wind_Direction_'+model[0]]=(model_data_dims,wind_direction)

        return wind_data_structure
    
def get_noise(Cn_data):
    noise_arr=np.zeros(Cn_data.shape[0])
    for ind in range(Cn_data.shape[0]):
        beam_Cn=Cn_data[ind,:]
        noise_arr[ind]=np.var(beam_Cn[np.where(beam_Cn>0)])

    return noise_arr

def calc_spectral_width_product(Rad_file):
    if RWP_READ.Read_RWP_Data(Rad_file).datatype!='RAD':
        raise Exception('Not a RAD Data File')
    file_data=RWP_READ.Read_RWP_Data(Rad_file).Get_data()
    radar_info,obs_info=RWP_READ.Read_RWP_Data(Rad_file).Get_Radar_RAD_Model_info()
    
    model_name=list(radar_info.index)

    Spectral_structure=xr.Dataset()
    for dim in list(file_data.dims):
        Spectral_structure.coords[dim]=file_data[dim]
    for model in model_name:
        radar_model_info=radar_info.loc[model]
        model_data_dims=tuple(['Beam_num_'+model[0],'Level_'+model[0]])

        off_zenith_angle=np.float(radar_model_info['东波束-铅垂线夹角'])
        half_power_beam_width=np.float(radar_model_info['垂直波束宽度'])
        vertical_resolution=get_radarfile_vertical_resolution(radar_info,model)
        
        Doppler_spectral_width=np.array(file_data['Velocity_spectrum_width_'+model[0]])**2

        theta_1=angle_to_radian(half_power_beam_width)
        sigma_a=theta_1/(4*((math.log(2))**(1/2)))
        horizontal_wind_speed=get_horizontal_wind_speed(np.array(file_data['Radial_velocity_'+model[0]]),off_zenith_angle,method=1)
        Beam_broadening=(sigma_a**2)*(horizontal_wind_speed**2)

        levels=np.array(file_data['Level_'+model[0]])
        wind_shear=get_wind_shear(horizontal_wind_speed,levels,beams='Muti')
        Shear_broadening=np.ones(wind_shear.shape)
        Shear_broadening*=-999.0
        Shear_broadening[:,:-1]=(0.5*abs(wind_shear[:,:-1])*vertical_resolution*math.sin(angle_to_radian(off_zenith_angle)))**2
        
        Data_process_broadening=0.04*Doppler_spectral_width
        
        non_zero_Cn=np.array(file_data['Velocity_spectrum_width_'+model[0]])[np.where(np.array(file_data['Velocity_spectrum_width_'+model[0]])>0)]
        Residual_Noise=get_noise(np.array(file_data['Velocity_spectrum_width_'+model[0]]))

        Turbulent_spectra_width=np.ones(Doppler_spectral_width.shape)
        Turbulent_spectra_width*=-999.0
        for ind in range(Turbulent_spectra_width.shape[0]):
            beam_turbulent_spect_windth=Doppler_spectral_width[ind,:-1]-Beam_broadening[ind,:-1]-Shear_broadening[ind,:-1]-Data_process_broadening[ind,:-1]-Residual_Noise[ind]
            beam_turbulent_spect_windth[np.where(beam_turbulent_spect_windth==0)]=Residual_Noise[ind]
            Turbulent_spectra_width[ind,:-1]=beam_turbulent_spect_windth
            
        Spectral_structure['Doppler_spectral_width_'+model[0]]=(model_data_dims,Doppler_spectral_width)
        Spectral_structure['Beam_broadening_'+model[0]]=(model_data_dims,Beam_broadening)
        Spectral_structure['Shear_broadening_'+model[0]]=(model_data_dims,Shear_broadening)
        Spectral_structure['Turbulent_spectra_width_'+model[0]]=(model_data_dims,Turbulent_spectra_width)

        Spectral_structure['Resodia_Noise_'+model[0]]=(('Beam_num_'+model[0]),Residual_Noise)

    return Spectral_structure

def calc_spherical_integration(model_levels,radar_model_info,obs_model_info,horizontal_wind_speed):
    para_omega=(1/8)*((math.log(2))**(1/2))
    range_resolution=(model_levels[-1]-model_levels[0])/(model_levels.shape[0])
    para_b=para_omega*range_resolution

    half_power_beam_width=np.float(radar_model_info['垂直波束宽度'])
    theta_1=angle_to_radian(half_power_beam_width)
    sigma_a=theta_1/(4*((math.log(2))**(1/2)))

    inter_pulse_period=1/(np.float(radar_model_info['脉冲重复频率']))
    coherent_intergration=np.float(obs_model_info['相干积累'])
    fft_points=np.float(obs_model_info['Fft点数'])
    spectral_average=np.float(obs_model_info['谱平均数'])
    dwell_time=inter_pulse_period*coherent_intergration*fft_points*spectral_average

    model_intergration=np.ones(model_levels.shape)
    model_intergration*=-999.0
    for lev_ind in range(model_levels.shape[0]):
        para_alpha=model_levels[lev_ind]*sigma_a

        para_L=abs(horizontal_wind_speed[lev_ind])*dwell_time

        angle_range=np.arange(1,91,1)
        intergration=0
        for angle_y in angle_range:
            rad_y=angle_to_radian(angle_y)
            for angle_x in angle_range:
                rad_x=angle_to_radian(angle_x)
                intergration+=(math.sin(rad_x)**3)*(((para_b**2)*(math.cos(rad_x)**2)+(para_alpha**2)*(math.sin(rad_x)**2)+((para_L**2)/12)*(math.sin(rad_x)**2)*(math.cos(rad_y)**2))**(1/3))
                
        model_intergration[lev_ind]=intergration*(angle_to_radian(1)**2)
    
    return model_intergration

def calc_parameter_J(model_levels,radar_model_info,obs_model_info,model_horizontal_wind_speed):
    model_beam_num=np.int(radar_model_info['波束数'])
    model_intergration=np.ones([model_beam_num,model_levels.shape[0]])
    model_intergration*=-999.0

    for beam in range(model_beam_num):
        model_intergration[beam,:]=12*gamma(2/3)*calc_spherical_integration(model_levels,radar_model_info,obs_model_info,model_horizontal_wind_speed[beam,:])
     
    model_para_J=12*gamma(2/3)*model_intergration

    return model_para_J

def calc_Eddy_Product(Rad_file):
    Spectral_structure=calc_spectral_width_product(Rad_file)
    radar_info,obs_info=RWP_READ.Read_RWP_Data(Rad_file).Get_Radar_RAD_Model_info()
    radar_data=RWP_READ.Read_RWP_Data(Rad_file).Get_data()
    Spectral_data=calc_spectral_width_product(Rad_file)

    model_name=list(radar_info.index)
    kolmogorov_const=1.6
    para_b=0.6
    para_Richardson_number_flux=0.25
    para_gama=para_Richardson_number_flux/(1-para_Richardson_number_flux)
    

    eddy_structure=xr.Dataset()
    for dim in list(radar_data.dims):
        eddy_structure.coords[dim]=radar_data[dim]

    for model in model_name:
        radar_model_info=radar_info.loc[model]
        obs_model_info=obs_info.loc[model]
        model_data_dims=tuple(['Beam_num_'+model[0],'Level_'+model[0]])

        model_off_zenith_angle=np.float(radar_model_info['东波束-铅垂线夹角'])
        model_horizontal_wind_speed=get_horizontal_wind_speed(np.array(radar_data['Radial_velocity_'+model[0]]),model_off_zenith_angle,method=1)
        model_levels=np.array(radar_data['Level_'+model[0]])

        model_para_J=calc_parameter_J(model_levels,radar_model_info,obs_model_info,model_horizontal_wind_speed)

        model_turb_broadening=np.array(Spectral_data['Turbulent_spectra_width_'+model[0]])
        model_eddy_dissipation_rate=np.ones(model_para_J.shape)
        model_eddy_dissipation_rate*=-999.0
        model_eddy_dissipation_rate[:,:-1]=((model_turb_broadening[:,:-1])**(3/2))*((4*math.pi/kolmogorov_const)**(3/2))*((model_para_J[:,:-1])**(-3/2))
        eddy_structure['Eddy_dissipation_rate_'+model[0]]=(model_data_dims,model_eddy_dissipation_rate)

        model_Brunt_Vaisala_frequency=np.ones(model_para_J.shape)
        model_Brunt_Vaisala_frequency*=-999.0
        model_Brunt_Vaisala_frequency[:,:-1]=(1/para_b)*(model_eddy_dissipation_rate[:,:-1]/model_turb_broadening[:,:-1])
        eddy_structure['Brunt_Vaisala_frequency_'+model[0]]=(model_data_dims,model_Brunt_Vaisala_frequency)

        model_vertical_eddy_diffusivity=np.ones(model_para_J.shape)
        model_vertical_eddy_diffusivity*=-999.0
        model_vertical_eddy_diffusivity[:,:-1]=para_gama*(model_eddy_dissipation_rate[:,:-1])/(model_Brunt_Vaisala_frequency[:,:-1]**2)
        eddy_structure['Vertical_eddy_diffusivity_'+model[0]]=(model_data_dims,model_vertical_eddy_diffusivity)

        model_outer_length_scale=np.ones(model_para_J.shape)
        model_outer_length_scale*=-999.0
        model_outer_length_scale[:,:-1]=(1/0.62)*(2*math.pi)*((model_eddy_dissipation_rate[:,:-1])/(model_Brunt_Vaisala_frequency[:,:-1]**3))**(1/2)     

    return eddy_structure

class RWP_ANALYSIS():
    '''
    再处理产品数据集
    '''
    class Get_Product():
        def __init__(self,file):
            station_num,station_lat,station_lon,station_height,data_time,data_property,radar_type,data_type_description,data_type,file_type=Datafile_info(file)
            self.lon=np.float(station_lon)
            self.lat=np.float(station_lat)
            self.height=np.float(station_height)
            self.time_teg='(BJT)'
            self.time=data_time
            self.datatype=data_type
            self.filename=file

        def Get_RAD_Product(self,GMT=False,method=5):
            station_num,station_lat,station_lon,station_height,data_time,data_property,radar_type,data_type_description,data_type,file_type=Datafile_info(self.filename)
            time_tag='(BJT)'
            if GMT:
                data_time=data_time-datetime.timedelta(hours=8)
                time_tag='(GMT)'

            line1='站号:'+station_num+'('+str(station_lat)+','+str(station_lon)+')'+'   站点高度:'+str(station_height)+'m'+'    观测时间:'+str(data_time)+time_tag
            line2='雷达类型:'+radar_type+'  数据类型:'+data_property+'->'+data_type_description
            data_description=line1+'\n'+line2

            Wind_Product=calc_wind_product(self.filename,method=method)
            Spectral_Product=calc_spectral_width_product(self.filename)
            Eddy_Product=calc_Eddy_Product(self.filename)

            Wind_Product.attrs['Description']=data_description
            Spectral_Product.attrs['Description']=data_description
            Eddy_Product.attrs['Description']=data_description

            return Wind_Product,Spectral_Product,Eddy_Product

    '''
    多文件风廓线雷达数据
    '''
    class Get_Muti_Product():
        def __init__(self,filelist):
            self.timetag='(BJT)'
            self.list=filelist
            self.start_time,self.end_time,self.state,self.timelist=Get_timelist(self,filelist)

            if self.state==1:
                Timelist_warning()

            check_state=file_check(filelist)
            self.check=file_check(filelist)
            if check_state=='error':
                raise Exception('Filelist is Inhomogeneous')
            elif check_state=='correct':
                self.lon=RWP_READ.Read_RWP_Data(filelist[0]).lon
                self.lat=RWP_READ.Read_RWP_Data(filelist[0]).lat
                self.height=RWP_READ.Read_RWP_Data(filelist[0]).height
                self.datatype=RWP_READ.Read_RWP_Data(filelist[0]).datatype

        def Get_Muti_Product_Data(self,GMT=False,visual=False):
            if self.check=='error':
                raise Exception('Filelist is Inhomogeneous')

            if self.datatype=='RAD':
                Wind_Data_Structure=xr.Dataset()
                Spectral_Data_Structure=xr.Dataset()
                Eddy_Data_Structure=xr.Dataset()

                Wind_Data_Structure.coords['Time']=self.timelist
                Spectral_Data_Structure.coords['Time']=self.timelist
                Eddy_Data_Structure.coords['Time']=self.timelist

                Wind_Data_Structure['Ori_Time']=(('Time'),Get_filelist_timeinfo(self.list))
                Spectral_Data_Structure['Ori_Time']=(('Time'),Get_filelist_timeinfo(self.list))
                Eddy_Data_Structure['Ori_Time']=(('Time'),Get_filelist_timeinfo(self.list))

                standard_wind_data,standard_spectral_data,standard_eddy_data=RWP_ANALYSIS.Get_Product(self.list[0]).Get_RAD_Product()

                wind_dims=list(standard_wind_data.dims.keys())
                for dim in wind_dims:
                    Wind_Data_Structure.coords[dim]=standard_wind_data[dim]
            
                wind_varlist=list(set((list(standard_wind_data.variables.keys())))^(set(list(standard_wind_data.dims.keys()))))
                for var in wind_varlist:
                    wind_var_new_dims_name=tuple(left_append('Time',list(standard_wind_data[var].dims)))
                    wind_var_new_dims_shape=append_arr_shape((self.timelist).shape[0],list(np.array(standard_wind_data[var]).shape))
                    wind_var_data=np.zeros(wind_var_new_dims_shape.astype(np.int64))
                    for ind in range(len(self.list)):
                        if visual:
                            print((self.list)[ind])
                        wind_temp_data=calc_wind_product(self.list[ind],method=5)
                        wind_var_data[ind]=wind_temp_data[var]
                    Wind_Data_Structure[var]=(wind_var_new_dims_name,wind_var_data)
            
                spectral_dims=list(standard_spectral_data.dims.keys())
                for dim in spectral_dims:
                    Spectral_Data_Structure.coords[dim]=standard_spectral_data[dim]
            
                Spectral_varlist=list(set((list(standard_spectral_data.variables.keys())))^(set(list(standard_spectral_data.dims.keys()))))
                for var in Spectral_varlist:
                    spectral_var_new_dims_name=tuple(left_append('Time',list(standard_spectral_data[var].dims)))
                    spectral_var_new_dims_shape=append_arr_shape((self.timelist).shape[0],list(np.array(standard_spectral_data[var]).shape))
                    spectral_var_data=np.zeros(spectral_var_new_dims_shape.astype(np.int64))
                    for ind in range(len(self.list)):
                        spectral_temp_data=calc_spectral_width_product(self.list[ind])
                        spectral_var_data[ind]=spectral_temp_data[var]
                    Spectral_Data_Structure[var]=(spectral_var_new_dims_name,spectral_var_data)

                eddy_dims=list(standard_eddy_data.dims.keys())
                for dim in eddy_dims:
                    Eddy_Data_Structure.coords[dim]=standard_eddy_data[dim]
            
                eddy_varlist=list(set((list(standard_eddy_data.variables.keys())))^(set(list(standard_eddy_data.dims.keys()))))
                for var in eddy_varlist:
                    eddy_var_new_dims_name=tuple(left_append('Time',list(standard_eddy_data[var].dims)))
                    eddy_var_new_dims_shape=append_arr_shape((self.timelist).shape[0],list(np.array(standard_eddy_data[var]).shape))
                    eddy_var_data=np.zeros(eddy_var_new_dims_shape.astype(np.int64))
                    for ind in range(len(self.list)):
                        eddy_temp_data=calc_Eddy_Product(self.list[ind])
                        eddy_var_data[ind]=eddy_temp_data[var]
                    Eddy_Data_Structure[var]=(eddy_var_new_dims_name,eddy_var_data)
                
                Wind_Data_Structure.attrs['Description']=standard_wind_data.attrs['Description']
                Spectral_Data_Structure.attrs['Description']=standard_spectral_data.attrs['Description']
                Eddy_Data_Structure.attrs['Description']=standard_eddy_data.attrs['Description']

                Wind_Data_Structure.attrs['Start Time']=str(self.start_time)
                Spectral_Data_Structure.attrs['Start Time']=str(self.start_time)
                Eddy_Data_Structure.attrs['Start Time']=str(self.start_time)

                Wind_Data_Structure.attrs['End Time']=str(self.end_time)
                Spectral_Data_Structure.attrs['End Time']=str(self.end_time)
                Eddy_Data_Structure.attrs['End Time']=str(self.end_time)

                if self.state==1:
                    Wind_Data_Structure.attrs['State']='Inhomogeneous'
                    Spectral_Data_Structure.attrs['State']='Inhomogeneous'
                    Eddy_Data_Structure.attrs['State']='Inhomogeneous'
                elif self.state==0:
                    Wind_Data_Structure.attrs['State']='Homogeneous'
                    Spectral_Data_Structure.attrs['State']='Homogeneous'
                    Eddy_Data_Structure.attrs['State']='Homogeneous'

            return Wind_Data_Structure,Spectral_Data_Structure,Eddy_Data_Structure