import os
import numpy as np
import netCDF4 as nc
import cmaps
import pandas as pd
import datetime
import xarray as xr
import warnings
import itertools
import struct
import codecs
import CINWPR

warnings.filterwarnings('ignore')

'''
基础函数
'''
def str_in(key_words,target_arr):
    lists=list(filter(lambda x: key_words in str(x),target_arr))
    return lists

def find_str_loc(key_words,target_arr,strict=True):
    loc_arr=np.zeros([0])
    start_loc=0
    
    if strict:
        while True:
            try:
                words_loc=target_arr[start_loc:].index(key_words)
                loc_arr=np.append(loc_arr,words_loc+start_loc)
                start_loc=words_loc+1
                
                if start_loc<=loc_arr[-1]:
                    break
            except:
                break
    else:
        for ind in range(len(target_arr)):
            if key_words in target_arr[ind]:
                loc_arr=np.append(loc_arr,ind)
            
    return loc_arr.astype(np.int64)

def remove_str(key_words,target_arr,strict=True):
    ori_ind=np.arange(len(target_arr))
    key_ind=find_str_loc(key_words,target_arr,strict=False)
    removed_arr=list(np.array(target_arr)[np.setdiff1d(ori_ind,key_ind)])

    return removed_arr

def time_format_num(num):
    if num<10:
        return '0'+str(num)
    else:
        return str(num)

def to_form_Datatime(year,month=1,day=1,hour=0,minute=0,second=0,GMT=False):
    time_str=time_format_num(year)+time_format_num(month)+time_format_num(day)+time_format_num(hour)+time_format_num(minute)+time_format_num(second)
    if GMT:
        return str(datetime.datetime.strptime(time_str,'%Y%m%d%H%M%S')-datetime.timedelta(hours=8))
    else:
        return str(datetime.datetime.strptime(time_str,'%Y%m%d%H%M%S'))

def loc_transfer(deg,mins,sec):
    mins+=sec/60
    deg+=mins/60
    
    return deg

def left_append(element,list_file):
    new_list=[]
    new_list.append(element)
    
    for strs in list_file:
        new_list.append(strs)

    return new_list

def remove_blank(strings_arr):
    useful_strings=[]
    for string in strings_arr:
        if string!='':
            useful_strings.append(string)

    return useful_strings

def get_time_arr(time_list):
    time_arr=np.zeros([1])
    for t_ind in range(len(time_list)-1):
        delta_second=(time_list[t_ind+1]-time_list[0]).seconds
        time_arr=np.append(time_arr,delta_second)
    
    return time_arr
'''
文件信息获取
'''
def get_fft_station_info(filename,file_inds=16,site_inds=168):
    file=open(filename,'rb')
    lines=file.readlines()
    site_infos=lines[0][file_inds:file_inds+site_inds]
    
    height=np.float64(str(codecs.decode(site_infos[-56:-40],'UTF-8')).split('\x00')[0])
    
    loc_infos=str(codecs.decode(site_infos[-72:-56],'UTF-8')).split('\x00')[0]
    deg,mins,sec=np.float(loc_infos.split('/')[0][1:]),np.float(loc_infos.split('/')[1]),np.float(loc_infos.split('/')[2])
    lats=loc_transfer(deg,mins,sec)

    loc_infos=str(codecs.decode(site_infos[-88:-72],'UTF-8')).split('\x00')[0]
    deg,mins,sec=np.float(loc_infos.split('/')[0][1:]),np.float(loc_infos.split('/')[1]),np.float(loc_infos.split('/')[2])
    lons=loc_transfer(deg,mins,sec)
    
    return lats,lons,height

def find_station_location(filename,data_type):
    if data_type in ['ROBS','OOBS','HOBS','RAD']:
        file=open(filename)
        file.readline()
        station_line=file.readline()
        station_infos=station_line.split(' ')
        
        station_lons=station_infos[1]
        station_lats=station_infos[2]
        station_height=station_infos[3]
    elif data_type == 'FFT':
        station_lats,station_lons,station_height=get_fft_station_info(filename)
    else:
        raise Exception('Invalid Data Type Or Unrecognized Filename')

    return station_lats,station_lons,station_height

def Datafile_info(file_name,split_signal='_'):
    file_info=file_name.split(split_signal)
    
    station_num=file_info[3]
    data_type=file_info[-1][:-4]
    
    station_lat,station_lon,station_height=find_station_location(file_name,data_type)
    data_time=datetime.datetime.strptime(file_info[4],'%Y%m%d%H%M%S')
    
    if file_info[5]=='O':
        data_property='观测数据'
    elif file_info[5]=='P':
        data_property='产品数据'
    elif file_info[5]=='R':
        data_property='状态数据'
    else:
        data_property='未知数据'
    
    if file_info[7]=='LC':
        radar_type='L波段 边界层风廓线雷达'
    elif file_info[7]=='PA':
        radar_type='P波段 对流层I型风廓线雷达'
    elif file_info[7]=='PB':
        radar_type='P波段 对流层II型风廓线雷达'
    else:
        radar_type='未知雷达'
        print('未知雷达类型')
     
    if data_type=='FFT':
        data_type_description='功率谱数据'
    elif data_type=='RAD':
        data_type_description='径向数据'
    elif data_type=='ROBS':
        data_type_description='实时采样高度产品数据'
    elif data_type=='HOBS':
        data_type_description='半小时平均采样高度产品数据'
    elif data_type=='OOBS':
        data_type_description='一小时平均采样高度产品数据'
    else:
        data_type_description='未知数据'
        
    file_type=file_info[-1][-3:]
    '''
    station_num:站号(字符串类型)
    station_lat,station_lon:站点经纬(浮点数类型,精度为小数点后3位)
    station_height:站点高度
    station_time:数据时间(datetime类型)，默认为北京时间，当GMT=True时为时间时
    data_property:数据性质(字符串类型)
    radar_type:雷达类型(字符串类型)
    data_type_description:数据类型说明(字符串类型)
    data_type:数据类型(字符串类型)
    file_type:文件类型(字符串类型)
    '''
    return station_num,station_lat,station_lon,station_height,data_time,data_property,radar_type,data_type_description,data_type,file_type

def lack_num_progress(lack_singal,arr_lines,mask_num=-9999.0):
    lack_loc=find_str_loc(lack_singal,arr_lines,strict=False)

    for loc in lack_loc:
        arr_lines[loc]=str(mask_num)

    return

def get_Time_P_mean(Time_P_num):
    if Time_P_num==0:
        Time_P='计算机时钟'
    elif Time_P_num==1:
        Time_P='GPS'
    elif Time_P_num==2:
        Time_P='其他'
    else:
        Time_P='未知时间来源'
    
    return Time_P

def get_correction_num_mean(correction_num):
    if correction_num==0:
        correction='无标校'
    elif correction_num==1:
        correction='自动标校'
    elif correction_num==2:
        correction='一周内人工标校'
    elif correction_num==3:
        correction='一月内人工标校'
    else:
        correction='未知标校'

    return correction

def Read_Product_Data(file_name,data_description,stationinfo_lines=3,final_signal='NNNN',split_signal=' ',lack_singal='//'):
    file=open(file_name)
    file_lines=file.readlines()
    file_lines=[line.strip('\n') for line in file_lines]
    finish_loc=find_str_loc(final_signal,file_lines)
    data_lines=file_lines[stationinfo_lines:finish_loc[0]]

    sample_height=np.zeros([0])
    wind_direction=np.zeros([0])
    horizontal_speed=np.zeros([0])
    vertical_speed=np.zeros([0])
    horizontal_confidence_level=np.zeros([0])
    vertical_confidence_level=np.zeros([0])
    vertical_CN2=np.zeros([0])

    for data_line in data_lines:
        data_info=data_line.split(split_signal)

        lack_num_progress(lack_singal,data_info)

        sample_height=np.append(sample_height,np.int(data_info[0]))
        wind_direction=np.append(wind_direction,np.float(data_info[1]))
        horizontal_speed=np.append(horizontal_speed,np.float(data_info[2]))
        vertical_speed=np.append(vertical_speed,np.float(data_info[3]))
        horizontal_confidence_level=np.append(horizontal_confidence_level,np.float(data_info[4]))
        vertical_confidence_level=np.append(vertical_confidence_level,np.float(data_info[5]))
        vertical_CN2=np.append(vertical_CN2,np.float(data_info[6]))

    sample_height.astype(np.int)

    DATA_Structure=xr.Dataset(data_vars=dict(Wind_direction=(["Level"],wind_direction),
                                       Horizontal_speed=(["Level"],horizontal_speed),
                                       Vertical_speed=(["Level"],vertical_speed),
                                       Horizontal_confidence_level=(['Level'],horizontal_confidence_level),
                                       Vertical_confidence_level=(['Level'],vertical_confidence_level),
                                       Vertical_CN2=(['Level'],vertical_CN2)),
                        coords=dict(Level=sample_height),
                        attrs=dict(Description=data_description))

    return DATA_Structure

def Get_RADfile_info(model_infos,GMT=False,split_signal=' '):
    radar_specifications_lines=model_infos[::2]
    observation_parameters_lines=model_infos[1::2]

    if len(model_infos)==2:
        model_name=['Low']
    elif len(model_infos)==4:
        model_name=['Low','Medium']
    elif len(model_infos)==6:
        model_name=['Low','Medium','High']
    else:
       raise Exception('Format Error')

    '''
    雷达性能参数
    '''
    antenna_gain=np.zeros([0])
    feeder_loss=np.zeros([0])
    east_angle=np.zeros([0])
    west_angle=np.zeros([0])
    north_angle=np.zeros([0])
    south_angle=np.zeros([0])
    mid_row_angle=np.zeros([0])
    mid_col_angle=np.zeros([0])
    beam_num=np.zeros([0])
    sample_freq=np.zeros([0])
    emi_wavelen=np.zeros([0])
    pulse_repet_freq=np.zeros([0])
    pulse_width=np.zeros([0])
    horizontal_beam_width=np.zeros([0])
    vertical_beam_width=np.zeros([0])
    emi_freq_peak=np.zeros([0])
    emi_freq_mean=np.zeros([0])
    start_height=np.zeros([0])
    end_height=np.zeros([0])

    for radar_specifications_line in radar_specifications_lines:
        radar_info=radar_specifications_line.split(split_signal)

        antenna_gain=np.append(antenna_gain,np.int(radar_info[0]))
        feeder_loss=np.append(feeder_loss,np.float(radar_info[1]))
        east_angle=np.append(east_angle,np.float(radar_info[2]))
        west_angle=np.append(west_angle,np.float(radar_info[3]))
        south_angle=np.append(south_angle,np.float(radar_info[4]))
        north_angle=np.append(north_angle,np.float(radar_info[5]))
        mid_row_angle=np.append(mid_row_angle,np.float(radar_info[6]))
        mid_col_angle=np.append(mid_col_angle,np.float(radar_info[7]))
        beam_num=np.append(beam_num,np.int(radar_info[8]))
        sample_freq=np.append(sample_freq,np.int(radar_info[9]))
        emi_wavelen=np.append(emi_wavelen,np.int(radar_info[10]))
        pulse_repet_freq=np.append(pulse_repet_freq,np.int(radar_info[11]))
        pulse_width=np.append(pulse_width,np.float(radar_info[12]))
        horizontal_beam_width=np.append(horizontal_beam_width,np.int(radar_info[13]))
        vertical_beam_width=np.append(vertical_beam_width,np.int(radar_info[14]))
        emi_freq_peak=np.append(emi_freq_peak,np.float(radar_info[15]))
        emi_freq_mean=np.append(emi_freq_mean,np.float(radar_info[16]))
        start_height=np.append(start_height,np.int(radar_info[17]))
        end_height=np.append(end_height,np.int(radar_info[18]))


    radar_spec_data={'天线增益':antenna_gain.astype(np.int),
                     '馈线损耗':feeder_loss,
                     '东波束-铅垂线夹角':east_angle,
                     '西波束-铅垂线夹角':west_angle,
                     '南波束-铅垂线夹角':south_angle,
                     '北波束-铅垂线夹角':north_angle,
                     '中波束(行)-铅垂线夹角':mid_row_angle,
                     '中波束(列)-铅垂线夹角':mid_col_angle,
                     '波束数':beam_num.astype(np.int),
                     '采样频率':sample_freq.astype(np.int),
                     '发射波长':emi_wavelen.astype(np.int),
                     '脉冲重复频率':pulse_repet_freq.astype(np.int),
                     '脉冲宽度':pulse_width,
                     '水平波束宽度':horizontal_beam_width.astype(np.int),
                     '垂直波束宽度':vertical_beam_width.astype(np.int),
                     '发射峰值功率':emi_freq_peak,
                     '发射平均功率':emi_freq_mean,
                     '起始采样高度':start_height.astype(np.int),
                     '终止采样高度':end_height.astype(np.int)}
    radar_spec_frame=pd.DataFrame(radar_spec_data,index=model_name,columns=list(radar_spec_data.keys()))

    """
    观测参数
    """
    time_source=[]
    start_time=[]
    end_time=[]
    correction=[]
    incoherent_acc=np.zeros([0])
    coherent_acc=np.zeros([0])
    fft=np.zeros([0])
    spectrum_mean=np.zeros([0])
    direct_order=[]
    rev_east_angle=np.zeros([0])
    rev_west_angle=np.zeros([0])
    rev_south_angle=np.zeros([0])
    rev_north_angle=np.zeros([0])

    for observation_parameters_line in observation_parameters_lines:
        obser_info=observation_parameters_line.split(split_signal)
        if '' in obser_info:
            obser_info=remove_blank(obser_info)

        time_source_num=np.int(obser_info[0])
        time_source.append(get_Time_P_mean(time_source_num))

        if GMT:
            start_obs_time=datetime.datetime.strptime(obser_info[1],'%Y%m%d%H%M%S')-datetime.timedelta(hours=8)
            end_obs_time=datetime.datetime.strptime(obser_info[2],'%Y%m%d%H%M%S')-datetime.timedelta(hours=8)
        else:
            start_obs_time=datetime.datetime.strptime(obser_info[1],'%Y%m%d%H%M%S')
            end_obs_time=datetime.datetime.strptime(obser_info[2],'%Y%m%d%H%M%S')    
        start_time.append(str(start_obs_time))
        end_time.append(str(end_obs_time))

        correction_num=np.int(obser_info[3])
        correction.append(get_correction_num_mean(correction_num))

        incoherent_acc=np.append(incoherent_acc,np.int(obser_info[4]))
        coherent_acc=np.append(coherent_acc,np.int(obser_info[5]))
        fft=np.append(fft,np.int(obser_info[6]))
        spectrum_mean=np.append(spectrum_mean,np.int(obser_info[7]))
        direct_order=obser_info[8]
        rev_east_angle=np.append(rev_east_angle,np.float(obser_info[9]))
        rev_west_angle=np.append(rev_west_angle,np.float(obser_info[10]))
        rev_south_angle=np.append(rev_south_angle,np.float(obser_info[11]))
        rev_north_angle=np.append(rev_north_angle,np.float(obser_info[12]))

    obser_para_data={'时间来源':time_source,
                     '观测开始时间':start_time,
                     '观测结束时间':end_time,
                     '标校状态':correction,
                     '非相干积累':incoherent_acc,
                     '相干积累':coherent_acc,
                     'Fft点数':fft,
                     '谱平均数':spectrum_mean,
                     '波束顺序标志':direct_order,
                     '东波束方位角修正值':rev_east_angle,
                     '西波束方位角修正值':rev_west_angle,
                     '南波束方位角修正值':rev_south_angle,
                     '北波束方位角修正值':rev_north_angle}
    obser_para_frame=pd.DataFrame(obser_para_data,index=model_name,columns=list(obser_para_data.keys()))

    return radar_spec_frame,obser_para_frame

def Get_muti_beam_data(datalines,split_signal=' ',lack_signal='//'):
    sample_height=np.zeros([0])
    speed_spec_width=np.zeros([0])
    SNR=np.zeros([0])
    radial_velocity=np.zeros([0])
    
    for data_line in datalines:
        data_info=str(data_line).split(split_signal)
        
        lack_num_progress(lack_signal,data_info)
        sample_height=np.append(sample_height,np.int(data_info[0]))
        speed_spec_width=np.append(speed_spec_width,np.float(data_info[1]))
        SNR=np.append(SNR,np.float(data_info[2]))
        radial_velocity=np.append(radial_velocity,np.float(data_info[3]))
    
    sample_height=np.unique(sample_height).astype(np.int64)
    beam_num=np.int(speed_spec_width.shape[0]/sample_height.shape[0])

    return sample_height,beam_num,speed_spec_width,SNR,radial_velocity

def Read_RAD_Data(file_name,data_description,info_lines=2,final_signal='NNNN',split_signal=' ',lack_signal='//',beam_signal='RAD',first_signal='FIRST',secend_signal='SECEND'):
    file=open(file_name)
    file_lines=file.readlines()
    file_lines=[line.strip('\n') for line in file_lines]

    model_first_locs=find_str_loc(first_signal,file_lines,strict=False)
    model_secend_locs=find_str_loc(secend_signal,file_lines,strict=False)
    model_first_locs=np.append(model_first_locs,len(file_lines))

    Data_Structure=xr.Dataset()
    model_name=['Level_L','Level_M','Level_H']
    beam_name=['Beam_num_L','Beam_num_M','Beam_num_H']
    SNR_name=['SNR_L','SNR_M','SNR_H']
    VSW_name=['Velocity_spectrum_width_L','Velocity_spectrum_width_M','Velocity_spectrum_width_H']
    radial_velocity_name=['Radial_velocity_L','Radial_velocity_M','Radial_velocity_H']

    for num in range(model_first_locs.shape[0]-1):
        if num==(model_first_locs.shape[0]-2):
            usefule_model_lines=file_lines[model_first_locs[num]:(model_first_locs[num+1])]
        else:
            usefule_model_lines=file_lines[model_first_locs[num]:(model_first_locs[num+1]-info_lines)]
       
        model_data_lines=remove_str(final_signal,usefule_model_lines)
        model_data_lines=remove_str(beam_signal,model_data_lines,strict=False)
        sample_height,beam_num,speed_spec_width,SNR,radial_velocity=Get_muti_beam_data(model_data_lines)

        speed_spec_width=speed_spec_width.reshape([beam_num,sample_height.shape[0]])
        SNR=SNR.reshape([beam_num,sample_height.shape[0]])
        radial_velocity=radial_velocity.reshape([beam_num,sample_height.shape[0]])

        Data_Structure.coords[model_name[num]]=sample_height
        Data_Structure.coords[beam_name[num]]=np.arange(1,(beam_num+1))
        Data_Structure[VSW_name[num]]=((beam_name[num],model_name[num]),speed_spec_width)
        Data_Structure[SNR_name[num]]=((beam_name[num],model_name[num]),SNR)
        Data_Structure[radial_velocity_name[num]]=((beam_name[num],model_name[num]),radial_velocity)

        Data_Structure.attrs['Description']=data_description

    return Data_Structure

def get_specific_fft_data(file_lines,start_ind,beam_num,gate_num,fft_num,byte_size=4,file_lines_num=0,):
    Fft_data=np.zeros([0])
    data_nums=0

    while True:
        if data_nums==(beam_num*gate_num*fft_num):
            break
            
        rest_len=len(file_lines[file_lines_num][start_ind:])
        if rest_len>=4:
            decode_data=file_lines[file_lines_num][(start_ind):(start_ind+4)]
            if rest_len==4:
                start_ind=0
                file_lines_num+=1
            else:
                start_ind+=4
        else:
            start_ind=byte_size-rest_len
            decode_data=file_lines[file_lines_num][(-rest_len):]+file_lines[(file_lines_num+1)][:start_ind]
            file_lines_num+=1
            
        Fft_data=np.append(Fft_data,struct.unpack('f',decode_data))
        data_nums+=1

    Fft_data=Fft_data.reshape([beam_num,gate_num,fft_num])
    
    return Fft_data,file_lines_num,start_ind

def get_specific_fft_obs_info(obs_model_info,model_name,GMT=False,totle_beam_num=6):
    STime=[]
    TimeP=np.zeros([0])
    SMillisecond=np.zeros([0])
    Calibration=np.zeros([0])
    BeamfxChange=np.zeros([0])
    ETime=[]
    NNtr=np.zeros([0])
    Ntr=np.zeros([0])
    Fft=np.zeros([0])
    SpAver=np.zeros([0])
    BeamDir=[]
    AzimuthE=np.zeros([0])
    AzimuthW=np.zeros([0])
    AzimuthS=np.zeros([0])
    AzimuthN=np.zeros([0])
    
    SYear=struct.unpack('H',obs_model_info[:2])[0]
    SMonth=struct.unpack('B',obs_model_info[2:3])[0]
    SDay=struct.unpack('B',obs_model_info[3:4])[0]
    SHour=struct.unpack('B',obs_model_info[4:5])[0]
    SMinute=struct.unpack('B',obs_model_info[5:6])[0]
    SSecond=struct.unpack('B',obs_model_info[6:7])[0]
    STime.append(to_form_Datatime(SYear,SMonth,SDay,SHour,SMinute,SSecond,GMT=GMT))
    TimeP=np.append(TimeP,struct.unpack('B',obs_model_info[7:8])[0])
    TimeP_mean=get_Time_P_mean(TimeP[:])
    SMillisecond=np.append(SMillisecond,struct.unpack('L',obs_model_info[8:12])[0])
    Calibration=np.append(Calibration,struct.unpack('h',obs_model_info[12:14])[0])
    Calibration_mean=get_correction_num_mean(Calibration[0])
    BeamfxChange=np.append(BeamfxChange,struct.unpack('h',obs_model_info[14:16])[0])

    EYear=struct.unpack('H',obs_model_info[16:18])[0]
    EMonth=struct.unpack('B',obs_model_info[18:19])[0]
    EDay=struct.unpack('B',obs_model_info[19:20])[0]
    EHour=struct.unpack('B',obs_model_info[20:21])[0]
    EMinute=struct.unpack('B',obs_model_info[21:22])[0]
    ESecond=struct.unpack('B',obs_model_info[22:23])[0]
    ETime.append(to_form_Datatime(EYear,EMonth,EDay,EHour,EMinute,ESecond,GMT=GMT))
    NNtr=np.append(NNtr,struct.unpack('h',obs_model_info[24:26])[0])
    Ntr=np.append(Ntr,struct.unpack('h',obs_model_info[26:28])[0])
    Fft=np.append(Fft,struct.unpack('h',obs_model_info[28:30])[0])
    SpAver=np.append(SpAver,struct.unpack('h',obs_model_info[30:32])[0])

    decode_beamdir=str(codecs.decode(obs_model_info[32:38],'UTF-8'))
    BeamDir=''
    for char in decode_beamdir:
        if ord(char)==0:
            BeamDir+='/'
        else:
            BeamDir+=char
    Beam_num=totle_beam_num-BeamDir.count('/')

    AzimuthE=np.round(struct.unpack('f',obs_model_info[44:48])[0],1)
    AzimuthW=np.round(struct.unpack('f',obs_model_info[48:52])[0],1)
    AzimuthS=np.round(struct.unpack('f',obs_model_info[52:56])[0],1)
    AzimuthN=np.round(struct.unpack('f',obs_model_info[56:60])[0],1)

    obser_data={'观测开始时间':STime[0],
                '时间来源':TimeP_mean,
                '标校状态':Calibration_mean,
                '波束方向改变':BeamfxChange[0],
                '观测结束时间':ETime[0],
                '非相干累计':NNtr[0],
                '相干累计':Ntr[0],
                'Fft点数':Fft[0],
                '谱平均数':SpAver[0],
                '波束顺序标志':BeamDir,
                '有效波束':Beam_num,
                '东波束方位角修正值':AzimuthE,
                '西波束方位角修正值':AzimuthW,
                '南波束方位角修正值':AzimuthS,
                '北波束方位角修正值':AzimuthN}

    obser_frame=pd.DataFrame(obser_data,index=model_name,columns=list(obser_data.keys()))

    return obser_frame

def get_specific_fft_per_info(per_model_info,model_name):
    Ae=struct.unpack('I',per_model_info[:4])
    AgcWast=struct.unpack('f',per_model_info[4:8])
    AngleE=struct.unpack('f',per_model_info[8:12])
    AngleW=struct.unpack('f',per_model_info[12:16])
    AngleS=struct.unpack('f',per_model_info[16:20])
    AngleN=struct.unpack('f',per_model_info[20:24])
    AngleR=struct.unpack('f',per_model_info[24:28])
    AngleL=struct.unpack('f',per_model_info[28:32])
    ScanBeamN=struct.unpack('I',per_model_info[32:36])
    SampleP=struct.unpack('I',per_model_info[36:40])
    WaveLength=struct.unpack('I',per_model_info[40:44])
    Prp=struct.unpack('f',per_model_info[44:48])
    PusleW=struct.unpack('f',per_model_info[48:52])
    HBeamW=struct.unpack('H',per_model_info[52:54])
    VBeamW=struct.unpack('H',per_model_info[54:56])
    TranPp=struct.unpack('f',per_model_info[56:60])
    TranAP=struct.unpack('f',per_model_info[60:64])
    StartSampleBin=struct.unpack('I',per_model_info[64:68])
    EndSampleBin=struct.unpack('I',per_model_info[68:72])
    BinLength=struct.unpack('h',per_model_info[72:74])
    BinNum=struct.unpack('h',per_model_info[74:76])

    per_data={'天线增益':Ae,
              '馈线损耗':AgcWast,
              '东波束-铅垂线夹角':AngleE,
              '西波束-铅垂线夹角':AngleW,
              '南波束-铅垂线夹角':AngleS,
              '北波束-铅垂线夹角':AngleN,
              '中波束(行)-铅垂线夹角':AngleR,
              '中波束(列)-铅垂线夹角':AngleL,
              '扫描波束数':ScanBeamN,
              '采样频率':SampleP,
              '发射波长':WaveLength,
              '脉冲重复频率':Prp,
              '脉冲宽度':PusleW,
              '水平波束宽度':HBeamW,
              '垂直波束宽度':VBeamW,
              '发射峰值功率':TranPp,
              '发射平均功率':TranAP,
              '起始采样高度':StartSampleBin,
              '终止采样高度':EndSampleBin,
              '距离库长':BinLength,
              '距离库数':BinNum}

    per_frame=pd.DataFrame(per_data,index=model_name,columns=list(per_data.keys()))
    
    return per_frame

def get_FFT_file_needed_info(file_name,return_choice,data_description=[''],file_header_len=16,site_info_len=168,performance_info_len=116,observation_info_line=100,GMT=False):
    file=open(file_name,'rb')
    file_lines=file.readlines()
    
    model_name=[['Low'],['Medium'],['High']]
    fft_data_name=['FFT_L','FFT_M','FFT_H']
    fft_point_name=['FFT_Point_L','FFT_Point_M','FFT_Point_H']
    beam_name=['Beam_Num_L','Beam_Num_M','Beam_Num_H']
    gate_name=['Level_L','Level_M','Level_H']

    obs_frame=pd.DataFrame()
    per_frame=pd.DataFrame()
    Data_Structure=xr.Dataset()

    file_lines_num=0
    model_start_ind=file_header_len+site_info_len
    model_num=0
    while True:
        if file_lines_num==len(file_lines):
            break

        model_per_frame=get_specific_fft_per_info(file_lines[file_lines_num][model_start_ind:],list(model_name[model_num]))
        model_obs_frame=get_specific_fft_obs_info(file_lines[file_lines_num][(model_start_ind+performance_info_len):],list(model_name[model_num]),GMT)
        per_frame=pd.concat([per_frame,model_per_frame],axis=0)
        obs_frame=pd.concat([obs_frame,model_obs_frame],axis=0)

        model_beam_num=model_per_frame['扫描波束数'][0]
        model_fft_num=model_obs_frame['Fft点数'][0]
        model_gate_num=model_per_frame['距离库数'][0]
        model_start_dist=model_per_frame['起始采样高度'][0]
        model_end_dist=model_per_frame['终止采样高度'][0]
        model_step=model_per_frame['距离库长'][0]

        model_fft_data,new_file_lines_num,new_start_ind=get_specific_fft_data(file_lines,start_ind=(model_start_ind+performance_info_len+observation_info_line),beam_num=np.int(model_beam_num),gate_num=np.int(model_gate_num),fft_num=np.int(model_fft_num),file_lines_num=file_lines_num)
        file_lines_num=new_file_lines_num
        model_start_ind=new_start_ind

        Data_Structure.coords[beam_name[model_num]]=np.arange(1,(model_beam_num+1))
        Data_Structure.coords[gate_name[model_num]]=np.arange(model_start_dist,(model_end_dist+1),model_step)
        Data_Structure.coords[fft_point_name[model_num]]=np.arange(1,(model_fft_num+1))
        Data_Structure[fft_data_name[model_num]]=((beam_name[model_num],gate_name[model_num],fft_point_name[model_num]),model_fft_data)
        model_num+=1
    
    Data_Structure.attrs['Description']=data_description
    
    if return_choice=='Model_info':
        return per_frame,obs_frame
    elif return_choice=='Data':
        return Data_Structure

def Read_FFT_Data(file_name,data_description=['']):
    Data_Structure=get_FFT_file_needed_info(file_name,return_choice='Data',data_description=data_description)
    
    return Data_Structure
'''
多文件处理
'''
def Get_filelist_timeinfo(file_list,GMT=False):
    timelist=[]

    for file in file_list:
        if GMT:
            file_time=RWP_READ.Read_RWP_Data(file).time+datetime.timedelta(hours=8)
        else:
            file_time=RWP_READ.Read_RWP_Data(file).time
        timelist.append(str(file_time))

    return timelist

def file_check(file_list):
    lons=np.zeros([0])
    lats=np.zeros([0])
    datatype=[]

    for file in file_list:
        lons=np.append(lons,RWP_READ.Read_RWP_Data(file).lon)
        lats=np.append(lats,RWP_READ.Read_RWP_Data(file).lat)
        datatype.append(RWP_READ.Read_RWP_Data(file).datatype)

    check_num=1
    if np.unique(lons).shape[0]==1:
        if np.unique(lats).shape[0]==1:
            if len(list(set(datatype)))==1:
                check_num=0

    if check_num==1:
        return 'error'
    elif check_num==0:
        return 'correct'
def Timelist_warning():
    warnings.warn('The time series is not uniform',ResourceWarning)

def Get_timelist(self,file_list):
    start_time=RWP_READ.Read_RWP_Data(file_list[0]).time
    end_time=RWP_READ.Read_RWP_Data(file_list[-1]).time

    delta_time=np.zeros([1])
    for num in range(len(file_list)-1):
        delta_time=np.append(delta_time,delta_time[-1]+(RWP_READ.Read_RWP_Data(file_list[num+1]).time-RWP_READ.Read_RWP_Data(file_list[num]).time).seconds)
    if np.unique(delta_time).shape[0]==1:
        timelist_state=0
    else:
        timelist_state=1

    return start_time,end_time,timelist_state,delta_time

def append_arr_shape(new_dims_len,old_shape):
    new_dims_arr=np.zeros([0])
    new_dims_arr=np.append(new_dims_arr,new_dims_len)
    for dim_len in old_shape:
        new_dims_arr=np.append(new_dims_arr,dim_len)

    new_dims_arr=new_dims_arr.astype(np.int64)
    return new_dims_arr


'''
文件保存
'''
def save_data(data_structure,file_name,format='netCDF4'):
    if format=='netCDF4':
        data_structure.to_netdef(file_name)
    return     
'''
风廓线雷达数据
'''

class RWP_READ():
    class Read_RWP_Data():
        def __init__(self,file_name):
            station_num,station_lat,station_lon,station_height,data_time,data_property,radar_type,data_type_description,data_type,file_type=Datafile_info(file_name)
            self.lon=np.float(station_lon)
            self.lat=np.float(station_lat)
            self.height=np.float(station_height)
            self.time_teg='(BJT)'
            self.time=data_time
            self.datatype=data_type
            self.filename=file_name

        def Get_Radar_RAD_Model_info(self,info_lines=2,model_start_signal='RAD FIRST',split_signal=' ',lack_singal='//',GMT=False):
            if self.datatype!='RAD':
                raise Exception('Not A RAD Data Files')

            file=open(self.filename)
            file_lines=file.readlines()
            file_lines=[line.strip('\n') for line in file_lines]

            model_start_loc=find_str_loc(model_start_signal,file_lines,strict=False)
            model_num=model_start_loc.shape[0]

            modelinfo_lines=[]
            for start_ind in model_start_loc:
                modelinfo_lines.append(file_lines[(start_ind-info_lines):start_ind][:])
            modelinfo_lines=list(itertools.chain(*modelinfo_lines))
            radar_spec_frame,obser_para_frame=Get_RADfile_info(modelinfo_lines,GMT=GMT)

            return radar_spec_frame,obser_para_frame

        def Get_Radar_FFT_Model_info(self,GMT=False):
            if self.datatype!='FFT':
                raise Exception('Not A FFT Data Files')

            perform_frame,obser_frame=get_FFT_file_needed_info(self.filename,return_choice='Model_info',GMT=GMT)

            return  perform_frame,obser_frame

        def Get_data(self,GMT=False):
            station_num,station_lat,station_lon,station_height,data_time,data_property,radar_type,data_type_description,data_type,file_type=Datafile_info(self.filename)
            time_tag='(BJT)'
            if GMT:
                data_time=data_time-datetime.timedelta(hours=8)
                time_tag='(GMT)'

            line1='站号:'+station_num+'('+str(station_lat)+','+str(station_lon)+')'+'   站点高度:'+str(station_height)+'m'+'    观测时间:'+str(data_time)+time_tag
            line2='雷达类型:'+radar_type+'  数据类型:'+data_property+'->'+data_type_description
            data_description=line1+'\n'+line2

            Products=['ROBS','HOBS','OOBS']
            if data_type in Products:
                DATA_Structure=Read_Product_Data(self.filename,data_description)
            elif data_type=='RAD':
                DATA_Structure=Read_RAD_Data(self.filename,data_description)
            elif data_type=='FFT':
                DATA_Structure=Read_FFT_Data(self.filename,data_description)
            else:
                raise Exception('Invalid Data Type')

            return DATA_Structure

    class Read_Muti_RWP_Data():
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

        def Get_Muti_Data(self,GMT=False,visual=False):
            if self.check=='error':
                raise Exception('Filelist is Inhomogeneous')

            Data_Structure=xr.Dataset()

            Data_Structure.coords['Time']=self.timelist
            Data_Structure['Ori_Time']=(('Time'),Get_filelist_timeinfo(self.list))

            standard_file_data=RWP_READ.Read_RWP_Data(self.list[0]).Get_data()
            dim_list=list(standard_file_data.dims.keys())
            for dim in dim_list:
                Data_Structure.coords[dim]=np.array(standard_file_data[dim])

            varlist=list(set((list(standard_file_data.variables.keys())))^(set(list(standard_file_data.dims.keys()))))
            for var in varlist:
                var_new_dims_name=tuple(left_append('Time',list(standard_file_data[var].dims)))
                var_new_dims_shape=append_arr_shape((self.timelist).shape[0],list(np.array(standard_file_data[var]).shape))
                var_data=np.zeros(var_new_dims_shape.astype(np.int64))
                for ind in range(len(self.list)):
                    temp_data=RWP_READ.Read_RWP_Data(self.list[ind]).Get_data()
                    try:
                        var_data[ind]=temp_data[var]
                    except:
                        temp_shape=np.array(temp_data[var]).shape
                        print(temp_shape)
                        temp_height=temp_shape.shape[1]
                        temp_beams=temp_shape.shape[0]
                        var_data[ind,:temp_beams,temp_height]=temp_data[var]

                    if visual==True:
                        print((self.list)[ind])
                Data_Structure[var]=(var_new_dims_name,var_data)
            
            Data_Structure.attrs['Description']=standard_file_data.attrs['Description']
            Data_Structure.attrs['Start Time']=str(self.start_time)
            Data_Structure.attrs['End Time']=str(self.end_time)
            
            if self.state==1:
                Data_Structure.attrs['State']='Inhomogeneous'
            elif self.state==0:
                Data_Structure.attrs['State']='Homogeneous'
        
            return Data_Structure