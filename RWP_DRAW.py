import matplotlib.pyplot as plt
import cmaps
import CINWPR
import datetime
import numpy as np
'''
基础函数
'''
def get_standard_timelist(ori_timelist,standard_format='%Y-%m-%d %H:%M:%S'):
    standard_timelist=[]
    for ori_time in ori_timelist:
        standard_timelist.append(datetime.datetime.strptime(ori_time,standard_format))

    return standard_timelist

def is_unique(arr_1d):
    if np.unique(arr_1d).shape[0]==1:
        return True
    else:
        return False

def time_state_check(time_list):
    hour_arr=np.zeros([0])
    day_arr=np.zeros([0])
    month_arr=np.zeros([0])
    year_arr=np.zeros([0])

    for time in time_list:
        hour_arr=np.append(hour_arr,time.hour)
        day_arr=np.append(day_arr,time.day)
        month_arr=np.append(month_arr,time.month)
        year_arr=np.append(year_arr,time.year)

    if is_unique(day_arr):
        state=1
    elif is_unique(month_arr):
        state=2
    elif is_unique(year_arr):
        state=3
    else:
        state=4

    return state,hour_arr,day_arr,month_arr,year_arr

def auto_calc_draw_range(draw_data):



    return

def min_max_scaler(arr_1d):
    min_max_arr=(arr_1d-arr_1d.min())/(arr_1d.max()-arr_1d.min())

    return min_max_arr

def get_format_timelabel(hour,minute=0):
    format_time=time_format_num(hour)+time_format_num(minute)

    return format_time

def seconds_transform(input_num,input_state):
    if input_state=='hour':
        seconds=input_num*3600
    elif input_state=='minute':
        seconds=input_num*60
    
    return seconds

def second_inverse_transform(input_second):
    seconds=input_second%60
    minutes=input_second//60
    hours=minutes//60
    minutes=minutes%60

    return int(hours),int(minutes),int(seconds)

def get_time_secntion_xaxis_info(timelist,time_state,interval,hour_arr,day_arr,month_arr,year_arr):
    interval_hour,interval_minute,interval_second=second_inverse_transform(interval)

    draw_x=get_time_arr(timelist)
    draw_x_ticklabels=[]
    if time_state==1:
        start_hour=hour_arr[0]
        day=day_arr[0]
        month=month_arr[0]
        year=year_arr[0]

        start_time=datetime.datetime(int(year),int(month),int(day),int(start_hour+interval_hour),int(interval_minute),int(interval_second))
        start_time_ticks=(start_time-timelist[0]).seconds
        draw_x_ticks=np.arange(start_time_ticks,draw_x.max()+1,interval)

        for tick_seconds in draw_x_ticks:
            tick_time=timelist[0]+datetime.timedelta(seconds=tick_seconds)
            draw_x_ticklabels.append(get_format_timelabel(tick_time.hour,tick_time.minute))
        
        draw_x_ticks/=draw_x.max()
        draw_x=min_max_scaler(draw_x)
        return draw_x,draw_x_ticks,draw_x_ticklabels

def get_height_section_yaxis_info(height_arr,height_interval):
    draw_y=height_arr
    draw_y_ticks=np.arange(height_interval,height_arr.max()+1,height_interval)
    draw_y_ticklabels=[]

    for height_tick in draw_y_ticks:
        draw_y_ticklabels.append(str(height_tick))
       
    return draw_y,draw_y_ticks,draw_y_ticklabels

'''
时间-高度剖面图
'''
def draw_time_height_section(ax,time_list,height_arr,draw_data,draw_levels,colorbar_ticks,data_declaration,time_interval=seconds_transform(3,'hour'),height_interval=600,color=cmaps.BlAqGrYeOrRe,x_fontsize=14,x_weight='bold',y_fontsize=14,y_weight='bold',
                             x_labelsize=18,x_labelweight='bold',y_labelsize=18,y_labelweight='bold',timelabel='LST',beam_direction=None,beam_size=18,beam_loc='right',beam_wight='bold',beam_pad=0.4,title=None,title_size=16,title_loc='left',title_weight='bold',title_pad=0.4,
                             ticks_direction='out',draw_extend='both',colorbar_ticklabelsize=14,colorbar_ticklen=0,extendrect=True,colorbar_orientation='vertical',colorbar_labelsize=18):
    '''
    ax为画图画布;
    time_list为时间轴，应为datatime形式的列表数据;
    height_arr为高度轴：应为numpy.array形式数据;
    draw_data为所需绘制的数据，应为numpy.array形式;
    time_interval为时间ticks间隔，以小时计算
    height_interval为高度间隔，以米(m)为单位，根据国内常见风廓线雷达的垂直高度分辨率推荐尝试600及其倍数
    data_declaration为填色数据说明，用于colorbar说明，格式为：'数据名称(数据单位)';
    color为填图颜色默认为：cmaps.BlAqGrYeOrReVi200;
    timelabel为时间轴说明标签，默认为当地标准时间(LST);
    beam_direction为波束方向说明;
    '''
    time_state,hour_arr,day_arr,month_arr,year_arr=time_state_check(time_list)
    
    draw_x,draw_x_ticks,draw_x_ticklabels=get_time_secntion_xaxis_info(time_list,time_state,time_interval,hour_arr,day_arr,month_arr,year_arr)
    draw_y,draw_y_ticks,draw_y_ticklabels=get_height_section_yaxis_info(height_arr,height_interval)
    
    ax.set_xlim(0,1)
    ax.set_xticks(draw_x_ticks)
    ax.set_ylim(height_arr.min(),int(height_arr.max()))
    ax.set_yticks(draw_y_ticks)
    ax.tick_params(axis='both',which='major',direction=ticks_direction,length=7,width=3,pad=5,top=True,right=True)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='minor',length=4,width=3,top=True,right=True)

    ax.set_xticklabels(draw_x_ticklabels,fontsize=x_fontsize,weight=x_weight)
    ax.set_yticklabels(draw_y_ticklabels,fontsize=y_fontsize,weight=y_weight)

    ax.set_xlabel('Time('+timelabel+')',fontsize=x_labelsize,weight=x_labelweight)
    ax.set_ylabel('Altitude(m)',fontsize=y_labelsize,weight=y_labelweight)
    
    ax_y=ax.twinx()
    ax_y.set_yticklabels([])
    ax_x=ax.twiny()
    ax_x.set_xticklabels([])
    ax_x.set_xlim(0,1)
    ax_x.tick_params(axis='both',which='major',direction=ticks_direction,length=7,width=3,pad=5,top=True,right=True)
    ax_x.minorticks_on()
    ax_x.tick_params(axis='both',which='minor',length=4,width=3,top=True,right=True)
    ax_y.tick_params(axis='both',which='major',direction=ticks_direction,length=7,width=3,pad=5,top=True,right=True)
    ax_y.minorticks_on()
    ax_y.tick_params(axis='both',which='minor',length=4,width=3,top=True,right=True)
    ax_y.set_ylim(height_arr.min(),int(height_arr.max()))
    cf=ax.contourf(draw_x,draw_y,draw_data,levels=draw_levels,extend=draw_extend,cmaps=color)
    cb=plt.colorbar(cf,orientation=colorbar_orientation,ticks=colorbar_ticks,extendrect=extendrect)
    cb.set_label(data_declaration,fontsize=colorbar_labelsize)
    cb.ax.tick_params(direction='in',labelsize=colorbar_ticklabelsize,length=colorbar_ticklen)
    
    if title is not None:
        ax.set_title(title,fontsize=title_size,loc=title_loc,weight=title_weight,pad=title_pad)
    if beam_direction is not None:
        ax.set_title(beam_direction,fontsize=beam_size,loc=beam_loc,weight=beam_wight,pad=beam_pad)
        
    return
   

'''
时间-高度剖面图
'''
def draw_time_height_wind_section(ax,time_list,height_arr,u_data,v_data,draw_levels,colorbar_ticks,data_declaration,time_interval=seconds_transform(3,'hour'),height_interval=600,color=cmaps.BlAqGrYeOrRe,x_fontsize=14,x_weight='bold',y_fontsize=14,y_weight='bold',
                             x_labelsize=18,x_labelweight='bold',y_labelsize=18,y_labelweight='bold',timelabel='LST',beam_direction=None,beam_size=18,beam_loc='right',beam_wight='bold',beam_pad=0.4,title=None,title_size=16,title_loc='left',title_weight='bold',title_pad=0.4,
                             ticks_direction='out',draw_extend='both',colorbar_ticklabelsize=14,colorbar_ticklen=0,extendrect=True,colorbar_orientation='vertical',colorbar_labelsize=18):
    '''
    ax为画图画布;
    time_list为时间轴，应为datatime形式的列表数据;
    height_arr为高度轴：应为numpy.array形式数据;
    draw_data为所需绘制的数据，应为numpy.array形式;
    time_interval为时间ticks间隔，以小时计算
    height_interval为高度间隔，以米(m)为单位，根据国内常见风廓线雷达的垂直高度分辨率推荐尝试600及其倍数
    data_declaration为填色数据说明，用于colorbar说明，格式为：'数据名称(数据单位)';
    color为填图颜色默认为：cmaps.BlAqGrYeOrReVi200;
    timelabel为时间轴说明标签，默认为当地标准时间(LST);
    beam_direction为波束方向说明;
    '''
    time_state,hour_arr,day_arr,month_arr,year_arr=time_state_check(time_list)
    
    draw_x,draw_x_ticks,draw_x_ticklabels=get_time_secntion_xaxis_info(time_list,time_state,time_interval,hour_arr,day_arr,month_arr,year_arr)
    draw_y,draw_y_ticks,draw_y_ticklabels=get_height_section_yaxis_info(height_arr,height_interval)
    
    ax.set_xlim(0,1)
    ax.set_xticks(draw_x_ticks)
    ax.set_ylim(height_arr.min(),int(height_arr.max()))
    ax.set_yticks(draw_y_ticks)
    ax.tick_params(axis='both',which='major',direction=ticks_direction,length=7,width=3,pad=5,top=True,right=True)
    ax.minorticks_on()
    ax.tick_params(axis='both',which='minor',length=4,width=3,top=True,right=True)

    ax.set_xticklabels(draw_x_ticklabels,fontsize=x_fontsize,weight=x_weight)
    ax.set_yticklabels(draw_y_ticklabels,fontsize=y_fontsize,weight=y_weight)

    ax.set_xlabel('Time('+timelabel+')',fontsize=x_labelsize,weight=x_labelweight)
    ax.set_ylabel('Altitude(m)',fontsize=y_labelsize,weight=y_labelweight)
    
    ax_y=ax.twinx()
    ax_y.set_yticklabels([])
    ax_x=ax.twiny()
    ax_x.set_xticklabels([])
    ax_x.set_xlim(0,1)
    ax_x.tick_params(axis='both',which='major',direction=ticks_direction,length=7,width=3,pad=5,top=True,right=True)
    ax_x.minorticks_on()
    ax_x.tick_params(axis='both',which='minor',length=4,width=3,top=True,right=True)
    ax_y.tick_params(axis='both',which='major',direction=ticks_direction,length=7,width=3,pad=5,top=True,right=True)
    ax_y.minorticks_on()
    ax_y.tick_params(axis='both',which='minor',length=4,width=3,top=True,right=True)
    ax_y.set_ylim(height_arr.min(),int(height_arr.max()))

    wind_speed=np.sqrt(u_data**2+v_data**2)

    cf=ax.contourf(draw_x,draw_y,wind_speed,levels=draw_levels,extend=draw_extend,cmaps=color)
    cb=plt.colorbar(cf,orientation=colorbar_orientation,ticks=colorbar_ticks,extendrect=extendrect)
    cb.set_label(data_declaration,fontsize=colorbar_labelsize)
    cb.ax.tick_params(direction='in',labelsize=colorbar_ticklabelsize,length=colorbar_ticklen)

    aq=ax.quiver(draw_x,draw_y,u_data,v_data)
    
    if title is not None:
        ax.set_title(title,fontsize=title_size,loc=title_loc,weight=title_weight,pad=title_pad)
    if beam_direction is not None:
        ax.set_title(beam_direction,fontsize=beam_size,loc=beam_loc,weight=beam_wight,pad=beam_pad)
        
    return