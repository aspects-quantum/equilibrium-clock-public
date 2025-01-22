import numpy as np
import matplotlib.pyplot as plt

import json
import sys
import os

directory = os.getcwd()
os.chdir(os.getcwd())

from pathlib import Path

def open_memmap(file_path: Path, mode="r"):
    """
    A function to open a memory map, while also parsing its name to retrieve the name, units, data_type and array shape.
    @param file_path: pathlib path to the folder.
    @param mode: the mode with which to open the memmap.
    @return: name, unit, memmap.
    """
    
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    name, unit, data_type, shape_string = file_path.stem.split("-", 3)
    # clipping the brackets off
    shape_string = shape_string[1:-1]

    # parsing the shape string
    shape = tuple(shape_string.split(",", shape_string.count(",")))
    shape = tuple(filter(lambda x: x != "", shape))
    shape = tuple(int(i) for i in shape)

    # loading the memmaps
    my_memmap = np.memmap(filename=file_path, dtype=data_type, mode=mode, shape=shape)

    return my_memmap

def load_qgor_trace_id(data_folder, data_id, data_format, file_names=[]):
    if len(file_names)==0:
        file_names[0] = 'arbitrary_lab_time-samples-'
        file_names[1] = 'B-V-'
        file_names[2] = 'time-s-'

    X = open_memmap('./'+ data_folder + '/' + data_id + 
                    '/'+ file_names[0]+ data_format +'.dat')
    A = open_memmap('./'+ data_folder + '/' + data_id + 
                    '/' + file_names[1]+ data_format +'.dat')
    B = open_memmap('./'+ data_folder + '/' + data_id + 
                    '/' + file_names[2]+ data_format +'.dat')
    data = {"samples": X,
            "I_sensor": A,
            "rel_time": B}
    
    return data

def load_id(data_folder, data_id, data_format=None,file_names=[]):
    try:
        meta_data = json.load(open(data_folder+'/'+data_id+'/meta_data.json'))
        if data_format == None:
            data_format = "float64-("+str(meta_data['data_shape'][0])+",)"
    except:
        raise Exception("meta_data.json is missing for data_id = "+data_id)

    data = dict(load_qgor_trace_id(data_folder, data_id, data_format,file_names).items())
    
    sensor_current = data['I_sensor']
    time = data['samples']*np.round(data['rel_time'].max() - data['rel_time'].min(), 1)
    
    return sensor_current, time, meta_data

# def load_many_q_datasets(data_ids, data_format):
#     data = [dict(load_qgor_trace_id(data_folder, str(i), data_format).items()) for i in data_ids]
#     return data