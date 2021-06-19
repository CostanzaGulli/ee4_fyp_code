# File:		csv_to_gui_bin.py
# Author: 	Lei Kuang
# Date:		21st August 2020
# @ Imperial College London

# Decode .csv file into GUI recognized format
# - Required files:
# - [1] info: Absolute Time, Time Elapsed, Temperature, Vref, Average Output
# - [2] vs:   n x 4368 (each row contains a ttn image)

import csv
import struct
import datetime

def get_date_time(fraction = 0):
    return datetime.datetime.now().isoformat().replace('-','').replace(':','')[0:(15 + (0 if fraction==0 else (1 + fraction)))]

def binary_file_append(file, data_list):
    fw = open(file, 'ab')
    for data in data_list:
        fw.write(struct.pack('>H', data)) # 1023 -> b'\x03\xff'
    fw.close()

from pathlib import Path
from tqdm.auto import tqdm

current_path = Path('.')

directories = [x for x in current_path.iterdir() if x.is_dir()]

for directory in tqdm(directories):
    data_export_file = list(directory.glob('*_data_export.csv'))
    data_vschem_file = list(directory.glob('*_vsChem_export.csv'))

    bin_file = list(directory.glob('VF_CSV_*.bin'))

    if len(bin_file) > 0:
        continue

    if len(data_export_file) == len(data_vschem_file) == 1:

        input_path = str(directory) + '/'
        info_path = data_export_file[0]
        chem_path = data_vschem_file[0]

        # Open file
        info_file = open(info_path)
        chem_file = open(chem_path)

        # Read as .csv
        info_csv  = list(csv.reader(info_file))
        chem_csv  = list(csv.reader(chem_file))

        info_list = []
        chem_list = []

        # Check length and remove useless information
        info_len = 0
        for row in info_csv:
            if(row!=[]):
                if(row[1][0].isdigit()):
                    info_len = info_len + 1
                    info_list.append(row)

        chem_len = 0
        for row in chem_csv:
            if(row!=[]):
                chem_len = chem_len + 1
                chem_list.append(row)

        if(info_len!=chem_len):
            print("Invalid File Pairs")
            raise IOError

        # Translation
        file_name = input_path + 'VF_CSV_%s.bin' % get_date_time(0)

        for n in range(0, len(info_list)):
            time_stamp = int(float(info_list[n][1])) * 10
            ttn_cur    = int(float(info_list[n][4]))
            ttn_tem    = int(float(info_list[n][2]))
            ttn_irf    = 0xFFFF

            # Convert string to integer
            data = []
            for element in chem_list[n]:
                try:
                    data.append(int(element))
                except:
                    pass

            if(len(data)!=78*56):
                print("No Enough Elements")
                raise IOError

            data.append(time_stamp)
            data.append(ttn_cur)
            data.append(ttn_tem)
            data.append(ttn_irf)

            binary_file_append(file_name, data)
