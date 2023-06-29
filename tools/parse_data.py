# Date: 06-07-2023
# Author: Mario Freund
# Purpose: Parse data from txt file, wireshark lines need to be removed before

from parse import *
import sys
import csv
from tqdm import tqdm

format_string_timestamp = "{}Timestamp [ticks]: {}\n"
format_string_control = "{}Control: {}\n"
format_string_type = "{}Type: {}\n"
format_string_frame_type = "{}Frame type: {}\n"
format_string_master_payload = "{} Master_request: {}\n"
format_string_slave_response = "{} Slave_response: {}\n"


def parse_mvb_data(record):
    line = []
    index = 0
    
    # timestamp
    timestamp_parsed = parse(format_string_timestamp, record[index])
    timestamp = timestamp_parsed[1]
    line.append(timestamp)
    index += 1

    # control
    control_parsed = parse(format_string_control, record[index])
    control = control_parsed[1]
    line.append(control)
    index += 1

    # type Event or Frame
    type_parsed = parse(format_string_type, record[index])
    type = type_parsed[1]
    line.append(type[:-6]) # slice hex away

    if (type == "Event (0x0)"):
        return line
    
    index += 3

    # Frame type Master or Slave
    try:
        frame_type_parsed = parse(format_string_frame_type, record[index])
        frame_type = frame_type_parsed[1]
    except: # there is an additional line in the status data
        index += 1
        try:
            frame_type_parsed = parse(format_string_frame_type, record[index])
            frame_type = frame_type_parsed[1]
        except: # there is one more additional line in the status data
            index += 1
            try:
                frame_type_parsed = parse(format_string_frame_type, record[index])
                frame_type = frame_type_parsed[1]
            except: # there is one more additional line in the status data
                index += 1
                try:
                    frame_type_parsed = parse(format_string_frame_type, record[index])
                    frame_type = frame_type_parsed[1]
                except: # there is one more additional line in the status data
                    index += 1
                    try:
                        frame_type_parsed = parse(format_string_frame_type, record[index])
                        frame_type = frame_type_parsed[1]
                    except:
                        index += 1 # there is one more additional line in the status data
                        frame_type_parsed = parse(format_string_frame_type, record[index])
                        frame_type = frame_type_parsed[1]
    line.append(frame_type[:-6]) # slice hex away
    index += 3

    if (frame_type == "Master frame (0x1)"): # continue with master frame
        master_payload_parsed = parse(format_string_master_payload, record[index])
        master_payload = master_payload_parsed[1]
        line.append(master_payload)
        return line
    
    slave_payload_parsed = parse(format_string_slave_response, record[index])
    slave_payload = slave_payload_parsed[1]
    line.append(slave_payload)
    return line


def get_record_indices(lines, record_format_string):
    indices = []
    i = 0
    #for line in lines:
        # remove empty lines
    #    if line == '\n':
    #        lines.pop(line)
    print("Getting record borders.....")
    for line in tqdm(lines):
        parsed = parse(record_format_string, line)
        if (parsed):
            indices.append(i)
        i += 1
    return indices

inputfile = sys.argv[1]
outputfile = sys.argv[2]
lines = []
output_buffer = []

record_format_string = "{}Record{}"

with open(inputfile, 'r') as f:
    lines = f.readlines()

indices = get_record_indices(lines, record_format_string)
print("Parsing data.....")
for i in tqdm(range(len(indices)-1)):
    line = parse_mvb_data(lines[indices[i]+1:indices[i+1]])
    output_buffer.append(line)

with open(outputfile, 'w') as csvfile:
    linewriter = csv.writer(csvfile, lineterminator='\n')
    linewriter.writerow(['Time(ticks)', 'Control', 'Type', 'Frame Type', 'Payload'])
    linewriter.writerows(output_buffer)







