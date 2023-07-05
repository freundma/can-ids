# Data: 06-09-2023
# Author: Mario Freund
# Purpose: Obtain certain statistics from MVB data master frames
#   - process data addresses and how often they appear
#   - message data addresses and how often they appear

import sys
import csv

def parse_payload(hex_string): # e.g. 0x3082
    fcode = hex_string[2]
    address_type = "process data"
    if (fcode == '0' or fcode == '1' or fcode == '2' or fcode == '3' or fcode == '4'):
        return address_type, hex_string[3:] # address
    if (fcode == 'c'): # message data
        address_type = "message data"
        return address_type, hex_string[3:]
    return None, None

infile = sys.argv[1]

process_data_map = {}
message_data_map = {}

with open(infile) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if (row['Frame Type'] != 'Master frame'):
            continue
        payload = row['Payload']
        address_type, address = parse_payload(payload)
        if (address_type == "process data"):
            if (address in process_data_map):
                process_data_map[address] += 1
            else:
                process_data_map[address] = 1
        if (address_type == "message data"):
            if (address in message_data_map):
                message_data_map[address] += 1
            else:
                message_data_map[address] = 1
    
    print("Found the following process data addresses.....")
    for address in process_data_map:
        print("address: {}, occurrences: {}".format(address, process_data_map[address]))

    print("Found the following message data addresses.....")
    for address in message_data_map:
        print("address: {}, occurrences: {}".format(address, message_data_map[address]))
    





