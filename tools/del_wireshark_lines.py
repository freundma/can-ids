# Date: 06-07-2023
# Author: Mario Freund
# Purpose: Delete info lines from wireshark trace of MVB data

import sys
from parse import *

input = sys.argv[1]
output = sys.argv[2]
length_of_wireshark_overhead = 69

format_string = "{}Record 1\n"

with open (input,'r') as f:
    lines = f.readlines()
    new_lines = []

    i = 0
    for line in lines:
        new_lines.append(line)
        
        # check if new sequence of data starts here
        parsed = parse(format_string, line)
        if (parsed):
            new_lines[i-length_of_wireshark_overhead:i] = []
            i -= length_of_wireshark_overhead
        i += 1

with open (output, 'w') as f:
    f.writelines(new_lines)

