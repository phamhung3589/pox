import statistics
import flowtable
import numpy as np
import collections
#TODO: ko phải mình table mà là cả flowtable object

def convert_ip_to_int(ip):
    number_list=ip.split('.')
    result=0
    for i,number_text in enumerate(number_list):
        result += int(number_text) * (2 ** ((3-i) * 8))
    return result

def normalize(raw_feature_list):
    """
    normalize a dictionary, each key (feature) contains the list of all values of that feature
    """
    result={}
    for feature in raw_feature_list:
        mean=statistics.mean(raw_feature_list[feature])
        stdev=statistics.pstdev(raw_feature_list[feature])
        print(feature,':','mean:',mean,'stdev:',stdev)
        for i in range(len(raw_feature_list[feature])):
            raw_feature_list[feature][i]-= mean
            raw_feature_list[feature][i]/= stdev

def get_match_hash_key(hashkey, table):
    """
    return None if currently not contain this key or the match key
    """
    if hashkey in table:
        # assume that this has at least one flow entry
        b = convert_ip_to_int(table[hashkey][0].nw_src)
        a = convert_ip_to_int(table[hashkey][0].nw_dst)
        match_hash = (a * a + a + b) if a >= b else (a + b * b)
        if match_hash in table:
            return match_hash
        else:
            return None
    else:
        return None

from collections import Counter
import copy
import classify_ip
def calculate_input_vector(table1, table2):
    keys1  = set(table1.keys())
    keys2 = set(table2.keys())

    new_flows = []
    for key in (keys2 - keys1):
        new_flows.extend(table2[key])

    for key in (keys1 & keys2):
        same_entries1=[]
        same_entries2=[]
        for entry1 in table1[key]:
            for entry2 in table2[key]:
                if (entry2 not in same_entries2) and entry1.is_matched(entry2):
                    same_entries1.append(entry1)
                    same_entries2.append(entry2)
                    break
        for i,entry2 in enumerate(table2[key]):
            if entry2 not in same_entries2:
                # if this is new flow
                new_flows.append(entry2)
            else:
                if entry2.n_packets < same_entries1[i].n_packets:
                    # means this is dead and there is a new flow entry
                    new_flows.append(entry2)
                elif entry2.n_packets > same_entries1[i].n_packets:
                    pseudo_flow = copy.copy(entry2)
                    # -1 to simulate that this is a new flow entry
                    pseudo_flow.n_packets = entry2.n_packets - same_entries1[i].n_packets -1
                    # set unknown fields to None
                    pseudo_flow.start_time = pseudo_flow.duration = pseudo_flow.idle_start_time = pseudo_flow.n_bytes = None
                    new_flows.append(pseudo_flow)

    total_packets = sum((entry.n_packets +1) for entry in new_flows)
    protocols = Counter(entry.proto for entry in new_flows for _ in range(entry.n_packets +1))
    destinations = Counter(entry.nw_dst for entry in new_flows for _ in range(entry.n_packets +1))
    countries = Counter(classify_ip.what_country(entry.nw_src) for entry in new_flows for _ in range(entry.n_packets +1))

    input_vector={}
    input_vector['most_dest_percent'] = destinations.most_common()[0][1] / total_packets
    # TODO: 1 cách j` đấy làm cho protocol và countries nói chung
    input_vector['ICMP_percent'] = protocols['ICMP'] / total_packets
    input_vector['TCP_percent'] = protocols['TCP'] / total_packets
    input_vector['US_percent'] = countries['United States'] / total_packets
    input_vector['Ecuador_percent'] = countries['Ecuador'] / total_packets
    input_vector['Japan_percent'] = countries['Japan'] / total_packets
    input_vector['China_percent'] = countries['China'] / total_packets
    #####khoa
    #####khoa
    return input_vector

import pickle
import os
STATS_INTERVAL = 0.1

import sys
file_type = sys.argv[1] if len(sys.argv) > 1 else 'normal'
with open('output/raw-'+file_type+'-flowtables-500', 'rb') as outfile:
    # this is all the raw tables
    full_list_tables = pickle.load(outfile)

# this dictionary contains all features, each feature is a list
feature_data_list={}
input_list = []
count = 0
for i,flow_table in enumerate(full_list_tables):
    if i==0:
        continue
    # for entry in flow_table:
    #     print(vars(entry))
    # print()
    if flow_table: # ensure that flow table is not empty. TODO: must > some number
        count += 1
        input_vector= calculate_input_vector(full_list_tables[i-1],flow_table)
        input_list.append(input_vector)
        print('{0:.2f}%:'.format(count/(len(full_list_tables)+1)*100), input_vector)
        print()
        # for feature in input_vector:
        #     # if this is the first time
        #     if feature not in feature_data_list:
        #         feature_data_list[feature]=[]
        #     feature_data_list[feature].append(input_vector[feature])

filename = 'output/flowtables-features'
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, 'wb') as outfile:
    pickle.dump(input_list,outfile)

# normalize(feature_data_list)
# normalized_list_tables=[]
# for i in range(len(feature_data_list['packets'])):
#     normalized_table={}
#     for feature in feature_data_list:
#         normalized_table[feature]=feature_data_list[feature][i]
#     normalized_list_tables.append(normalized_table)
#     print(normalized_table)
#
# filename = 'output/normalized-flowtables-features'
# os.makedirs(os.path.dirname(filename), exist_ok=True)
# with open(filename, 'wb') as outfile:
#     pickle.dump(normalized_list_tables,outfile)
