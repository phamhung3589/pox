import pickle
import random
import numpy as np


def split(my_list = None, ratio = 2.0/3.0, only_type = None):
    np.random.shuffle(my_list) 
    # print "MYLIST =\n",my_list
    if only_type != None:
        condition_array = my_list[:,-1] == only_type
        type_part = my_list[condition_array]
        nontype_part = my_list[np.logical_not(condition_array)]
        # print "TYPE_PART=", type_part
        # print "NON_TYPE_PART=", nontype_part

        border = int(ratio*len(type_part))
        part1 = type_part[:border,:]
        part2 = np.concatenate((type_part[border:,:],nontype_part[int(ratio*len(nontype_part)):,:]))
        # print "PART 1 =", type_part[:border,:]
        # print "PART 2 =", len(part2)
    else:
        border = int(ratio * len(my_list))
        part1 = my_list[:border,:]
        part2 = my_list[border:,:]

    info(part1)
    info(part2)
    return (part1,part2)

def info(my_list):
    count_atk = np.count_nonzero(my_list[:,-1])
    print('This list has', len(my_list), 'samples, in which', count_atk, 'attack samples, ', len(my_list) - count_atk, 'normal samples')
