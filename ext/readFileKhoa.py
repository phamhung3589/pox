# import pickle 
# from pox.core import core
# from pox.lib.util import dpidToStr
# import pox.openflow.libopenflow_01 as of
import pandas as pd
import numpy as np
import time as t
import pickle
# include as part of the betta branch
# from pox.openflow.of_json import *
# from pox.lib import address
# with open('./outDDOS/test', 'rb') as outfile:
#   test = pickle.load(outfile)
#   print 'So flow table=',len(test)
#   print 'FlowTable\n',test.nw_proto
# featureTable = pd.DataFrame(index=range(0,1000),columns=['ent_ip_src','ent_tp_src','ent_tp_dst','ent_packet_type','total_packets'])
# feature_vector = pd.Series([23,2322,1,0,1999], index=['ent_ip_src', 'ent_tp_src', 'ent_tp_dst', 'ent_packet_type', 'total_packets'])
# featureTable.loc[0]= feature_vector
# featureTable.loc[1]= feature_vector
# print ('FUCK\n=',featureTable)
# pickle.dump(featureTable,open('./outputFeature/testFile','wb'))
# 20160318-feature
# with open('./outputFeature/testFile', 'rb') as outfile:
# with open('./outputFeature/20160321-feature', 'rb') as outfile:
def normalize(dframe):
    """
    normalize a dictionary, each key (feature) contains the list of all values of that feature
    """
    # result={}
    # for i in range(len(dframe.columns)):
    #     m = mean[dframe.columns[i]]
    #     s = std[dframe.columns[i]]
    #     x = dframe[dframe.columns[i]]
    #     x = 0.5*(np.tanh(0.1*(x-m)/s)+1)
    #     dframe[dframe.columns[i]] = x

    for column in dframe:
    	mean = dframe[column].mean()
    	std = dframe[column].std()
    	dframe[column] = 0.5*(np.tanh(0.1*(dframe[column]-mean)/std)+1)




featureTable=pd.DataFrame(columns=['ent_ip_src','ent_tp_src','ent_tp_dst','ent_packet_type','total_packets'])
featureTableNor = pd.read_pickle('./outputFeature/20160321-feature1-Nor')
featureTableAtk = pd.read_pickle('./outputFeature/20160321-feature1-Atk')
############# ATTACK
# print('So vector=',len(test1))
# print('FeatureTable\n',test1)
################# BEFORE extend
#######################Nor###########
# >>> read.test1.mean()
# ent_ip_src            2.639206
# ent_tp_src            0.456511
# ent_tp_dst            0.745057
# ent_packet_type       0.296004
# total_packets      1735.408304

# >>> read.test1.std()
# ent_ip_src           0.402509
# ent_tp_src           0.369411
# ent_tp_dst           1.345285
# ent_packet_type      0.281048
# total_packets      515.211236
#########################Atk#########
# >>> read.test1.mean()
# ent_ip_src             11.878608
# ent_tp_src              0.049518
# ent_tp_dst              0.043321
# ent_packet_type         0.025283
# total_packets      321888.144928

# >>> read.test1.std()
# ent_ip_src              0.296927
# ent_tp_src              0.020231
# ent_tp_dst              0.017256
# ent_packet_type         0.009422
# total_packets      107375.796310
##################################
N=1000
columns = ['ent_ip_src','ent_tp_src','ent_tp_dst','ent_packet_type','total_packets']
df= pd.DataFrame(columns=columns)
for column in df:
	df[column]= np.random.randn(N)
	mean_tmp = df[column].mean()
	std_tmp = df[column].std()
	df[column] = (df[column] - mean_tmp) / std_tmp
 
	tweak = 1                                        #### ADJUST VARIATION HERE
	std_desired = featureTableAtk[column].std()*tweak                                
	mean_desired = featureTableAtk[column].mean()
	df[column] = (df[column] * std_desired) + mean_desired

featureTableAtk = featureTableAtk.append(df)
print "mean = \n", featureTableAtk.mean()
print "std = \n", featureTableAtk.std()
# print(featureTableAtk)
######## new mean and std ############
# featureTableAtk["AtkNor"]=1                                 #### MARK as ATTACK
# with open('./outputFeature/20160321-time', 'rb') as outfile:
test2 = pd.read_pickle('./outputTime/20160322-time1-Atk')


######################## NORMAL
N=1000
columns = ['ent_ip_src','ent_tp_src','ent_tp_dst','ent_packet_type','total_packets']
df= pd.DataFrame(columns=columns)
for column in df:
	df[column]= np.random.randn(N)
	mean_tmp = df[column].mean()
	std_tmp = df[column].std()
	df[column] = (df[column] - mean_tmp) / std_tmp

	tweak=1
	std_desired = featureTableNor[column].std()*tweak
	mean_desired = featureTableNor[column].mean()
	df[column] = (df[column] * std_desired) + mean_desired

featureTableNor = featureTableNor.append(df)
print "mean = \n", featureTableNor.mean()
print "std = \n", featureTableNor.std()
######## new mean and std ############
# print(featureTableNor)														#### PRINT CSV
# featureTableNor["AtkNor"]=0                                #### MARK as NORMAL

# with open('./outputFeature/20160321-time', 'rb') as outfile:
test2 = pd.read_pickle('./outputTime/20160321-time1-Nor')

featureTable=featureTable.append(featureTableNor)
featureTable=featureTable.append(featureTableAtk)
meanStats = featureTable.mean()
stdStats  = featureTable.std()
meanStats.to_pickle("meanStats") 
stdStats.to_pickle("stdStats")
normalize(featureTable)

featureTable["AtkNor"]=0
length=len(featureTable)
featureTable.AtkNor[1289:length]=1
featureTable.to_csv("normalizedFeature.csv",index=False)                          