#!/usr/bin/python
# Copyright 2012 William Yu
# wyu@ateneo.edu
#
# This file is part of POX.
#
# POX is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# POX is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with POX. If not, see <http://www.gnu.org/licenses/>.
#

"""
This is a demonstration file created to show how to obtain flow 
and port statistics from OpenFlow 1.0-enabled switches. The flow
statistics handler contains a summary of web-only traffic.
"""

# standard includes
from pox.core import core
from pox.lib.util import dpidToStr
import pox.openflow.libopenflow_01 as of
import time as t
import pickle
import pandas as pd 
import numpy as np
import copy
import minisom
# include as part of the betta branch
from pox.openflow.of_json import *

from Tkinter import *
root = Tk()
import tkMessageBox

var =  StringVar()
var.set('NoRMaL')
color = StringVar()
networkStatus = Label(root,textvariable=var,fg= "red",font=("Helvetica", 70))
networkStatus.pack()

log = core.getLogger()
n1=0
n2=0
table=[]
# handler for timer function that sends the requests to all the
# switches connected to the controller.
def _timer_func ():
  for connection in core.openflow._connections.values():
    global n1
    #print 'n1a=',n1
    connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
    global n0
    n0=t.time()
    #connection.send(of.ofp_stats_request(body=of.ofp_port_stats_request()))
  log.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))

# calculate entropy
def calculate_Entropy(df):
    # print("count=",count)
    prob = df/df.sum(axis=0)
    entropy = (-prob*np.log2(prob)).sum()
    return entropy

def normalize(vector):
    for i in vector.index:
      vector[i] = 0.5*(np.tanh(0.1*(vector[i]-mean[i])/std[i])+1)

# handler to display flow statistics received in JSON format
# structure of event.stats is defined by ofp_flow_stats()
def _handle_flowstats_received (event):
  stats = flow_stats_to_list(event.stats)
  log.debug("FlowStatsReceived from %s: %s", 
    dpidToStr(event.connection.dpid), stats)
  global cnt,df1,df2 
  global arr1,arr2
  if (event.connection.dpid):
    print "dpid=", event.connection.dpid
    # print stats
    print "************************************************"
    n1=t.time()                              ###########################n1#####################
    flowtable=[]
    for f in event.stats:  
      flowtable.append([hex(f.cookie),
                        # f.duration_sec,
                        # f.duration_nsec,
                        # f.table_id,
                        # f.priority,
                        f.packet_count,
                        # f.byte_count,
                        # f.idle_timeout,
                        # f.hard_timeout,
                        f.match.nw_proto,
                        # f.match.in_port,
                        # f.match.dl_vlan,
                        # f.match.dl_vlan_pcp,
                        # str(f.match.dl_src),
                        # str(f.match.dl_dst),
                        str(f.match.nw_src),
                        str(f.match.nw_dst),
                        # str(f.match.nw_tos),
                        str(f.match.tp_src),
                        str(f.match.tp_dst)]) 
    lenFlow = len(flowtable)
    n2=t.time()                              ###########################n2-n1 = getflow#####################
    print("n2-n1=",n2-n1)
    if (lenFlow!=0) and (cnt<1):
      print('cnt=',cnt)
      cnt = cnt + 1
      arr1 = np.array(flowtable)
      df1 = pd.DataFrame(arr1, columns=['cookie',
                    # 'duration_sec',
                    # 'duration_nsec',
                    # 'table_id',
                    # 'priority',
                    'packet_count',
                    # 'byte_count',
                    # 'idle_timeout',
                    # 'hard_timeout',
                    'nw_proto',
                    # 'in_port',
                    # 'dl_vlan',
                    # 'dl_vlan_pcp',
                    # 'dl_src',
                    # 'dl_dst',
                    'nw_src',
                    'nw_dst',
                    # 'nw_tos',
                    'tp_src',
                    'tp_dst'])
      df1.packet_count = df1.packet_count.astype(np.float)
    elif (lenFlow!=0) and (cnt >=1):
      n3=t.time()           ######################################################### n3 ########
      print('cnt=',cnt)
      cnt = cnt+1
      arr2 = np.array(flowtable)  
      df2 = pd.DataFrame(arr2, columns=['cookie',
                    # 'duration_sec',
                    # 'duration_nsec',
                    # 'table_id',
                    # 'priority',
                    'packet_count',
                    # 'byte_count',
                    # 'idle_timeout',
                    # 'hard_timeout',
                    'nw_proto',
                    # 'in_port',
                    # 'dl_vlan',
                    # 'dl_vlan_pcp',
                    # 'dl_src',
                    # 'dl_dst',
                    'nw_src',
                    'nw_dst',
                    # 'nw_tos',
                    'tp_src',
                    'tp_dst'])
      # new_flows = pd.DataFrame(columns=['cookie',
      #               # 'duration_sec',
      #               # 'duration_nsec',
      #               # 'table_id',
      #               # 'priority',
      #               'packet_count',
      #               # 'byte_count',
      #               # 'idle_timeout',
      #               # 'hard_timeout',
      #               'nw_proto',
      #               # 'in_port',
      #               # 'dl_vlan',
      #               # 'dl_vlan_pcp',
      #               # 'dl_src',
      #               # 'dl_dst',
      #               'nw_src',
      #               'nw_dst',
      #               # 'nw_tos',
      #               'tp_src',
      #               'tp_dst'])
      new_flows = pd.DataFrame(columns=['nw_src',
                    # 'duration_sec',
                    # 'duration_nsec',
                    # 'table_id',
                    # 'priority',
                    'nw_dst',
                    # 'byte_count',
                    # 'idle_timeout',
                    # 'hard_timeout',
                    'cookie',
                    # 'in_port',
                    # 'dl_vlan',
                    # 'dl_vlan_pcp',
                    # 'dl_src',
                    # 'dl_dst',
                    'packet_count',
                    'nw_proto',
                    # 'nw_tos',
                    'tp_src',
                    'tp_dst'])
      df2.packet_count = df2.packet_count.astype(np.float)
      # print 'df1 \n', df1
      # print 'df2 \n', df2
      ############               add diff IP_src,IP_dst

################# v.2 #######################
      df1.set_index(['nw_src', 'nw_dst'], inplace=True)
      df2.set_index(['nw_src', 'nw_dst'], inplace=True)
      combined= df2.join(df1,lsuffix='_x',rsuffix='_y')
      # print "S=\n",s
      s=combined.loc[np.isnan(combined.packet_count_y)]
      s.reset_index(level=['nw_src', 'nw_dst'], inplace=True)
      s=s.drop('tp_src_y',axis=1)
      s=s.drop('tp_dst_y',axis=1)
      s=s.drop('cookie_y',axis=1)
      s=s.drop('packet_count_y',axis=1)
      s=s.drop('nw_proto_y',axis=1)
      s.columns=['nw_src','nw_dst','cookie','packet_count','nw_proto','tp_src','tp_dst']
      new_flows=new_flows.append(s)
      # print "NEW=\n",new_flows
      df1.reset_index(level=['nw_src', 'nw_dst'], inplace=True)
      df2.reset_index(level=['nw_src', 'nw_dst'], inplace=True)
################# v.2 #######################

      ################################## Same IP, port  & protocol differerent ***********************************
      
      same1same2=combined.loc[(combined['tp_src_x']==combined['tp_src_y']) &  (combined['tp_dst_x']==combined['tp_dst_y']) & (combined['nw_proto_x']==combined['nw_proto_y'])]                ##### get all same entries same1 + same2
      same2 = pd.concat([same1same2['cookie_x'],same1same2['packet_count_x'],same1same2['nw_proto_x'],same1same2['tp_src_x'],same1same2['tp_dst_x']], axis=1, keys=['cookie','packet_count','nw_proto','tp_src','tp_dst'])               ################ take out same 2 from same1+same2
      same2.reset_index(level=['nw_src','nw_dst'],inplace=True)            ### reset index  before change again
      df2.set_index(['nw_src','nw_dst','nw_proto','tp_src','tp_dst'],inplace=True)           ### set new index, find new record (new flow) based on this , REMEMBER TO RESET IT AGAIN !!!!!!
      same2.set_index(['nw_src','nw_dst','nw_proto','tp_src','tp_dst'],inplace=True)
      result2 = df2.join(same2,lsuffix='_x',rsuffix='_y')     ###### same2.join  or  df2.join ????    form table with same index from both table df2,same2  
      result2=result2.loc[np.isnan(result2.packet_count_y)]          ##### record contain NaN is new record 
      result2.reset_index(level=['nw_src','nw_dst','nw_proto','tp_src','tp_dst'], inplace=True)
      result2=result2.drop('cookie_y',axis=1)
      result2=result2.drop('packet_count_y',axis=1)
      result2.columns=['nw_src','nw_dst','nw_proto','tp_src','tp_dst','cookie','packet_count']
      # print "RESULT2= FUCK FUCK FUCK\n",result2
      new_flows=new_flows.append(result2)
      # print "NEW=\n",new_flows
      df2.reset_index(['nw_src','nw_dst','nw_proto','tp_src','tp_dst'],inplace=True)
      # print "DF2 =\n",df2
      # ##########################################          New pkt count < old pkt count => new flow => add
      # print "MERG=",merg
      merg=pd.merge(df1, df2, on=['nw_src', 'nw_dst'], how='inner')
      # print "MERGE= \n",merg
      df=merg.loc[(merg['tp_src_x']==merg['tp_src_y']) &  (merg['tp_dst_x']==merg['tp_dst_y']) & (merg['nw_proto_x']==merg['nw_proto_y']) & (merg['packet_count_y']<merg['packet_count_x'])]
      # print "DF=",df
      df=df.drop('tp_src_x',axis=1)
      df=df.drop('tp_dst_x',axis=1)
      df=df.drop('cookie_x',axis=1)
      df=df.drop('packet_count_x',axis=1)
      df=df.drop('nw_proto_x',axis=1)
      df.columns=['nw_src','nw_dst','nw_proto','tp_src','tp_dst','cookie','packet_count'] 
      new_flows=new_flows.append(df)
      # print "NEW=\n",new_flows
      # ###########################################          new pkt count > old => more pkt of same flow => add
      # print merg
      df=merg.loc[(merg['tp_src_x']==merg['tp_src_y']) &  (merg['tp_dst_x']==merg['tp_dst_y']) & (merg['nw_proto_x']==merg['nw_proto_y']) & (merg['packet_count_y']>merg['packet_count_x'])]
      df.packet_count_y=df.packet_count_y-df.packet_count_x-1
      df=df.drop('tp_src_x',axis=1)
      df=df.drop('tp_dst_x',axis=1)
      df=df.drop('cookie_x',axis=1)
      df=df.drop('packet_count_x',axis=1)
      df=df.drop('nw_proto_x',axis=1)
      df.columns=['nw_src','nw_dst','nw_proto','tp_src','tp_dst','cookie','packet_count'] 
      new_flows=new_flows.append(df)
      # new_flows.reset_index(drop=True)            ######## reset index new flows
      # print "NEW=\n",new_flows
      print "DF2 length =",len(df2)
      print "NUMBER OF NEW FLOWS = ",len(new_flows)
      ent_ip_src = calculate_Entropy(new_flows.groupby(['nw_src'])['packet_count'].sum())
      ent_tp_src = calculate_Entropy(new_flows.groupby(['tp_src'])['packet_count'].sum())
      ent_tp_dst = calculate_Entropy(new_flows.groupby(['tp_dst'])['packet_count'].sum())
      ent_packet_type = calculate_Entropy(new_flows.groupby(['nw_proto'])['packet_count'].sum())
      total_packets = new_flows.packet_count.sum()
      feature_vector = pd.Series([ent_ip_src,ent_tp_src,ent_tp_dst,ent_packet_type,total_packets], index=['ent_ip_src', 'ent_tp_src', 'ent_tp_dst', 'ent_packet_type', 'total_packets'])
      print "Feature list \n ", feature_vector
      feature_vector.to_pickle('feature_vector')
      normalize(feature_vector)
      tobeClassifed = feature_vector.values
      print "ATK OR NOR ? \t",minisom.som.what_type(tobeClassifed)
      if minisom.som.what_type(tobeClassifed):
          for i in range(0,3):
              print "ATTACK!!!"
      #     var.set('AtTack')
      # else:
      #     var.set('NoRMaL')
      # root.update_idletask()
      df1 = df2.copy()
      n12 = t.time()     ################  n12-n1 = total time ##############
      req2rep = n12 - n1
      print('From Request to Reply =',req2rep)
      print('total delay =', n12 -n1)
      rowForDuc = pd.Series([req2rep,len(df2)], index = ['req2rep','tableSize'])
      # print('row Duc = ',rowForDuc)
      global statsForDuc
      statsForDuc = statsForDuc.append(rowForDuc, ignore_index = True)
      # print('statsForDuc =',statsForDuc)
      if cnt==100:
          statsForDuc.to_pickle("statsForDuc")
      # if total_packets>=10000:
      #   print("ATTACK DETECTED!!!! \n ATTACK DETECTED !!!!")
      # if (cnt < 145) or (cnt>145):                       
      #   featureTable.loc[cnt-2] = feature_vector  
      #   timeTable.loc[cnt-2] = timeVector
      #   # pd.Series(np.array([0]), index= range(0,299))
      # elif cnt==145:                   ### 1hour mark =730
      #   featureTable.loc[cnt-2]= feature_vector  
      #   timeTable.loc[cnt-2] = timeVector
      #   printItem1 = featureTable
      #   printItem2 = timeTable
      #   printItem1.to_pickle('./ext/outputFeature/20160322-feature1-Atk')
      #   printItem2.to_pickle('./ext/outputTime/20160322-time1-Atk')
      # elif cnt==290:                   ### 1hour mark =730
      #   featureTable.loc[cnt-2]= feature_vector  
      #   timeTable.loc[cnt-2] = timeVector
      #   printItem1 = featureTable
      #   printItem2 = timeTable
      #   printItem1.to_pickle('./ext/outputFeature/20160322-feature2-Atk')
      #   printItem2.to_pickle('./ext/outputTime/20160322-time2-Atk')


# handler to display port statistics received in JSON format
def _handle_portstats_received (event):
  stats = flow_stats_to_list(event.stats)
  log.debug("PortStatsReceived from %s: %s", 
    dpidToStr(event.connection.dpid), stats)
    
# main functiont to launch the module
def launch ():
  from pox.lib.recoco import Timer
  global start,cnt,mean,std
  cnt=0
  mean = pd.read_pickle("./somInput/meanStats")
  std = pd.read_pickle("./somInput/stdStats")
  start=t.time()
  print 'start=',start
  global featureTable,timeTable
  timeTable = pd.DataFrame(columns=['Time','ReqRep_Delay','Cal_Delay'])
  featureTable = pd.DataFrame(columns=['ent_ip_src','ent_tp_src','ent_tp_dst','ent_packet_type','total_packets'])
  global statsForDuc
  statsForDuc = pd.DataFrame(columns=['req2rep','tableSize'])
  # attach handsers to listners
  core.openflow.addListenerByName("FlowStatsReceived", 
    _handle_flowstats_received) 
  core.openflow.addListenerByName("PortStatsReceived", 
    _handle_portstats_received) 
  # timer set to execute every five seconds
  Timer(5, _timer_func, recurring=True)

