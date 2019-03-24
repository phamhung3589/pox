#!/usr/bin/python

# standard includes
from pox.core import core
import time as t
import pickle
import pandas as pd
import numpy as np
import threading
# import minisom
import local_outlier_factor as lof_alg
import detect_udp_attack
import detect_tcp_syn
import Backprop
import mode
# include as part of the betta branch
from pox.openflow.of_json import *

global packets, statistic, ip, count
packets = pd.DataFrame([])
count = 0
log = core.getLogger()
knn = detect_udp_attack.knn
tcp_syn = detect_tcp_syn.knn
parameters = Backprop.neural
mod = mode.mod
# handler for timer function that sends the requests to all the
# switches connected to the controller.
def _timer_func ():
  for connection in core.openflow._connections.values():
    connection.send(of.ofp_stats_request(body=of.ofp_flow_stats_request()))
  log.debug("Sent %i flow/port stats request(s)", len(core.openflow._connections))


def calculate_Entropy(df):
    # print("count=",count)
    prob = df/df.sum(axis=0)
    entropy = (-prob*np.log2(prob)).sum()
    return entropy


# Using this function to detect UDP attack
def normalize(vector):
    for i in vector.index:
      vector[i] = (vector[i] - min_feature[i])/(max_feature[i] - min_feature[i])

# Using this dunction to detect ICMP attack
def normalize_icmp(vector):
    for i in vector.index:
        vector[i] = (vector[i] - min_feature_icmp[i])/ (max_feature_icmp[i] - min_feature_icmp[i])

# Normalize feature in anomaly detection - Using algorithm: Local outlier factor
def normalize_minmax(vector):
    for i in vector.index:
        vector[i] = (vector[i] - std[i])/ (mean[i] - std[i])

# handler to display flow statistics received in JSON format
# structure of event.stats is defined by ofp_flow_stats()
def _handle_flowstats_received (event):
  stats = flow_stats_to_list(event.stats)
  log.debug("FlowStatsReceived from %s: %s", 
    dpidToStr(event.connection.dpid), stats)
  global cnt, df1, df2, arr1, arr2, change_mode, tcp_activation

  if event.connection.dpid:
    if change_mode==mod.MODE_NORMAL or \
       change_mode==mod.MODE_CLASSIFIER or \
       change_mode==mod.MODE_DETECT_ICMP or \
       change_mode==mod.MODE_DETECT_UDP:

        status(event)

    elif change_mode==mod.MODE_DETECT_TCPSYN:
        print "detect TCP SYN attack using KNN"
        tcp_activation = 1



def status(event):
    global cnt, arr1, arr2, df1, df2, change_mode
    print "dpid=", event.connection.dpid
    # print stats
    print "************************************************"
    n1 = t.time()  ###########################n1#####################
    flowtable = []
    for f in event.stats:
        flowtable.append([f.packet_count,
                          str(f.match.nw_src),
                          str(f.match.nw_dst),
                          str(f.match.tp_src),
                          str(f.match.tp_dst),
                          f.match.nw_proto])

    lenFlow = len(flowtable)
    n2 = t.time()  ###########################n2-n1 = getflow#####################
    print "n2-n1=", n2 - n1
    if (lenFlow != 0) and (cnt < 1):
        print'cnt=', cnt
        cnt = cnt + 1
        arr1 = np.array(flowtable)
        df1 = pd.DataFrame(arr1, columns=['total_pkt',
                                          'ip_src',
                                          'ip_dst',
                                          'port_src',
                                          'port_dst',
                                          'proto'])

        df1['total_pkt'] = df1['total_pkt'].astype(np.float)

    elif (lenFlow != 0) and (cnt >= 1):
        n3 = t.time()  ######################################################### n3 ########
        print('cnt=', cnt)
        cnt = cnt + 1
        arr2 = np.array(flowtable)
        df2 = pd.DataFrame(arr2, columns=['total_pkt',
                                          'ip_src',
                                          'ip_dst',
                                          'port_src',
                                          'port_dst',
                                          'proto'])

        new_flows = pd.DataFrame(columns=['ip_src', 'total_pkt'])
        df2['total_pkt'] = df2['total_pkt'].astype(np.float)
        ###########               add diff IP_src,IP_dst
        ################ v.2 #######################
        df1.set_index(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'proto'], inplace=True)
        df2.set_index(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'proto'], inplace=True)
        s = df2.loc[df2.index.difference(df1.index), :]
        s.reset_index(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'proto'], inplace=True)
        # s.drop(['ip_dst', 'port_src', 'port_dst', 'proto'], axis=1, inplace=True)
        new_flows = new_flows.append(s)

        # total packet new > total packet old => new flow => add
        common = df2.index.intersection(df1.index)
        s = df2.loc[common]
        s1 = s - df1.loc[common]
        s2 = s1[s1['total_pkt'] > 0]
        s2.reset_index(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'proto'], inplace=True)
        # s2.drop(['ip_dst', 'port_src', 'port_dst', 'proto'], axis=1, inplace=True)
        new_flows = new_flows.append(s)

        # total packet new < total packet old => new flow => add
        s = s[s1['total_pkt'] < 0]
        s.reset_index(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'proto'], inplace=True)
        # s.drop(['ip_dst', 'port_src', 'port_dst', 'proto'], axis=1, inplace=True)
        new_flows = new_flows.append(s)
        # print "hung", new_flows
        df1.reset_index(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'proto'], inplace=True)
        df2.reset_index(['ip_src', 'ip_dst', 'port_src', 'port_dst', 'proto'], inplace=True)

        print "NUMBER OF NEW FLOWS = ", len(new_flows)
        ent_ip_src = calculate_Entropy(new_flows.groupby(['ip_src'])['total_pkt'].sum())
        total_packets = new_flows['total_pkt'].sum()
        # print "Hung", new_flows

        # Mode normal - processing with local outlier factor
        if change_mode == mod.MODE_NORMAL:
            feature_vector = pd.Series([ent_ip_src, total_packets], index=['ent_ip_src', 'total_packets'])
            print "Feature list \n ", feature_vector
            normalize_minmax(feature_vector)
            # print "Feature list \n ", feature_vector
            tobe_classifed = feature_vector.values
            change_mode = lof_alg.lof1.predict(tobe_classifed)[0]
            if change_mode == 0:
                print "Network is safe"

        # mode classifier
        if change_mode==mod.MODE_CLASSIFIER:
            print "Dangerous!!!!\nDangerous!!!!\nChange controller to mode classification"
            new_flows['proto'] = new_flows['proto'].astype(np.float)
            classifier = new_flows.groupby('proto')['total_pkt'].sum()
            classifier = classifier.loc[[1, 17, 6]]
            if classifier.loc[1] > mod.THRESHOLD_ICMP:
                change_mode = mod.MODE_DETECT_ICMP
                print "Suspect ICMP attack - change controller to mode detect ICMP attack"

            elif classifier.loc[17] > mod.THRESHOLD_UDP:
                change_mode = mod.MODE_DETECT_UDP
                print "Suspect UDP attack - change controller to mode detect UDP attack"

            elif classifier.loc[6] > mod.THRESHOLD_TCP_SYN:
                change_mode = mod.MODE_DETECT_TCPSYN
                print "Suspect TCP SYN attack - change controller to mode detect TCP SYN attack"

            else:
                change_mode = mod.MODE_NORMAL
                print "Not detect attack - change controller to mode normal"


        # mode detect udp attack
        elif change_mode==mod.MODE_DETECT_UDP:
            print "detect UDP attack using KNN"
            ent_tp_src = calculate_Entropy(new_flows.groupby(['port_src'])['total_pkt'].sum())
            ent_tp_dst = calculate_Entropy(new_flows.groupby(['port_dst'])['total_pkt'].sum())
            ent_packet_type = calculate_Entropy(new_flows.groupby(['proto'])['total_pkt'].sum())

            feature_vector = pd.Series([ent_ip_src, ent_tp_src, ent_tp_dst, ent_packet_type, total_packets],
                                       index=['ent_ip_src',
                                              'ent_tp_src',
                                              'ent_tp_dst',
                                              'ent_packet_type',
                                              'total_packets'])

            print "Feature list \n ", feature_vector
            normalize(feature_vector)
            tobeClassifed = feature_vector.values
            change_mode = knn.calculate(tobeClassifed)
            if change_mode == 1:
                change_mode += 2
                print " UDP Attack!!!\n UDP Attack!!!\n UDP Attack!!!"
            else:
                print "Relax... It's a mistake"


        # detect ICMP attack using deep learning
        elif change_mode==mod.MODE_DETECT_ICMP:
            print "Detect ICMP attack using deep learning"
            ent_tp_src = calculate_Entropy(new_flows.groupby(['port_src'])['total_pkt'].sum())
            ent_tp_dst = calculate_Entropy(new_flows.groupby(['port_dst'])['total_pkt'].sum())
            ent_packet_type = calculate_Entropy(new_flows.groupby(['proto'])['total_pkt'].sum())

            feature_vector = pd.Series([ent_ip_src, ent_tp_src, ent_tp_dst, ent_packet_type, total_packets],
                                       index=['ent_ip_src',
                                              'ent_tp_src',
                                              'ent_tp_dst',
                                              'ent_packet_type',
                                              'total_packets'])

            print "Feature list \n ", feature_vector
            normalize_icmp(feature_vector)
            tobeClassifed = np.reshape(feature_vector.values, (-1,1))
            change_mode = Backprop.predict_realtime(tobeClassifed, parameters)
            if change_mode == 1:
                change_mode += 1
                msg = of.ofp_flow_mod()
                msg.priority = mod.PRIORITY
                msg.match.dl_type = 0x800
                msg.match.nw_proto = 1
                for connection in core.openflow.connections:
                    connection.send(msg)
                print " ICMP Attack!!!\n ICMP Attack!!!\n ICMP Attack!!!"
            else:
                print "Relax... it's a mistake"

        # print "df1: ", df1
        # print "df2: ", df2
        df1 = df2.copy()
        n12 = t.time()  ################  n12-n1 = total time ##############
        req2rep = n12 - n1
        print('From Request to Reply =', req2rep)


def _tcp_status(event):
    global packets, ip, count, change_mode
    if tcp_activation:
        global packets, start
        table = []
        packet = event.parsed
        # tcp = packet.find('tcp')
        if packet.find('tcp') and packet.find('tcp').SYN and packet.find('tcp').ACK == False:
            table.append([of.ofp_match.from_packet(packet).tp_src,
                          # of.ofp_match.from_packet(packet).tp_dst,
                          of.ofp_match.from_packet(packet).nw_src,
                          # of.ofp_match.from_packet(packet).nw_dst
                          ])

            if len(packets) == 0:
                packets = pd.DataFrame(table,
                                       columns=['source_port','IP_source'])
            else:
                new_packets = pd.DataFrame(table,
                                           columns=['source_port', 'IP_source',])
                packets = packets.append(new_packets, ignore_index=True)

            ip = pd.read_csv('test.csv')
            # ip_update = ip['IP_source'].as_matrix()
            # if str(of.ofp_match.from_packet(packet).nw_src) in ip_update:
            #     packet_out = of.ofp_packet_out()
            #     # flow_mod = of.ofp_flow_mod()
            #     packet_out.buffer_id = event.ofp.buffer_id
            #     packet_out.match = of.ofp_match.from_packet(packet)
            #     # packet_out.data = event.data
            #     # packet_out.in_port = event.port
            #     event.connection.send(packet_out)
        # elif count > 2:
        #     change_mode = 0
        if t.time() - start >= y5:
            count += 1
            # print len(ip)
            start = t.time()
            thread1 = threading.Thread(target=processing_statistic, args=(packets,))
            thread1.start()
            packets = pd.DataFrame([])


def processing_statistic(pk):
    global statistic, change_mode, tcp_activation, count
    global ip
    if len(pk) != 0:
        statistic = pk
        # print(statistic)
        statistic['destination_port'] = 1
        new_statistic = statistic.groupby(['IP_source', 'source_port']).count()
        new_statistic_2 = statistic.drop(['source_port'], axis=1).groupby(['IP_source']).count()
        # print new_statistic_2
        new_statistic = new_statistic/new_statistic_2
        new_statistic = -new_statistic*np.log2(new_statistic)
        new_statistic = new_statistic.groupby(['IP_source']).sum()

        # #### using algorithm
        # # print new_statistic
        # tobeClassifier = new_statistic.as_matrix()
        # # print tobeClassifier.shape
        # labels = tcp_syn.calculate_batch(tobeClassifier)
        # labels = np.array(labels)
        # print labels
        # change_mode = np.max(labels)*4

        #### Not using algorithm
        new_statistic = new_statistic.iloc[new_statistic['destination_port'].as_matrix() > 5, :]
        change_mode = int(len(new_statistic) != 0)*4
        print "change_mode=", change_mode
        if change_mode == mod.MODE_NORMAL and count > 2:
            tcp_activation = 0
            count = 0
        else:
            print "TCP-SYN attack!!!\nTCP-SYN attack!!!\nTCP-SYN attack!!!"
            # new_statistic = new_statistic.iloc[labels == 1, :]
            # count = 0
            if len(ip) != 0 and len(new_statistic) != 0:
                # ip.set_index('IP_source', inplace=True)
                # new_statistic['IP_source'] = new_statistic['IP_source'].astype(str)
                new_statistic.index = new_statistic.index.astype(str)
                new_statistic = new_statistic.loc[new_statistic.index.difference(ip.index), ]

                for ip_src in new_statistic.index:
                    msg = of.ofp_flow_mod()
                    msg.priority = mod.PRIORITY
                    msg.match.dl_type = 0x800
                    msg.match.nw_src = IPAddr(ip_src)
                    for connection in core.openflow.connections:
                        connection.send(msg)

                with open('test.csv', 'a') as f:
                    new_statistic.to_csv(f, encoding='utf-8', header=False)
            print "=================================================="


# main functiont to launch the module
def launch ():
  from pox.lib.recoco import Timer
  global start, cnt, mean, std, max_feature, min_feature, max_feature_icmp, min_feature_icmp
  global change_mode, tcp_activation
  tcp_activation = 0
  change_mode = 0
  cnt=0
  start = t.time()
  # mean = pd.read_pickle("./somInput/meanStats")
  # std = pd.read_pickle("./somInput/stdStats")
  mean = pd.read_pickle("./somInput/maxFeature")
  std = pd.read_pickle("./somInput/minFeature")
  max_feature = pd.read_pickle("./somInput/max_feature.pickle")
  min_feature = pd.read_pickle("./somInput/min_pickle.pickle")
  max_feature_icmp = pd.read_pickle("./somInput/max_feature_icmp")
  min_feature_icmp = pd.read_pickle("./somInput/min_feature_icmp")
  print 'start=', start
  # attach handsers to listners
  core.openflow.addListenerByName("FlowStatsReceived", 
    _handle_flowstats_received)
  core.openflow.addListenerByName("PacketIn",
                                  _tcp_status)
  # timer set to execute every five seconds
  Timer(5, _timer_func, recurring=True)
