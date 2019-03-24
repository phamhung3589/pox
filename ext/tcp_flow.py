from pox.core import core
import time as t
import pandas as pd
import numpy as np
import threading
import pox.openflow.libopenflow_01 as of
from pox.lib.addresses import IPAddr
from mode import mod

# include as part of the betta branch
global packets
global statistic
global ip
packets = pd.DataFrame([])
global count1
count1 = 0
data = pd.DataFrame([pd.read_csv("test.csv").iloc[0,:].as_matrix()], columns=['IP_source', 'destination_port'])
with open("test.csv", "w") as f:
    data.to_csv(f, index=False)
# handler for timer function that sends the requests to all the
# switches connected to the controller.


# def _timer():
#     global packets
#     global values
#     # thread1 = threading.Thread(target=processing_statistic, args=(packets, ))
#     # thread1.start()
#     packets = pd.DataFrame([])
#     core.openflow.addListenerByName("PacketIn",
#                                     _handle_flowstats_received)


def processing_statistic(pk):
    global statistic
    global ip
    if len(pk) != 0:
        statistic = pk
        # print(statistic)
        # statistic[2] = 1
        statistic['destination_port'] = 1
        # new_statistic = statistic.drop(['IP_destination'], axis=1).groupby(['IP_source', 'source_port']).count()
        # new_statistic_2 = statistic.drop(['source_port', 'IP_destination'], axis=1).groupby(['IP_source']).count()
        new_statistic = statistic.groupby(['IP_source', 'source_port']).count()
        new_statistic_2 = statistic.drop(['source_port'], axis=1).groupby(['IP_source']).count()
        # print new_statistic_2
        new_statistic = new_statistic/new_statistic_2
        new_statistic = -new_statistic*np.log2(new_statistic)
        # new_statistic = new_statistic.groupby(['IP_source']).sum()
        new_statistic = new_statistic.groupby(['IP_source']).sum()
        # print new_statistic
        #
    # with open("./outputCSV/tcp_data.csv", "a") as f:
    #     new_statistic['sum_ip'] = new_statistic_2.as_matrix()
    #     new_statistic.reset_index(inplace=True)
    #     new_statistic.drop('IP_source', axis=1, inplace=True)
    #     new_statistic.set_index("sum_ip", inplace=True)
    #     print new_statistic
            # new_statistic.to_csv(f, index=True, header=False)
            #
        new_statistic = new_statistic.iloc[new_statistic['destination_port'].as_matrix() > 5,:]
        if len(ip) != 0 and len(new_statistic) != 0:
            # new_statistic['IP_source'] = new_statistic['IP_source'].astype(str)
            new_statistic.index = new_statistic.index.astype(str)
            new_statistic = new_statistic.loc[new_statistic.index.difference(ip.index), ]
            # print new_statistic

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



# def calculate_Entropy(df):
#     # print("count=",count)
#     prob = df/df.sum(axis=0)
#     entropy = (-prob*np.log2(prob)).sum()
#     return entropy


# handler to display flow statistics received in JSON format
# structure of event.stats is defined by ofp_flow_stats()

def _handle_flowstats_received (event):
    global packets, count1
    global statistic
    global start
    global ip
    table = []
    packet = event.parsed
    # tcp = packet.find('tcp')
    if packet.find('tcp') and packet.find('tcp').SYN and packet.find('tcp').ACK==False:
        table.append([of.ofp_match.from_packet(packet).tp_src,
                      # of.ofp_match.from_packet(packet).tp_dst,
                      of.ofp_match.from_packet(packet).nw_src
                      # of.ofp_match.from_packet(packet).nw_dst
                      ])
        # packets = packets.append(table)
        if len(packets) == 0:
            packets = pd.DataFrame(table, columns=['source_port', 'IP_source'])
        else:
            new_packets = pd.DataFrame(table, columns=['source_port', 'IP_source'])
            packets = packets.append(new_packets, ignore_index=True)

        ip = pd.read_csv('test.csv', index_col="IP_source")
        # ip_update = np.array(ip.index)

        # if str(of.ofp_match.from_packet(packet).nw_src) in ip_update:
        # #     block_ip(event, packet)
        #     count1 += 1
        #     packet_out = of.ofp_packet_out()
        #     # flow_mod = of.ofp_flow_mod()
        #     packet_out.priority = 500
        #     packet_out.buffer_id = event.ofp.buffer_id
        #     packet_out.match = of.ofp_match.from_packet(packet)
        #     # packet_out.data = event.data
        #     # packet_out.in_port = event.port
        #     event.connection.send(packet_out)


    if t.time() - start >= 5.5:
        print count1
        start = t.time()
        thread1 = threading.Thread(target=processing_statistic, args=(packets, ))
        thread1.start()
        packets = pd.DataFrame([])
        # mitigate()
        # for connection in core.openflow.connections:  # _connections.values() before betta
        #     connection.send(of.ofp_flow_mod(command=of.OFPFC_DELETE))



# main functiont to launch the module
def launch():
    global start, cnt, mean, std
    global change_mode
    change_mode = 0
    cnt = 0
    start = t.time()
    print 'start=', start
    core.openflow.addListenerByName("PacketIn",
                                    _handle_flowstats_received)