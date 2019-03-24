
import numpy as np
from numpy import pi
import pandas as pd
from bokeh.client import push_session
from bokeh.driving import cosine
from bokeh.plotting import figure, curdoc
from bokeh.io import gridplot,vplot,hplot
x = np.linspace(0, 150, 100)
y1 = np.zeros(100)
y2 = np.zeros(100)
y3 = np.zeros(100)
y4 = np.zeros(100)
y5 = np.zeros(100)

# p1 = figure(title="Ent_ip_src")
# p2 = figure(title="Ent_tp_src")
# p3 = figure(title="Ent_tp_dst")
# p4 = figure(title="Ent_packet_type")
p5 = figure(title="Total Packets")
p = figure(title = "5 Selected Flow Feature")

# r = p.line([0, 4*pi], [0, 1], color="yellow")
# r1 = p1.line(x, y1, color="firebrick", line_width=4,legend="ent_ip_src")
# r2 = p2.line(x, y2, color="navy", line_width=4,legend="ent_tp_src")
# r3 = p3.line(x, y3, color="purple", line_width=4,legend="ent_tp_dst")
# r4 = p4.line(x, y4, color="green", line_width=4,legend="ent_packet_type")
r5 = p5.line(x, y5, color="brown", line_width=4,legend="total packets")
r1 = p.line(x, y1, color="firebrick", line_width=2,legend="ent_ip_src")
r2 = p.line(x, y2, color="navy", line_width=2,legend="ent_tp_src")
r3 = p.line(x, y3, color="purple", line_width=2,legend="ent_tp_dst")
r4 = p.line(x, y4, color="green", line_width=2,legend="ent_packet_type")
# r5 = p.line(x, y5, color="brown", line_width=2,legend="total packets")

# p1.xaxis.axis_label = "Time(seconds)"
# p2.xaxis.axis_label = "Time(seconds)"
# p3.xaxis.axis_label = "Time(seconds)"
# p4.xaxis.axis_label = "Time(seconds)"
p5.xaxis.axis_label = "Time(seconds)"
# p1.yaxis.axis_label = "Ent Values"
# p2.yaxis.axis_label = "Ent Values"
# p3.yaxis.axis_label = "Ent Values"
# p4.yaxis.axis_label = "Ent Values"
p5.yaxis.axis_label = "Total Packets (packets)"

p.xaxis.axis_label = "Time(seconds)"
p.yaxis.axis_label = "Ent Values"
# open a session to keep our local document in sync with server
session = push_session(curdoc())

@cosine(w=0.03)
def update(step):
   # r2.data_source.data["y"] = y * step
   # r2.glyph.line_alpha = 1 - 0.8 * abs(step)
    global y1,y2,y3,y4,y5
    y1 = np.delete(y1,0)
    y2 = np.delete(y2,0)
    y3 = np.delete(y3,0)
    y4 = np.delete(y4,0)
    y5 = np.delete(y5,0)

    read = pd.read_pickle('./feature_vector') 
    y1 = np.append(y1,read.ent_ip_src)
    y2 = np.append(y2,read.ent_tp_src)
    y3 = np.append(y3,read.ent_tp_dst)
    y4 = np.append(y4,read.ent_packet_type)
    y5 = np.append(y5,read.total_packets)

    r1.data_source.data["y"] = y1 
    r2.data_source.data["y"] = y2 
    r3.data_source.data["y"] = y3 
    r4.data_source.data["y"] = y4 
    r5.data_source.data["y"] = y5 
    # print "y1=",y1
curdoc().add_periodic_callback(update, 5000)         #millisecs
# grid = vplot([[p1, p2, p3], [p4, p5, None]])
# vertical = vplot(p1,p2,p3,p4,p5)
vertical = hplot(p,p5)
session.show() # open the document in a browser

session.loop_until_closed() # run forever
