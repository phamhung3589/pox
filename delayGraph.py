import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import pandas as pd
df = pd.read_pickle("statsForKhoa")
# plot delay
# y1 = df.req2rep[df.tableSize>=2600]
# y2 = df.req2rep[df.tableSize<2600]
# y2 = y2.append(pd.Series([y1.iloc[0]],index=[y1.index[0]]))
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.set_title("Request to Reply Delay")
# ax1.set_xlabel("Time (seconds)")
# ax1.set_ylabel("Delay (seconds)")
# x1 = y1.index*5
# x2 = y2.index*5
# ax1.plot(x2,y2,c='b',marker = 'o', linewidth = 4.0, label = "During Normal")
# ax1.plot(x1,y1,c='r',marker = 'o', linewidth = 4.0, label = "During Attack")
# leg = ax1.legend()
# plt.show()

# plot tableSize
y1 = df.tableSize[df.tableSize>=2600]
y2 = df.tableSize[df.tableSize<2600]
y2 = y2.append(pd.Series([y1.iloc[0]],index=[y1.index[0]]))
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Flow Table Size")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Tables Size (flows)")
x1 = y1.index*5
x2 = y2.index*5
ax1.plot(x2,y2,c='b',marker = 'D',linewidth = 4.0, label = "During Normal")
ax1.plot(x1,y1,c='r',marker = 'D',linewidth = 4.0, label = "During Attack")
leg = ax1.legend(bbox_to_anchor=(0.45, 0.85), bbox_transform=plt.gcf().transFigure)
# bbox_to_anchor=( x_axis, y_axis), to the left is smaller, lower is smaller
plt.show()
