import numpy as np
import matplotlib.pyplot as plt
import csv

data = None
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    data = list(csv_reader)
    data = np.array(data,dtype=np.float32)

# hist2d =np.histogram2d(data[:,0],data[:,1],bins=[10,10])[0]
plt.hist2d(data[:,0],data[:,1],bins=[100,100
                                     ])
plt.show()