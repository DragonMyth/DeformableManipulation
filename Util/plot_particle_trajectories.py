import matplotlib.pyplot as plt


import numpy as np
from tkinter.filedialog import askopenfilename
import joblib

openFileOption = {}
openFileOption['initialdir'] = '../data/trajs'

# filename = askopenfilename(**openFileOption)
filename = askopenfilename(**openFileOption)
traj_info = joblib.load(filename)

print("File Loaded!")


