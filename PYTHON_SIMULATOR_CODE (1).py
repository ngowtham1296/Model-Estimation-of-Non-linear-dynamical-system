#!/usr/bin/env python
import mujoco_py
import os
import numpy as np
import csv
import pandas as pd
import xlsxwriter
mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'cartpole1.xml')
model = mujoco_py.load_model_from_path('cartpole1.xml')
sim = mujoco_py.MjSim(model)
viewer = mujoco_py.MjViewer(sim)
sim.data.ctrl[0] = 10

list1 = []

for i in range(1000):
	data = np.c_[sim.data.qpos[0],sim.data.qpos[1],sim.data.qvel[0],sim.data.qvel[1]]
	sim.step()
	list1.append(data)
	viewer.render()


x = list1
DATA = np.array(x)
val = DATA[0:1000,0]
final = np.matrix(val)
print(final)
data = pd.DataFrame(final)
datatoexcel = pd.ExcelWriter("FromPython.xlsx", engine='xlsxwriter')
data.to_excel(datatoexcel, sheet_name='Sheet1')
datatoexcel.save()