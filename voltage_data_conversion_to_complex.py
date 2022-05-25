import math
import numpy as np
import pandas as pd


def make_complex_value(angle, vf_module):
    return vf_module * complex(real=math.cos(math.radians(angle)), imag=math.sin(math.radians(angle)))


files_name_list = [f'ByPhasisVoltageData/Phasis-{index}.xlsx' for index in range(1, 4)]

for file_name in files_name_list:
    file = pd.ExcelFile(file_name)
    distortions = pd.read_excel(file, file.sheet_names[0])
    angles = pd.read_excel(file, file.sheet_names[1])

    data = []
    for line in range(0, 6):
        data_line = []
        for i in range(len(distortions)):
            distortion = distortions[distortions.columns[line]][i]
            angle = angles[angles.columns[line]][i]
            data_line.append(make_complex_value(angle=angle, vf_module=distortion))
        data.append(data_line)
    new_df = pd.DataFrame(data=np.array(data).T.tolist())
    with pd.ExcelWriter(file_name, mode='a') as writer:
        new_df.to_excel(writer, sheet_name="Complex Volatge", index=False)