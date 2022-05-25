import math
import numpy as np
import pandas as pd


def make_complex_value(angle, if_module):
    return if_module * complex(real=math.cos(math.radians(angle)), imag=math.sin(math.radians(angle)))


files_name_list = [f'ByPhasisCurrentData/Phasis-{index}.xlsx' for index in range(2, 4)]

for file_name in files_name_list:
    file = pd.ExcelFile(file_name)
    distortions = pd.read_excel(file, file.sheet_names[0])
    angles = pd.read_excel(file, file.sheet_names[1])

    data = []
    for line in range(0, len(distortions.columns)):
        data_line = []
        for i in range(len(distortions)):
            distortion = distortions[distortions.columns[line]][i]
            angle = angles[angles.columns[line]][i]

            data_line.append(make_complex_value(angle=angle, if_module=distortion))
        data.append(data_line)
    new_df = pd.DataFrame(data=np.array(data).T.tolist())
    with pd.ExcelWriter(file_name, mode='a') as writer:
        new_df.to_excel(writer, sheet_name="Complex Currents", index=False)