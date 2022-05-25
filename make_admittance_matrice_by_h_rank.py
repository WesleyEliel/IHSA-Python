import random

import pandas as pd

file = pd.ExcelFile("AdmittanceData.xlsx")

admittance_mat = pd.read_excel(file, file.sheet_names[0])

h_rank_list = list(range(3, 13, 2))
for h_rank in h_rank_list:
    data = []
    for line in range(len(admittance_mat)):
        data_line = []
        for i in range(0, len(admittance_mat.columns)):

            element = admittance_mat[admittance_mat.columns[i]][line]
            real = complex(element).real
            imag = h_rank*complex(element).imag
            try:
                this_complex = 1 / (complex(real, imag))
            except ZeroDivisionError:
                this_complex = complex(0, 0)
            data_line.append(complex(round(this_complex.real, 2), round(this_complex.imag, 2)))
        data.append(data_line)


    new_df = pd.DataFrame(data=data)
    sheet_name = f'Harmonique de Rang {h_rank}'
    with pd.ExcelWriter('AdmittanceData.xlsx', mode='a') as writer:
        new_df.to_excel(writer, sheet_name=sheet_name, index=False)
