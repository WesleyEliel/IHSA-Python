import pandas as pd

file = pd.ExcelFile("../new.xlsx")

admittance_mat = pd.read_excel(file, file.sheet_names[0])

data = []
for line in range(len(admittance_mat)):
    data_line = []
    for i in range(0, len(admittance_mat.columns)):
        element = admittance_mat[admittance_mat.columns[i]][line]
        str_complex = str(element).split('+')
        real = round(float(str_complex[0].strip().replace(' ', '').replace(',', '.')), 8)
        imag = round(float(str_complex[1].strip().replace(' ', '').replace(',', '.').replace('i', '')), 8)
        data_line.append(complex(real, imag))
    data.append(data_line)
    print(f'Fin ligne {line} \n')

new_df = pd.DataFrame(data=data)
sheet_name = f'Admittance Complexe'
with pd.ExcelWriter('AdmittanceData.xlsx', mode='w') as writter:
    new_df.to_excel(writter, sheet_name=sheet_name, index=False)
