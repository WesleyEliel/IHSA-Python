import pandas as pd



file_1 = pd.ExcelFile("AdmittanceData.xlsx")
file_2 = pd.ExcelFile("ByPhasisVoltageData/Phasis-1.xlsx")


admittance_mat = pd.read_excel(file_1, file_1.sheet_names[0])

volatge_mat = pd.read_excel(file_2, file_2.sheet_names[2])



h_rank_list = list(range(3, 13, 2))

def htll_h_between_b_b_prime(Rh, Zh, h, b, b_prime):
    # print(f'R / {b}-{b_prime}')
    h_index = h_rank_list.index(h) + 1
    Vh_at_the_bus_b = complex(volatge_mat[volatge_mat.columns[h_index]][b])
    Vh_at_the_bus_b_prime = complex(volatge_mat[volatge_mat.columns[h_index]][b_prime])

    # print(f'V({h}, {b}); V({h}, {b_prime})  =  ({Vh_at_the_bus_b}; {Vh_at_the_bus_b_prime})')
    htll_h_b_b_prime = ((Rh) / (Zh) ** 2) * abs(Vh_at_the_bus_b - Vh_at_the_bus_b_prime) ** 2

    return htll_h_b_b_prime


htll_h_list = []
print(len(admittance_mat))
print(len(admittance_mat.columns))
for h_rank in h_rank_list:
    data = [] 
    for line in range(len(admittance_mat)):
        data_line = []
        for i in range(0, len(admittance_mat.columns)):
            element = complex(admittance_mat[admittance_mat.columns[i]][line])
            real = element.real
            imag = element.imag
            Zh = complex(real, h_rank*imag)
            Rh = real * ( 1 + (0.646*(h_rank**2))/(192 + 0.51*(h_rank**2)))
            print(Zh, Rh)
            if complex(real, imag) != 0:
                htll_h_b_b_prime = htll_h_between_b_b_prime(Rh=Rh, Zh=Zh, h=h_rank, b=line, b_prime=i)
                data_line.append(htll_h_b_b_prime)
            else:
                htll_h_b_b_prime = complex(0, 0)
        data.append(sum(data_line))
    htll_h_list.append(sum(data))
print(sum(htll_h_list))