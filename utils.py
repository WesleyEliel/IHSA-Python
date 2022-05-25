# -*- coding: utf-8 -*-
"""
Created on August 26 19:28 2021

@author:
    Wesley Eliel MONTCHO
"""

import itertools
import math
from multiprocessing import Value, Array, Process

import numpy as np
import pandas as pd
import ray
from ray.util.multiprocessing import Pool

admittance_file = pd.ExcelFile("AdmittanceData.xlsx")
current_file = pd.ExcelFile("ByPhasisCurrentData/Phasis-1.xlsx")
voltage_file = pd.ExcelFile("ByPhasisVoltageData/Phasis-1.xlsx")

admittance_mat = pd.read_excel(admittance_file, admittance_file.sheet_names[0])
current_mat = pd.read_excel(current_file, current_file.sheet_names[2])
voltage_mat = pd.read_excel(voltage_file, voltage_file.sheet_names[2])

Iaf_set = []


def abs_to_power_2(the_complex):
    return (abs(the_complex)) ** 2


def Iaf_at_bus(bus, Iaf_for_h_at_bus_list):
    x = list(map(abs_to_power_2, Iaf_for_h_at_bus_list))
    return math.sqrt(sum(x))


def make_Iaf_for_each_h_at_bus(In, gamma, t_at_b_for_h):
    return complex(t_at_b_for_h * In.real * gamma, t_at_b_for_h * In.imag * gamma)


def calculate_Iaf(bus_number, gamma, t_list_for_this_bus):
    bus = bus_number
    Inl_list_for_this_bus = []
    Iaf_for_h_at_bus_list = []

    for i in range(1, len(current_mat.columns)):
        element = complex(current_mat[current_mat.columns[i]][bus])
        Inl_list_for_this_bus.append(element)
    # print("Inl_list_for_this_bus", Inl_list_for_this_bus)
    for t, In in zip(t_list_for_this_bus, Inl_list_for_this_bus):
        Iaf_for_h_at_bus_list.append(make_Iaf_for_each_h_at_bus(In=In, gamma=gamma, t_at_b_for_h=t))

    Iaf_at_bus_value = Iaf_at_bus(bus=bus, Iaf_for_h_at_bus_list=Iaf_for_h_at_bus_list)

    # print(f'Au Noeud {bus} on a Iaf {Iaf_at_bus_value}')
    return Iaf_at_bus_value, Iaf_for_h_at_bus_list, Inl_list_for_this_bus


def get_vh_list_by_bus():
    mat = voltage_mat.to_numpy(dtype=complex).tolist()
    return mat

def get_ih_list_by_bus():
    mat = current_mat.to_numpy(dtype=complex).tolist()
    return mat


def get_admittance_mat():
    mat = []
    for index in range(1, 6):
        mat.append(pd.read_excel(admittance_file, admittance_file.sheet_names[index]).to_numpy(dtype=complex).tolist())
    return mat


def get_if_list():
    to_return = []
    this_data = pd.read_excel(current_file, current_file.sheet_names[0])
    for line in range(len(this_data)):
        element = this_data[this_data.columns[0]][line]
        to_return.append(element)
    return to_return


def get_vf_list():
    to_return = []
    this_data = pd.read_excel(voltage_file, voltage_file.sheet_names[0])
    for line in range(len(this_data)):
        element = this_data[this_data.columns[0]][line]
        to_return.append(element)
    return to_return


def get_candidate_bus():
    to_return = []
    this_data = pd.read_excel(voltage_file, voltage_file.sheet_names[0])
    for line in range(len(this_data)):
        element = this_data[this_data.columns[7]][line]
        if float(element) >= 5:
            to_return.append(line)
    print(len(to_return))
    return to_return


@ray.remote
class LocationCombinationGeneration(object):
    def __init__(self, number_of_bus, candidate_bus_index_list, *args, **kwargs):
        self.number_of_bus = number_of_bus
        self.candidate_bus_index_list = candidate_bus_index_list
        self.candidate_bus_combination = []
        self.initial_final_combination = []
        self.final_combination = []

    def initialize_bus_combination(self):
        self.candidate_bus_combination = list(
            map(list, itertools.product([0, 1], repeat=len(self.candidate_bus_index_list))))

    def initialize_final_combination(self):
        self.initialize_bus_combination()
        self.initial_final_combination = list(
            itertools.repeat(list(itertools.repeat(0, self.number_of_bus)), len(self.candidate_bus_combination)))
        self.final_combination = [None] * len(self.initial_final_combination)

    def final_combination_generation_process(self, index, x, y):
        self.final_combination[index] = y[:]
        for x_element_index, x_element_value in enumerate(x):
            self.final_combination[index][self.candidate_bus_index_list[x_element_index]] = x_element_value
            """self.final_combination[index][1] = 1
            self.final_combination[index][41] = 1
            # final_combination[index][42] = 1
            # final_combination[index][43] = 1
            self.final_combination[index][46] = 1"""
            self.final_combination[index][45] = 1
            self.final_combination[index][65] = 1
            self.final_combination[index][60] = 1

    def generate_final_combination(self):
        for index, (x, y) in enumerate(zip(self.candidate_bus_combination, self.initial_final_combination)):
            self.final_combination_generation_process(index=index, x=x, y=y)

    def make_combination(self):
        self.generate_final_combination()

    def get_combination(self):
        final_combination = [item for item in self.final_combination if item.count(1) <= 8]
        return final_combination


"""
def generate_location_combination(number_of_bus, candidate_bus_index_list):
    init_candidate_bus_combination = list(map(list, itertools.product([0, 1], repeat=len(candidate_bus_index_list))))

    initial_final_combination = list(
        itertools.repeat(list(itertools.repeat(0, number_of_bus)), len(init_candidate_bus_combination)))

    final_combination = [None] * len(initial_final_combination)

    def doYourTask(index, x, y):
        print(index)
        print(x)
        print(y)
        for index, (x, y) in enumerate(zip(init_candidate_bus_combination, initial_final_combination)):
            final_combination[index] = y[:]
            for x_element_index, x_element_value in enumerate(x):
                final_combination[index][candidate_bus_index_list[x_element_index] - 1] = x_element_value
                final_combination[index][1] = 1
                final_combination[index][41] = 1
                # final_combination[index][42] = 1
                # final_combination[index][43] = 1
                # final_combination[index][44] = 1
                # final_combination[index][65] = 1
                final_combination[index][60] = 1

    for index, (x, y) in enumerate(zip(init_candidate_bus_combination, initial_final_combination)):
        p_index = Value('i', index)
        p_x = Array('i', x)
        p_y = Array('i', y)
        p = Process(target=doYourTask, args=(p_index, p_x, p_y))
        p.start()
        p.join()

    final_combination = [item for item in final_combination if item.count(1) <= 10]

    return final_combination

"""


def generate_location_combination(number_of_bus, candidate_bus_index_list):
    l = LocationCombinationGeneration.remote(number_of_bus, candidate_bus_index_list)
    l.initialize_final_combination.remote()
    l.make_combination.remote()
    result = ray.get(l.get_combination.remote())
    return result

def getHTLL():
    h_rank_list = list(range(3, 13, 2))
    vh_by_buses_list = get_vh_list_by_bus()
    print(len(vh_by_buses_list[0]))
    
    admittance_mat = get_admittance_mat()
    httl_for_h_index_list = []
    for h_rank_index, h_rank in enumerate(h_rank_list):
        httl_for_h_index = 0.0
        for bus_index in range(67):
            for bus_prime_index in range(67):
                Vh_at_the_bus_b = complex(vh_by_buses_list[bus_index][h_rank_index + 1])
                Vh_at_the_bus_b_prime = complex(vh_by_buses_list[bus_prime_index][h_rank_index + 1])
                Zh = complex(admittance_mat[h_rank_index][bus_index][bus_prime_index])
                Rh = Zh.real
                # print(f'V({h}, {b}); V({h}, {b_prime})  =  ({Vh_at_the_bus_b}; {Vh_at_the_bus_b_prime})')
                try:
                    htll_h_b_b_prime = ((Rh) / (Zh) ** 2) * abs(Vh_at_the_bus_b - Vh_at_the_bus_b_prime) ** 2
                except ZeroDivisionError:
                    htll_h_b_b_prime = 0.0
                httl_for_h_index += htll_h_b_b_prime
        httl_for_h_index_list.append(httl_for_h_index)

    return sum(httl_for_h_index_list)

# print(get_vh_list_by_bus())



def calculate_Iaf_Manually():
    result_1 = pd.ExcelFile("Results/Results1.xlsx")
    result_1_iaf_list = pd.read_excel(result_1, result_1.sheet_names[1])

    Iaf_for_h_at_buses = result_1_iaf_list.to_numpy(dtype=complex).tolist()
    
    Iaf_at_bus_values = []

    for index, element in enumerate(Iaf_for_h_at_buses):
        print(index, element)
        Iaf_at_bus_value = Iaf_at_bus(bus=index+1, Iaf_for_h_at_bus_list=element)
        Iaf_at_bus_values.append(Iaf_at_bus_value)

    with pd.ExcelWriter('Results/ResultsOfIafByBuses.xlsx', mode='a') as writer:
        pd.DataFrame(data=Iaf_at_bus_values).to_excel(writer, sheet_name="I at Buses for Phasis 3",
                                                                            index=False)


"""
if __name__ == '__main__':
    print('in the main function')

    with open('file.txt', 'a') as file:
                    file.writelines(f'{getHTLL()}, {abs(getHTLL())} \n\n')
                    file.close()

    calculate_Iaf_Manually()
"""