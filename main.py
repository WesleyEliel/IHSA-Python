# -*- coding: utf-8 -*-
"""
Created on August 26 17:50 2021

@author:
    Wesley Eliel MONTCHO
"""
import math
import numpy as np
import pandas as pd
import random
from random import uniform

from utils import generate_location_combination, calculate_Iaf, get_if_list, get_vf_list, get_vh_list_by_bus, \
    get_admittance_mat, voltage_mat, current_mat

admittance_file = pd.ExcelFile("AdmittanceData.xlsx")


def make_thdi_at_b(If, new_ih_at_bus_list):
    value = list(map(lambda x: (abs(x)) ** 2, new_ih_at_bus_list))
    if abs(If) != 0:
        return math.sqrt(sum(value)) / abs(If)
    else:
        return 0.0


def make_thdv_at_b(Vf, new_vh_at_bus_list):
    value = list(map(lambda x: (abs(x)) ** 2, new_vh_at_bus_list))
    if abs(Vf) != 0:
        return math.sqrt(sum(value)) / abs(Vf)
    else:
        return 0.0


def make_global_thd(thd_at_buses):
    return sum(thd_at_buses[1:]) / len(thd_at_buses)


h_rank_list = list(range(3, 13, 2))


class IHSAlgorithm:
    def __init__(self):
        self._HM = []  # Harmony Memory
        self._HMForCurrent = []  # Harmony For Current
        self._HMForVoltage = []  # Harmony For Voltage
        self._HMS = 10  # Harmony Memory Size
        self._HMCRmax = 0.9  # Harmony Memory Considering Rate
        self._HMCRmin = 0.1
        self._PARmax = 2  # Pitch Adjusting Rate
        self._PARmin = 0.2
        self._BWmax = 0.9  # Band Width
        self._BWmin = 0.4
        self._NumOfIterations = 12  # Max Iteration Times
        self._variables_number = 67 * 5
        self._variables = []
        self._varUpperBounds = []
        self._varLowerBounds = []
        self._IafForHAtBusesLists = []
        self._VAfterInjectionAtBusesLists = []
        self._IAfterInjectionAtBusesLists = []
        self._HTLL_List = []
        self._FilterCostList = []
        self._locationChoiceMemory = []
        self._forcedLocationToFilter = []
        self._filterLocationCombination = []
        self._f = np.empty(self._HMS)
        self._generation = 0
        self._objective_function = self.compute_objective_functions
        self.compute = lambda X: self._objective_function(X)
        self._trace = []
        self._lastBestSolutionIteration = 0

    def compute_objective_functions(self, new_vector):
        if isinstance(new_vector, dict):
            decision_variable_values = list(new_vector.values())
        else:
            decision_variable_values = new_vector

        random_location_combination_choice = random.choice(self._filterLocationCombination)

        data = []

        data_completed = True
        for element in list(np.array_split(np.array(decision_variable_values), 67)):
            data.append(element)

        vh_list_by_bus = get_vh_list_by_bus()
        admittance_mat = get_admittance_mat()
        If_list = get_if_list()
        Vf_list = get_vf_list()

        data_current_after_injection = []
        data_voltage_after_injection = []

        Iaf_for_h_at_buses_lists = []
        new_potentials_at_buses = []
        Iaf_list = []
        thdi_at_buses = []
        thdv_at_buses = []

        def setIafForHAtBusesLists():
            nonlocal data
            for index, value in enumerate(data):
                Iaf, Iaf_for_h_at_bus_list, Inl_list_for_this_bus = calculate_Iaf(bus_number=index,
                                                                                  gamma=
                                                                                  random_location_combination_choice[
                                                                                      index],
                                                                                  t_list_for_this_bus=value)

                """print("Iaf_for_h_at_bus_list Before", Iaf_for_h_at_bus_list)
                 for iaf_index, iaf_value in enumerate(Iaf_for_h_at_bus_list):
                     nb = round(100 * abs(iaf_value)) / 100 * abs(iaf_value)
                     new_iaf_value = iaf_value * nb
                     Iaf_for_h_at_bus_list[iaf_index] = new_iaf_value
                 print("Iaf_for_h_at_bus_list After", Iaf_for_h_at_bus_list)
                 """



                Iaf_list.append(Iaf)
                Iaf_for_h_at_buses_lists.append(Iaf_for_h_at_bus_list)
                h_rank_satisfied_condition = False
                Ih_list_at_bus_after_current_injection = []

                """if Iaf < 70:
                    h_rank_list_min_value = [2.3, 1.14, 0.77, 0.40, 0.33]
                    h_rank_satisfied_condition = True
                    for old, new in zip(Inl_list_for_this_bus, Iaf_for_h_at_bus_list):
                        Ih_list_at_bus_after_current_injection.append(old - new)
                    # print(Ih_list_at_bus_after_current_injection)
                    for h_rank_index, h_rank_min_value in enumerate(h_rank_list_min_value):
                        # h_rank_satisfied_condition *= h_rank_min_value > abs(Ih_list_at_bus_after_current_injection[h_rank_index])
                        h_rank_satisfied_condition *= True"""

                """if h_rank_satisfied_condition:
                    Iaf_for_h_at_buses_list.append(Iaf_for_h_at_bus_list)
                    data_completed *= True
                else:
                    data_completed *= False"""

        def setDataVoltageAfterInjection():
            nonlocal vh_list_by_bus
            nonlocal admittance_mat
            for bus_index, bus in enumerate(vh_list_by_bus):
                new_voltage_at_bus = []
                for h_index, element in enumerate(bus):
                    if h_index != 0:
                        old_voltage_for_h_at_this_bus = element
                        delta_voltage = 0.0

                        for b_prime_index in range(len(vh_list_by_bus)):
                            try:
                                delta_voltage += (1 / complex(
                                    admittance_mat[h_index - 1][bus_index][b_prime_index])) * complex(
                                    Iaf_for_h_at_buses_lists[b_prime_index][h_index - 1])
                            except ZeroDivisionError:
                                delta_voltage += complex(0, 0) * complex(
                                    Iaf_for_h_at_buses_lists[b_prime_index][h_index - 1])
                        new_voltage_for_h_at_this_bus = old_voltage_for_h_at_this_bus + delta_voltage

                        new_voltage_at_bus.append(new_voltage_for_h_at_this_bus)
                data_voltage_after_injection.append(new_voltage_at_bus)

        def AsetDataCurrentAfterInjection():
            nonlocal data_voltage_after_injection

            for bus_index in range(67):
                current_data_at_bus = []
                for h_index in range(5):
                    new_ih_at_bus_for_h = 0.0
                    for bus_prime_index in range(67):
                        new_ih_at_bus_for_h = (data_voltage_after_injection[bus_index][h_index] -
                                               data_voltage_after_injection[bus_prime_index][h_index]) * complex(
                            admittance_mat[h_index][bus_index][bus_prime_index])
                    current_data_at_bus.append(new_ih_at_bus_for_h)
                data_current_after_injection.append(current_data_at_bus)

        def setDataCurrentAfterInjection():
            nonlocal data_voltage_after_injection

            v_mat = voltage_mat.to_numpy(dtype=complex).tolist()
            i_mat = current_mat.to_numpy(dtype=complex).tolist()

            p_q_file = pd.ExcelFile("P-Q.xlsx")
            p_q_mat = pd.read_excel(p_q_file, p_q_file.sheet_names[0]).to_numpy(dtype=complex).tolist()
            for bus_index, ([p, q], v_by_h_rank_at_b) in enumerate(zip(p_q_mat, data_voltage_after_injection)):
                current_data_at_bus = []
                for h_rank, v_h in enumerate(v_by_h_rank_at_b):
                    x1 = abs(v_mat[bus_index][0]) ** 2
                    x2 = ((h_rank + 1) * abs(v_mat[bus_index][0]) ** 2)
                    if x1 != complex(0, 0):
                        _x1 = 1 / x1
                    else:
                        _x1 = 0

                    if x2 != complex(0, 0):
                        _x2 = 1 / x2
                    else:
                        _x2 = 0

                    y_h_b = complex(p * _x1, -q * _x2)
                    i_h_b = y_h_b * v_h
                    """if abs(Iaf_for_h_at_buses_lists[bus_index][h_rank]) != 0:
                        print("\n\n\n\n i_h_b", bus_index, abs(i_h_b), i_h_b,
                              abs(Iaf_for_h_at_buses_lists[bus_index][h_rank]), Iaf_for_h_at_buses_lists[bus_index][h_rank],
                              abs(i_mat[bus_index][h_rank + 1]), i_mat[bus_index][h_rank + 1])"""
                    current_data_at_bus.append(i_h_b)
                data_current_after_injection.append(current_data_at_bus)

        def getThdi():
            nonlocal thdi_at_buses
            for If, new_ih_at_bus_list in zip(If_list, data_current_after_injection):
                # print(If, new_ih_at_bus_list)
                thdi_at_buses.append(make_thdi_at_b(If, new_ih_at_bus_list))

        def getThdv():
            nonlocal thdv_at_buses
            for Vf, new_vh_at_bus_list in zip(Vf_list, data_voltage_after_injection):
                thdv_at_buses.append(make_thdv_at_b(Vf, new_vh_at_bus_list))
            thdv_at_buses = [0 if x != x else x for x in thdv_at_buses]

        def getHTLL():
            nonlocal data_voltage_after_injection
            nonlocal admittance_mat
            httl_for_h_index_list = []
            for h_rank_index, h_rank in enumerate(h_rank_list):
                httl_for_h_index = 0.0
                for bus_index in range(67):
                    for bus_prime_index in range(67):
                        Vh_at_the_bus_b = complex(data_voltage_after_injection[bus_index][h_rank_index])
                        Vh_at_the_bus_b_prime = complex(data_voltage_after_injection[bus_prime_index][h_rank_index])
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

        def getCost():
            nonlocal random_location_combination_choice
            nonlocal Vf_list
            nonlocal data_voltage_after_injection
            nonlocal data_current_after_injection
            nonlocal Iaf_for_h_at_buses_lists
            paf_cost = 0.0
            for index in range(len(random_location_combination_choice)):
                v1_b = Vf_list[index]
                sum_square_v_h_at_b = 0.0
                sum_square_i_af_h_at_b = 0.0

                for h_index in range(5):
                    sum_square_v_h_at_b += abs(data_voltage_after_injection[index][h_index]) ** 2
                    sum_square_i_af_h_at_b += abs(Iaf_for_h_at_buses_lists[index][h_index]) ** 2
                vaf_b = math.sqrt((v1_b ** 2 + sum_square_v_h_at_b) * sum_square_i_af_h_at_b)
                paf_cost += random_location_combination_choice[index] * (0.09 + (6.3 * vaf_b) / 10000000)
            return paf_cost

        setIafForHAtBusesLists()
        setDataVoltageAfterInjection()
        setDataCurrentAfterInjection()
        getThdi()
        getThdv()

        htll = getHTLL()
        cost = getCost()

        # print(htll, cost)

        verify_condition_for_current = all(var < 0.1 for var in thdi_at_buses)
        verify_condition_for_voltage = all(var < 0.05 for var in thdv_at_buses)
        verify_all_conditions = verify_condition_for_voltage and verify_condition_for_current
        # print(verify_condition_for_current, verify_condition_for_voltage)

        if verify_all_conditions:
            print(make_global_thd(thdi_at_buses))
            # print(thdi_at_buses)

            """with open('file.txt', 'w') as file:
                                                    for index, element in enumerate(thdi_at_buses):
                                                        if element > 0.1:
                                                            if not index in self._forcedLocationToFilter:
                                                                self._forcedLocationToFilter.append(index)
                                                            file.writelines(f'{index}\n')
                                                    file.close()"""

        # print(100 * make_global_thd(thdv_at_buses), 100 * make_global_thd(thdi_at_buses))

        return [True, thdv_at_buses, thdi_at_buses,
                [data_voltage_after_injection, data_current_after_injection, Iaf_for_h_at_buses_lists, htll, cost],
                random_location_combination_choice]

    def setBounds(self, index, lower, upper):
        if len(self._varLowerBounds) <= index:
            self._varLowerBounds.append(lower)
            self._varUpperBounds.append(upper)
        else:
            self._varLowerBounds[index] = lower
            self._varUpperBounds[index] = upper

    def _setDefaultBounds(self):
        for i in range(self._variables_number):
            self.setBounds(i, 0, 1)

    def setVariables(self):
        for i in range(self._variables_number):
            self._variables.append(f't{i + 1}')

    def setFilterLocationsCombination(self):
        self._filterLocationCombination = generate_location_combination(number_of_bus=67,
                                                                        candidate_bus_index_list=[2, 42, 46, 64])

    def getVariables(self):
        return self._variables

    def setInitialsValues(self):
        self.setVariables()
        self._setDefaultBounds()
        self.setFilterLocationsCombination()

    def initializeHM(self):
        def catchZeroDivision(i):
            inputVector = {}
            for counter, var in enumerate(self._variables):
                inputVector.update({var: uniform(self._varLowerBounds[counter], self._varUpperBounds[counter])})
            self._HM.append(inputVector)
            try:
                computed = self.compute(list(inputVector.values()))
                self._f[i] = make_global_thd(computed[2])
                self._VAfterInjectionAtBusesLists.append(computed[-2][0])
                self._IAfterInjectionAtBusesLists.append(computed[-2][1])
                self._IafForHAtBusesLists.append(computed[-2][2])
                self._HTLL_List.append(computed[-2][3])
                self._FilterCostList.append(computed[-2][4])
                self._locationChoiceMemory.append(computed[-1])
                self._HMForCurrent.append(computed[2])
                self._HMForVoltage.append(computed[1])
            except ZeroDivisionError or RuntimeWarning:
                print("Division by Zero error occur")
                raise

        self._f = np.empty(self._HMS)
        for i in range(self._HMS):
            print(f'Size line {i}')
            catchZeroDivision(i)
        """with open('file.txt', 'w') as file:
            file.writelines(f'{self._IafForHAtBusesLists}\n\n\n\n\n\n')
            file.close()"""
    def improvise(self):
        new = {}
        verify_constraints = False
        while not verify_constraints:
            for i, variables in enumerate(self._variables):
                upperBound = self._varUpperBounds[i]
                lowerBound = self._varLowerBounds[i]
                # memoryConsideration
                if uniform(0, 1) < self._HMCR:
                    D1 = int(uniform(0, self._HMS))
                    D2 = self._HM[D1].get(variables)
                    new.update({variables: D2})

                    # pitchAdjustment
                    if uniform(0, 1) < self._PAR:
                        if uniform(0, 1) < 0.5:
                            D3 = (new.get(variables) -
                                  uniform(0, self._BW)
                                  )
                            if lowerBound <= D3:
                                new.update({variables: D3})
                        else:
                            D3 = (new.get(variables) +
                                  uniform(0, self._BW)
                                  )
                            if upperBound >= D3:
                                new.update({variables: D3})

                else:
                    new.update({variables: uniform(lowerBound,
                                                   upperBound)})
            computed = self.compute_objective_functions(new_vector=list(new.values()))
            verify_constraints = computed[0]
        return new

    def updateHM(self, new):
        computed = self.compute(new)
        f = make_global_thd(computed[2])
        # for finding minimum
        fMaxValue = np.amax(self._f)
        if f < fMaxValue:
            print("\n\n\n\n Optimisation data \n\n\n\n\n")
            print(f' fMaxValue : {fMaxValue} & f : {f} \n')

            for i, value in enumerate(self._f):
                if fMaxValue == value:
                    print(f'Index of I: {i}')
                    self._f[i] = f
                    self._HM[i] = new
                    self._locationChoiceMemory[i] = computed[-1]
                    self._IafForHAtBusesLists[i] = computed[-2][2]
                    self._VAfterInjectionAtBusesLists[i] = computed[-2][0]
                    self._IAfterInjectionAtBusesLists[i] = computed[-2][1]
                    self._HTLL_List[i] = computed[-2][3]
                    self._FilterCostList[i] = computed[-2][4]
                    self._HMForCurrent[i] = computed[2]
                    self._HMForVoltage[i] = computed[1]
                    break
            print(computed[-2][2])
            print("\n\n\n\n\n\n\n")

    def _findTrace(self):
        index = np.argmin(self._f)
        variables = self._HM[index]
        if variables not in self._trace:
            self._trace.append(variables)
            self._lastBestSolutionIteration = self._generation

    def doYourTask(self):
        def catchZeroDivision():
            try:
                new = self.improvise()  # (self._generation - 1) % self._HMS
                self.updateHM(new)
            except ZeroDivisionError or RuntimeWarning:
                print('i caughed ZeroDiv in IHS.updateHM')
                catchZeroDivision()

        self.initializeHM()
        while self._generation < self._NumOfIterations:
            self._generation += 1
            print(self._generation)
            self._updateHMCR()
            self._updatePAR()
            self._updateBW()
            catchZeroDivision()
            self._findTrace()

    def _updateHMCR(self):
        self._HMCR = (self._HMCRmax - self._generation *
                      (self._HMCRmax - self._HMCRmin) / self._NumOfIterations)

    def _updatePAR(self):
        self._PAR = (self._PARmin + self._generation *
                     (self._PARmax - self._PARmin) / len(self._variables))

    def _updateBW(self):
        c = math.log(self._BWmin / self._BWmax)
        self._BW = self._BWmax * math.exp(self._generation * c)

    def getOptimalSolution(self):
        index = np.argmin(self._f)
        functionValue = self._f[index]
        variables = self._HM[index]
        preparedVariables = []
        for key, value in variables.items():
            try:
                preparedVariables.append(f'{key}:\t{value}')
            except TypeError as e:
                print(e)
                return
        # print(self._f)
        # print(self._f[index])
        # print(functionValue, preparedVariables)
        return index, functionValue, preparedVariables

    def getTrace(self):
        return self._trace

    def getLastBestSolutionIteration(self):
        return self._lastBestSolutionIteration


if __name__ == "__main__":
    ihs = IHSAlgorithm()
    ihs.setInitialsValues()
    ihs.doYourTask()
    index, best_value, variables_value = ihs.getOptimalSolution()


    print("\n\n\n\n\n\n\n\n\n\n\n\n")
    print("================RESULTATS================ \n\n")
    print(index, best_value)
    print(ihs._IafForHAtBusesLists[index])
    print("\n\n\n\n")
    for index, element in enumerate(ihs._IafForHAtBusesLists):
        print("\n\n")
        print(f'Index : {index} \n')
        print(element)
    print("\n\n\n\n\n\n")


    """with open('Results/Best3.txt', 'a') as file:
                    file.writelines("\n\n\n\n ########## Last perform ########## \n\n")
                    file.writelines(f'Index of the best result  : \t {index} \n')
                    file.writelines(f'Total Thdv of the best result : \t {best_value}\n')
                    file.writelines("\n ############################################## \n\n")
                    file.close()
            
                print(ihs._HTLL_List)
                print([abs(i) for i in ihs._HTLL_List])
                print(abs(ihs._HTLL_List[index]))
                print(abs(ihs._FilterCostList[index]))
                with pd.ExcelWriter('Results/Results1.xlsx', mode='a') as writer:
                    pd.DataFrame(data=ihs._IafForHAtBusesLists[index]).to_excel(writer, sheet_name="Iaf at Buses", index=False)
                    pd.DataFrame(data=ihs._IAfterInjectionAtBusesLists[index]).to_excel(writer, sheet_name="I at Buses",
                                                                                        index=False)
                    pd.DataFrame(data=ihs._VAfterInjectionAtBusesLists[index]).to_excel(writer, sheet_name="V at Buses",
                                                                                        index=False)
                    pd.DataFrame(data=[ihs._HTLL_List[index], abs(ihs._HTLL_List[index])]).to_excel(writer, sheet_name="HTLL",
                                                                                                    index=False)
                    pd.DataFrame(data=[ihs._FilterCostList[index]]).to_excel(writer, sheet_name="Active filter Cost",
                                                                             index=False)
            
                # print(ihs._HMForCurrent)
                # print(ihs._HMForVoltage)
                # print(ihs._HMForVoltage[index])
            
                current_result = []
                voltage_result = []
                location_result = []
            
                for currents, voltages, locations in zip(ihs._HMForCurrent, ihs._HMForVoltage, ihs._locationChoiceMemory):
                    current_result.append([make_global_thd(currents[:]), 0, 0, 0])
                    current_result[-1].extend(currents[:])
                    voltage_result.append([make_global_thd(voltages[:]), 0, 0, 0])
                    voltage_result[-1].extend(voltages[:])
                    location_result.append(locations[:])
                    location_result[-1].extend(["", "", "", sum(locations[:])])
            
                data = {'current_result': current_result, 'voltage_result': voltage_result, 'location_result': location_result}
                for key in data.keys():
                    new_df = pd.DataFrame(data=data[key])
                    sheet_name = key.upper()
                    with pd.ExcelWriter('Results/Results1.xlsx', mode='a') as writer:
                        new_df.to_excel(writer, sheet_name=sheet_name, index=False)
            """