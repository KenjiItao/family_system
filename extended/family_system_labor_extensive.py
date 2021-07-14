import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pickle
import os
import sys
import math
import random
from scipy import stats
from scipy.optimize import curve_fit
import gc

if not os.path.exists("./res"):
    os.mkdir("./res")
    
plt.switch_backend('agg')

def linear_fit(x, a, b):
    return a * x + b

class Society:
    def __init__(self, families):
        self.families = families
        self.df = pd.DataFrame()

class Family:
    def __init__(self, wealth, order, separation, theta, threshold, boys, girls, husbands, wives, flag):
        self.wealth = wealth
        self.order = order
        self.separation = separation
        self.theta = theta
        self.threshold = threshold
        self.boys = boys
        self.girls = girls
        self.husbands = husbands
        self.wives = wives
        self.flag = flag

def distribution(wealth, order, population):
    dist = np.exp(- order * np.arange(population))
    dist = wealth * dist / np.sum(dist)

    return dist

def generation(families):
    grooms = []
    brides = []

    for family in families:
        family.wealth = (1 - decay) * family.wealth
        theta = family.theta
        if theta < 1.0 and theta > 0.0:
            ratio = family.boys * theta / (family.boys * theta + family.girls * (1-theta))
        elif theta == 1.0:
            ratio = 1.0
        else:
            ratio = 0.0
        if random.random() > family.separation:
            dist_women = distribution((1 - ratio) * family.wealth, family.order, family.girls)
            dist_men = distribution(ratio * family.wealth, family.order, family.boys)
            for wealth in dist_men:
                if wealth - epsilon > 0:
                    if wealth == dist_men[0]:
                        flag = 1
                    else:
                        flag = 0
                    grooms.append(Family(wealth - epsilon, family.order, family.separation, family.theta, family.threshold, 0, 0, 1, 0, flag))
            brides.extend(dist_women)
        else:
            dist_women = distribution((1 - ratio) * family.wealth, family.order, family.girls)
            family.husbands = family.boys
            family.wives = 0
            family.wealth = ratio * family.wealth
            if family.boys > 0:
                grooms.append(family)
            brides.extend(dist_women)

    families = []
    random.shuffle(grooms)

    j = 0
    while len(grooms) > 0:
        remove_ls = []
        for i in range(len(grooms)):
            groom = grooms[i]
            for k in range(groom.husbands):
                if j >= len(brides):
                    families.append(groom)
                    remove_ls.append(groom)
                    break
                bride = brides[j]
                if groom.wives == 0 and groom.wealth + bride - bridewealth > 0:
                    groom.wealth += bride - bridewealth
                    groom.wives = 1
                    j += 1
                elif (groom.wealth + bride - bridewealth) / (groom.husbands + groom.wives + 1) > max(groom.threshold, 0):
                    groom.wealth += bride - bridewealth
                    groom.wives += 1
                    j += 1
                else:
                    families.append(groom)
                    remove_ls.append(groom)
                    break

        grooms = list(set(grooms) - set(remove_ls))

    if len(families) > 0:
        rate = min(1, capacity / len(families))
    else:
        rate = 1

    next_gen = []

    for family in families:
        family.wealth +=  rate * max(0.1, random.gauss(1.0, noise)) * (1 + family.wealth * feedback) * (family.husbands + family.wives) / 4

        if family.boys > 0:
            dist_men = distribution(family.wealth, family.order, min(family.husbands, family.wives))
            for wealth in dist_men:
                if wealth - epsilon > 0:
                    wives = 1
                    if wealth == dist_men[0]:
                        flag = 1
                        if family.wives > family.husbands:
                            wives = 1 + family.wives - family.husbands
                    else:
                        flag = 0
                    wealth -= epsilon
                    boys = np.random.poisson(birth * wives + wealth * feedback)
                    girls = np.random.poisson(birth * wives + wealth * feedback)

                    if boys + girls > 0:
                        [order, separation, theta, threshold] = [family.order, family.separation, family.theta, family.threshold] + np.random.normal(0.0, mutation, 4)

                        if order <= 0.0:
                            order = 0.01
                        if separation < 0.0:
                            separation = 0.0
                        elif separation > 1.0:
                            separation = 1.0
                        if theta < 0.0:
                            theta = 0.0
                        elif theta > 1.0:
                            theta = 1.0
                        if threshold < 0.0:
                            threshold = 0.0

                        next_gen.append(Family(wealth, order, separation, theta, threshold, boys, girls, 1, wives, flag))
        else:
            family.boys = np.random.poisson(birth * family.wives + family.wealth * feedback)
            family.girls = np.random.poisson(birth * family.wives + family.wealth * feedback)

            if family.boys + family.girls > 0:
                [family.order, family.separation, family.theta, family.threshold] = [family.order, family.separation, family.theta, family.threshold] + np.random.normal(0.0, mutation, 4)

                if family.order <= 0.0:
                    family.order = 0.01
                if family.separation < 0.0:
                    family.separation = 0.0
                elif family.separation > 1.0:
                    family.separation = 1.0
                if family.theta < 0.0:
                    family.theta = 0.0
                elif family.theta > 1.0:
                    family.theta = 1.0
                if family.threshold < 0.0:
                    family.threshold = 0.0

                next_gen.append(family)

    families = next_gen[:]

    orders = np.array([family.order for family in families])
    hist, bins = np.histogram(orders, bins = 10)
    orders = [np.mean(orders), hist / np.sum(hist), bins]

    separations = np.array([family.separation for family in families])
    hist, bins = np.histogram(separations, bins = 10)
    separations = [np.mean(separations), hist / np.sum(hist), bins]

    thetas = np.array([family.theta for family in families])
    hist, bins = np.histogram(thetas, bins = 10)
    thetas = [np.mean(thetas), hist / np.sum(hist), bins]

    thresholds = np.array([family.threshold for family in families])
    hist, bins = np.histogram(thresholds, bins = 10)
    thresholds = [np.mean(thresholds), hist / np.sum(hist), bins]

    wives = np.array([family.wives for family in families])
    hist, bins = np.histogram(wives)
    wives = [np.mean(wives), hist / np.sum(hist), bins]

    population = 0
    for family in families:
        population += family.boys + family.girls

    return families, population, orders, separations, thetas, thresholds, wives, [family.wealth for family in families if family.flag == 1], [family.wealth for family in families if family.flag == 0]

def main():
    societies = []
    initial_pop = 2 * num_families
    for i in range(num_society):
        societies.append(Society([Family(1.0, math.log(2), 0.5, 0.5, 0, 1, 1, 0, 0, 0) for j in range(num_families)]))

    for iter in range(iteration):
        next_gen = []

        for society in societies:
            society.families, population, orders, separations, sigams, thresholds, wives, wealths1, wealths2 = generation(society.families)
            society.df[iter] = [orders, separations, sigams, thresholds, wives, wealths1, wealths2]

            if population > 2 * initial_pop:
                random.shuffle(society.families)
                n = math.floor(math.log2(population / initial_pop))
                k = math.floor(len(society.families)/2**n)
                for i in [0]*(2**n-1):
                    next_gen.append(Society(society.families[:k]))
                    society.families = society.families[k:]
                    next_gen[-1].df = society.df.copy()
                next_gen.append(society)
            elif population > 0.1 * initial_pop:
                next_gen.append(society)

        societies = [society for society in next_gen if len(society.families) > 0]

        if len(societies) > num_society:
            random.shuffle(societies)
            societies = societies[:num_society]

        if len(societies) == 0:
            break

    societies = [society for society in societies if len(society.families) > 1]

    df_res = pd.DataFrame(index = ["birth", "mutation", "epsilon",  "num_society", "num_family", "cap", "feedback", "decay", "noise", "bridewealth", "wealth", "order", "separation",  "theta", "threshold", "wife", "rich_exp", "poor_power"])
    if len(societies) > 0:
        for society in societies:
            orders, separations, thetas, thresholds, wives = [], [], [], [], []
            for i in range(iteration - 100, iteration):
                orders.append(society.df.iat[0,i][0])
                separations.append(society.df.iat[1,i][0])
                thetas.append(society.df.iat[2,i][0])
                thresholds.append(society.df.iat[3,i][0])
                wives.append(society.df.iat[4,i][0])

            wealths=[]
            for i in range(1000, iteration):
                wealths.extend(society.df.iat[5,i])
                wealths.extend(society.df.iat[6,i])
            try:
                wealths = np.sort(np.array(wealths))[::-1]

                wealths = np.log(wealths)
                (val, bins)  = np.histogram(wealths, bins=100)
                bins = (bins[1:] + bins[:-1]) / 2
                peak = bins[np.argmax(val)]
                peak_ratio = np.argmax(wealths < peak) / len(wealths)

                cur1 = wealths[round(len(wealths) * 0.01):int(round(len(wealths) * 0.1))]
                (val, bins)  = np.histogram(cur1, bins=30)
                bins = (bins[1:] + bins[:-1]) / 2
                param, cov = curve_fit(linear_fit, np.exp(bins), np.log(val))
                rich_exp_log = - 1 / param[0]

                cur1 = wealths[int(round(len(wealths) * peak_ratio)):round(len(wealths) * 0.9)]
                (val, bins) = np.histogram(cur1, bins=30)
                bins = (bins[1:] + bins[:-1]) / 2
                param, cov = curve_fit(linear_fit, bins, np.log(val))
                poor_power_log = param[0]

                res = [np.mean(np.exp(wealths)), np.mean(np.array(orders)), np.mean(np.array(separations)), np.mean(np.array(thetas)), np.mean(np.array(thresholds)), np.mean(np.array(wives)), rich_exp_log, poor_power_log]
                params = [birth, mutation, epsilon, num_society, num_families, cap, feedback, decay, noise, bridewealth]
                params.extend(res)
                df_res[len(df_res.columns)] = params
            except:
                pass

    return  df_res

k = 0
birth = 1.5
mutation = 0.03
epsilon = 0.1
epsilon = 0.00001
epsilon = 0.001
num_society = 50
num_families = 30
cap = 0.7
cap = 3
capacity = round(num_families * cap)
iteration = 2000
trial = 0
feedback = 0.3
decay = 0.7
noise = 0.1
bridewealth = 0.0001

for epsilon in [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]:
    for cap in [0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        df= pd.DataFrame()
        path = f"{num_society}com_{num_families}fam_b{round(birth * 10)}pd_mu{round(mutation * 1000)}pm_epsilon{round(epsilon * 100000)}pm_cap{round(cap * 100)}pc_f{round(feedback * 100)}pc_d{round(decay * 100)}pc_noise{round(noise * 100)}pc_bridewealth{round(bridewealth * 100000)}pm"

        capacity = round(num_families * cap)

        for trial in range(10):
            try:
                res = main()
                df = pd.concat([df, res.T])
                gc.collect()
            except:
                pass

        df.to_csv(f"res/res_{path}.csv")
