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

plt.switch_backend('agg')

def linear_fit(x, a, b):
    return a * x + b

class Community:
    def __init__(self, families):
        self.families = families
        self.df = pd.DataFrame()

class Family:
    def __init__(self, wealth, order, separation, children, adults, rich_flag, flag):
        self.wealth = wealth
        self.order = order
        self.separation = separation
        self.children = children
        self.adults = adults
        self.rich_flag = rich_flag
        self.flag = flag

def distribution(wealth, order, population):
    dist = np.exp(- order * np.arange(population))
    dist = wealth * dist / np.sum(dist) - epsilon

    return dist

def generation(families):
    next_gen = []

    for family in families:
        family.wealth = (1 - decay) * family.wealth
        if random.random() > family.separation:
            dist = distribution(family.wealth, family.order, family.children)
            for wealth in dist:
                if wealth > 0:
                    if wealth == dist[0]:
                        flag = 1
                    else:
                        flag = 0
                    next_gen.append(Family(wealth, family.order, family.separation, 0, 1, family.rich_flag, flag))

        else:
            family.adults = family.children
            next_gen.append(family)

    families = next_gen[:]

    if len(families) > 0:
        rate = min(1, capacity / len(families))
    else:
        rate = 1

    next_gen = []

    for family in families:
        family.wealth +=  rate * (1 + max(-0.9, random.gauss(0.0, noise))) * (1 + family.wealth) * math.log(family.adults + 1)

        if family.children > 0:
            dist = distribution(family.wealth, family.order, family.adults)
            for wealth in dist:
                if wealth > 0:
                    if wealth == dist[0]:
                        flag = 1
                    else:
                        flag = 0
                    children = np.random.poisson(birth + wealth * feedback)
                    if children > 0:
                        order = family.order + random.gauss(0.0, mutation)
                        if order <= 0.0:
                            order = 0.01

                        separation = family.separation + random.gauss(0.0, mutation)

                        if separation < 0.0:
                            separation = 0.0
                        elif separation > 1.0:
                            separation = 1.0

                        next_gen.append(Family(wealth, order, separation, children, 0, family.rich_flag, flag))
        else:
            family.children = np.random.poisson(birth + family.wealth * feedback)

            if family.children > 0:
                family.order += random.gauss(0.0, mutation)
                if family.order <= 0.0:
                    family.order = 0.01

                family.separation += random.gauss(0.0, mutation)
                if family.separation < 0.0:
                    family.separation = 0.0
                elif family.separation > 1.0:
                    family.separation = 1.0

                next_gen.append(family)

    families = next_gen[:]

    if len(families) > 0:
        cur = np.array([family.wealth for family in families])
        criterion = np.sort(cur)[max(0, round(len(cur) * 0.9) - 1)]
    else:
        criterion = 0

    population = 0
    for family in families:
        population += family.children
        if family.wealth >= criterion:
            family.rich_flag += 1
        else:
            family.rich_flag = 0

    orders = np.array([family.order for family in families])
    hist, bins = np.histogram(orders, bins = 10)
    orders = [np.mean(orders), hist / np.sum(hist), bins]

    separations = np.array([family.separation for family in families])
    hist, bins = np.histogram(separations, bins = 10)
    separations = [np.mean(separations), hist / np.sum(hist), bins]

    return families, population, orders, separations, [family.wealth for family in families if family.flag == 1], [family.wealth for family in families if family.flag == 0], np.mean(np.array([family.rich_flag for family in families if family.flag == 1]))


def main():
    communities = []
    for i in range(num_community):
        communities.append(Community([Family(1.0, math.log(2), 0.5, 1, 0, 0, 0) for j in range(num_families)]))

    for iter in range(iteration):
        next_gen = []

        for community in communities:
            community.families, population, orders, separations, wealths1, wealths2, rich_flags = generation(community.families)
            community.df[iter] = [orders, separations, wealths1, wealths2, rich_flags]

            if population > 2 * num_families:
                random.shuffle(community.families)
                n = math.floor(math.log2(population / num_families))
                k = math.floor(len(community.families)/2**n)
                for i in [0]*(2**n-1):
                    next_gen.append(Community(community.families[:k]))
                    community.families = community.families[k:]
                    next_gen[-1].df = community.df.copy()
                next_gen.append(community)
            elif population > 0.1 * num_families:
                next_gen.append(community)

        communities = [community for community in next_gen if len(community.families) > 0]

        if len(communities) > num_community:
            random.shuffle(communities)
            communities = communities[:num_community]

        if len(communities) == 0:
            break

    communities = [community for community in communities if len(community.families) > 1]

    if len(communities) > 0:
        df_res = pd.DataFrame(index = ["birth", "mutation", "epsilon",  "num_community", "num_family", "cap", "feedback", "decay", "noise", "wealth", "order", "separation",  "rich_flag", "half_index", "rich_exp", "poor_power","gibrat"])
        for community in communities:
            cur = np.array([[family.wealth, family.order, family.separation, family.rich_flag] for family in community.families])
            wealths, orders, separations, rich_flags = [], [], [], []
            for i in range(iteration - 100, iteration):
                wealths.extend(community.df.iat[2,i])
                orders.append(community.df.iat[0,i][0])
                separations.append(community.df.iat[1,i][0])
                rich_flags.append(community.df.iat[4,i])

            wealths=[]
            for i in range(1000, iteration):
                wealths.extend(community.df.iat[2,i])
            for i in range(1000, iteration):
                wealths.extend(community.df.iat[3,i])
            try:
                rich_flag =  np.mean(np.array(rich_flags)) / 0.1

                wealths = np.sort(np.array(wealths))[::-1]
                score = wealths / np.sum(wealths)
                half_index = np.argmax(score.cumsum() > 0.5) / len(score)
                gibrat = 1.0 / math.sqrt(2 * np.var(np.log(wealths)))

                one_ratio = np.argmax(wealths < 1.2) / len(wealths)

                wealths = np.log(wealths)
                (val, bins)  = np.histogram(wealths, bins=100)
                bins = (bins[1:] + bins[:-1]) / 2
                peak = bins[np.argmax(val)]
                peak_ratio = np.argmax(wealths < peak) / len(wealths)

                cur1 = wealths[round(len(wealths) * 0.01):int(round(len(wealths) * 0.1))]
                (val, bins)  = np.histogram(cur1, bins=30)
                bins = (bins[1:] + bins[:-1]) / 2
                param, cov = curve_fit(linear_fit, np.exp(bins), np.log(val))
                rich_exp = - 1 / param[0]
                predict = np.exp(np.exp(bins) * param[0])
                mse_exp = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)


                cur1 = wealths[int(round(len(wealths) * peak_ratio)):round(len(wealths) * 0.9)]
                (val, bins) = np.histogram(cur1, bins=30)
                bins = (bins[1:] + bins[:-1]) / 2
                param, cov = curve_fit(linear_fit, bins, np.log(val))
                poor_power = param[0]
                predict = np.exp(bins) ** param[0]
                mse_power = np.mean((val / np.sum(val) - predict / np.sum(predict)) ** 2)

                res = [np.mean(np.array(wealths)), np.mean(np.array(orders)), np.mean(np.array(separations)), rich_flag, half_index, rich_exp, poor_power, gibrat]
                params = [birth, mutation, epsilon, num_community, num_families, cap, feedback, decay, noise]
                params.extend(res)
                df_res[len(df_res.columns)] = params
            except:
                pass

    return  df_res


k = 0
birth = 1.5
mutation = 0.03
epsilon = 0.01
epsilon = 0.0001
epsilon = 0.001
num_community = 50
num_families = 30
cap = 0.7
cap = 3
capacity = round(num_families * cap)
iteration = 2000
trial = 0
feedback = 0.3
decay = 0.5
noise = 0.1

for epsilon in [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]:
    for cap in [0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
        df_res = pd.DataFrame()
        path = f"{num_community}com_{num_families}fam_b{round(birth * 10)}pd_mu{round(mutation * 1000)}pm_epsilon{round(epsilon * 100000)}pm_cap{round(cap * 100)}pc_f{round(feedback * 100)}pc_d{round(decay * 100)}pc_noise{round(noise * 100)}pc"

        capacity = round(num_families * cap)

        for trial in range(20):
            try:
                res = main()
                df_res = pd.concat([df_res, res.T])
            except:
                pass

        df_res.to_csv(f"res_{path}.csv")
