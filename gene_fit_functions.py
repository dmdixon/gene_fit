import numpy as np
import pandas as pd
import time as rtime
import textwrap
from itertools import chain
from scipy.interpolate import interp1d
from copy import deepcopy
from scipy.stats import chisquare
import uncertainties.unumpy as unumpy

def Init_Pop(N):
    chroms=[]
    for _ in range(N):
        chrom=''
        for key in gc.fit_params.keys():
            param,e_param=unumpy.nominal_values(gc.fit_params[key]),unumpy.std_devs(gc.fit_params[key])
            chrom+='{:.{prec}f}'.format(np.random.uniform(param-e_param,param+e_param),prec=gc.param_deci).replace('.','').zfill(gc.gen_len)
        chroms.append(chrom)
    return chroms

def Decode(chrom):
    params=[]
    params_split=textwrap.wrap(chrom,gc.gen_len)
    params=[float(param[:-gc.param_deci]+'.'+param[-gc.param_deci:]) for param in params_split]
    return params

def Recode(params):
    chrom=''
    for param in params:
        chrom+='{:.{prec}f}'.format(param,prec=gc.param_deci).replace('.','').zfill(gc.gen_len)
    return chrom


def Roulette_Selection(fitnesses):
    fitness_sum=np.nansum(fitnesses)
    fitness_scores= np.cumsum(fitnesses/fitness_sum)
    indices=[]
    count=0

    while count < round(len(fitness_scores)*gc.pop_prop):
        arrow = np.random.uniform()
        choices=np.where(fitness_scores >= arrow)[0]
        if len(choices)>0:
            indices.append(np.nanmin(choices))
            count+=1
        else:
            continue

    return indices

def Crossover(chroms):
    chrom_list=[]
    for n in range(int(len(chroms)/2)):
        chrom1=chroms[2*n]
        chrom2=chroms[2*n+1]

        cross_index=np.random.randint(0,len(chrom1)-1)

        cross1=chrom2[cross_index:]
        cross2=chrom1[cross_index:]

        chrom1=chrom1[:cross_index]+cross1
        chrom2=chrom2[:cross_index]+cross2

        chrom_list.append(chrom1)
        chrom_list.append(chrom2)

    return chrom_list

def Mutation(chrom):
    params=Decode(chrom)
    keys=list(gc.fit_params.keys())
    for n in range(len(params)):
        param,e_param=unumpy.nominal_values(gc.fit_params[keys[n]]),unumpy.std_devs(gc.fit_params[keys[n]])
        if (param>=param-e_param) & (param<=param+e_param):
            params[n]+=np.random.uniform(-e_param,e_param)
        else:
            params[n]=np.random.uniform(param-e_param,param+e_param)

    mutated_chrom=Recode(params)

    return mutated_chrom
