# src/benchmark.py

import numpy as np 
import pandas as pd


# import matplotlib.pyplot as plt
# import seaborn as sns
from multiprocessing import Pool
from tqdm import tqdm

import os

class Banchmark:
    nome = ''
    def __init__(self, nome):
        self.nome = nome
        print('Running Banchmark constructor')

    def carrega_datasets(self):
        print('Loading dataset')

        diretorioDtasets = 'data'
        arquivos = os.listdir(diretorioDtasets)
        if '.gitignore' in arquivos: arquivos.remove('.gitignore')

        dataFrames = []
        for i in arquivos:
            dataFrames.append(pd.read_csv(diretorioDtasets + '/' + i))

        return dataFrames
    
    def preprocessa_datasets(self, dataframes):
        for i in range(0, len(dataframes), 1):
            dataframes[i] = dataframes[i].drop(['samples'], axis=1)

        dataframesNormalizados = []
    
        for dataframe in dataframes:
            dataframeNormalizado = self.normaliza_datasets(dataframe.drop(['type'], axis=1))
            dataframeNormalizado['type'] = dataframe['type']
            dataframesNormalizados.append(dataframeNormalizado)

        
        return dataframesNormalizados
    
    def normaliza_datasets(self, dataframe):

        for coluna in tqdm(dataframe.columns):
            dataframe[coluna] = (dataframe[coluna] - dataframe[coluna].mean())/np.std(dataframe[coluna])

        return dataframe





# def run_benchmark():

#     resultados_base = "?????????????"

#     print("Resultados do modelo base:", resultados_base)

#     return resultados_base

# def load_datasets():

#     resultados_base = "?????????????"

#     print("Resultados do modelo base:", resultados_base)

#     return resultados_base
