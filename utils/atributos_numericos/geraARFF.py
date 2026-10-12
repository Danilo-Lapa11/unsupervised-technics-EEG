import arff
import pandas as pd
import numpy as np


def base_arff(matriz_atr, filename):
    """
    Criar um arquivo arff a partir da matriz de atributos + classes
    :param matriz_atr: matriz de atributos
    :param filename: nome final do arquivo
    :return: não retorna nada
    """
    # criando as colunas do dataframe
    colunas = np.arange(matriz_atr.shape[1])
    colunas = [str(int(colunas[i])) for i in colunas]
    colunas[-1] = 'CLASS'  # modificando a última coluna para identificar a classe

    # criando o dataframe
    df_atributos = pd.DataFrame(matriz_atr, columns=colunas)

    t = df_atributos.columns[-1]  # coluna de classes do problema

    # montando o vetor com os @attribute do arff
    atributos = [(c, 'NUMERIC') for c in df_atributos.columns.values[:-1]]
    atributos += [('CLASS', df_atributos[t].unique().astype(str).tolist())]
    data = df_atributos.values

    # criando o dicionário arff
    arff_dic = {
        'attributes': atributos,
        'data': data,
        'relation': 'atributosExplicitosHCI_2classes',
        'description': ''
    }

    # criando o arquivo e salvando no diretório específico
    with open(filename, "w", encoding="utf8") as f:
        arff.dump(arff_dic, f)
