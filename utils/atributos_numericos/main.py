"""
Código para leitura de sinais fisiológicos, extração de atributos e criação
de um arquivo .arff

"""

# importando as bibliotecas
import extracao_atributos as ext
import geraARFF
import pandas as pd
import numpy as np
import os

"""
VARIÁVEIS:

* Y - sinal
Para Y(i,j), i deve ser o sinal no tempo e j os canais do EEG

* Xf - sinal após o janelamento
* Fs - frequencia de amostragem do sinal 
* janela - tamanho da janela em s
* superposição - superposição da janela em s
* n_canais - numero de canais do sinal 
* n_sinal - numero de sinais que precisam ser carregados
* Classe - classe do sinal que se encontra em carregamento 

"""

# Arquivos que serão lidos
file_names = ["H_Part_4_S_Trial4_emotion__.txt",
              "S_Part_27_S_Trial5_emotion_.txt"]  # adicionar o nome do arquivos na lista

""" parâmetros iniciais """
NumFiles = len(file_names)  # número de arquivos
Fs = 256  # frequência de amostragem
janela = 5  # tempo em segundos
superposicao = 1
n_canais = 38
classes = [0, 2]  # a ordem das classes devem corresponder à ordem dos arquivos da variável file_names
arquivo_final = os.path.join(os.getcwd(), "atributosExplicitosOctaveHCI_2classes.arff")

""" vetores inicializados """
atributos_total = np.array([])  # inicializando vetor com os atributos + classes

"""
Para este problema em específico 

0 - corresponde a hapiness
2 - corresponde a sadness

"""

for sinal in range(0, NumFiles):

    # carrega o sinal
    Y = pd.read_csv(file_names[sinal], delimiter=",", header=None).values.T  # lê o sinal em txt e armazena na matriz Y
    # caso o arquivo do sinal não esteja no formato .txt ou .csv, essa parte do código deve ser modificada

    # selecionando os canais de interesse
    # ATENÇÃO: lembrar que, em Python, a contagem começa do zero!!!!!!!!!!!
    Y = np.concatenate((Y[:, 0:32], Y[:, 32:35], Y[:, 40:41], Y[:, 44:46]), axis=1)

    # Extração de atributos
    atributos = ext.ExtrairAtributos(Y, Fs, janela, superposicao, n_canais)
    [l, c] = atributos.shape

    classe = classes[sinal]  # capturando a classe referente ao sinal
    vetor_classe = np.full((l, 1), classe)  # criando uma lista para armazenar a classe do sinal
    atributos = np.concatenate((atributos, vetor_classe), axis=1)

    if sinal == 0:
        atributos_total = atributos
    else:
        atributos_total = np.concatenate((atributos_total, atributos), axis=0)


# gerando o arquivo .arff para os experimentos
geraARFF.base_arff(atributos_total, arquivo_final)
