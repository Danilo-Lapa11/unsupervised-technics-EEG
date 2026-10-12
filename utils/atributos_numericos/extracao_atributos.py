# importando as bibliotecas para excutar a função
import numpy as np
import CalcAtributos
import math


def ExtrairAtributos(Y, Fs, janela, superposicao, n_canais):
    """
    :param Y: sinal
    :param Fs: frequencia de amostragem do sinal
    :param janela: tamanho da janela em s
    :param superposicao: superposição da janela
    :param n_canais: numero de canais do sinal (retirar o número de eletrodos de referência)
    :return: vetor de atributos
    """
    atributos_janela = np.array([])

    ################################# JANELAMENTO ######################################
    jan_tot = round(janela / (1 / Fs))  # número de pontos de uma janela de 5s
    superpos = round(superposicao / (1 / Fs))  # número de pontos da superposição 1s
    step = jan_tot - superpos  # passo
    num_janelas = int(((len(Y) - jan_tot) / step) + 1)  # número de janelas do sinal
    

    ########################## EXTRAÇÃO DE ATRIBUTOS ###################################
    for h in range(0, num_janelas):
        vec_atributos = np.array([])

        for k in range(0, n_canais):
            Xf = Y[step * h: step * h + jan_tot, k]

            """ Calcular e extrair os atributos por janela """
            # Montar a matriz de atributos
            atr_janela = CalcAtributos.atributos(Xf)

            vec_atributos = np.append(vec_atributos, [[atr_janela]])

        if h == 0:
            atributos_janela = vec_atributos
        else:
            atributos_janela = np.vstack((atributos_janela, vec_atributos))

    return atributos_janela
