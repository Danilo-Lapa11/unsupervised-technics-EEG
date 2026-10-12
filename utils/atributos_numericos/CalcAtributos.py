from scipy import signal
from scipy.stats import kurtosis
import numpy as np


def atributos(k):
    # k é o sinal janelado

    """
    AAC   %Average Amplitude Change
    DASDV %Difference Absolute Deviation
    IAV   %Intagrated Absolute Value
    LOGD  %Logarithm Detector
    MAV   %Mean Absolute Value
    MLOGK %Mean Logarithm Kernel
    RMS   %Root Mean Square
    KURT  %Kurtosis
    SSC   %Slope Sign Changes
    SSI   %Simple Square Integral
    SSI   %Simple Square Integral
    VAR   %Variance
    WFL   %Waveform Length
    ZCS   %Zero Crossings
    TM3   %Third Moment
    TM4   %Fourth Moment
    TM5   %Fifth Moment
    STD   %Standard Deviation
    MVAL  %Mean Value
    MAX   %Maximum Amplitude
    PSR   %Power Spectrum Ratio
    PKF   %Peak Frequency
    MNP   %Mean Power
    MDF   %Median Frequency
    MNF   %Mean Frequency
    TTP   %Total Power
    VCF   %Variance of Central Frequency
    SM1   %1st Spectral Moments
    SM2   %2s     "        "
    SM3   %3rd    "        "
    """

    [P, F] = signal.periodogram(k)
    lin = len(P)

    # calcular os atributos
    AAC = (1 / lin) * (np.sum(np.abs(np.diff(k))))
    DASDV = np.sqrt((1 / (lin - 1)) * np.sum(np.diff(k) ** 2))
    IAV = np.sum(k)
    LOGD = np.exp((1 / lin) * (np.sum(np.log10(0.0001 + (np.abs(k))))))
    MAV = (1 / lin) * (np.sum(np.abs(k)))
    MLOGK = (1 / lin) * (np.abs(np.sum(k)))
    RMS = np.sqrt((1 / lin) * (np.sum(k ** 2)))
    KURT = kurtosis(k)
    SSC = len(np.nonzero(np.diff(np.sign(np.diff(k)))))
    SSI = np.sum(k ** 2)
    VAR = np.var(k)
    ZCS = len(np.nonzero(np.diff(np.sign(k))))
    TM3 = np.abs((1 / lin) * (np.sum(k ** 3)))
    TM4 = np.abs((1 / lin) * (np.sum(k ** 4)))
    TM5 = np.abs((1 / lin) * (np.sum(k ** 5)))
    STD = np.std(k)
    MVAL = (1 / lin) * (np.sum(k))
    MAX = np.amax(k)
    PSR = np.amax(P) / (np.sum(P))
    MNF = np.sum(F * P) / np.sum(P)
    MNP = np.sum(P) / len(F)
    PKF = np.amax(P)
    TTP = np.sum(P)
    SM1 = np.sum(F * P)
    SM2 = np.sum((F ** 2) * P)
    SM3 = np.sum((F ** 3) * P)
    VCF = (((SM2 / TTP) - (SM1 / TTP)) ** 2)
    MDF = (1 / 2) * (np.sum(P))

    allFeatures = np.array([AAC, DASDV, IAV, LOGD, MAV, MLOGK, RMS, KURT, SSC, SSI, VAR, ZCS, TM3, TM4, TM5, STD,
                            MVAL, MAX, PSR, MNF, MNP, PKF, TTP, SM1, SM2, SM3, VCF, MDF])

    return allFeatures
