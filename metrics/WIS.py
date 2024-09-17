import torch
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from torch.nn import MSELoss, L1Loss
import os
from matplotlib import pyplot as plt
import pandas.api.types

class ParticipantVisibleError(Exception):
    pass

def WIS_and_coverage(y_true,lower,upper,alpha):

        if np.isnan(lower)  == True:
            raise ParticipantVisibleError("lower interval value contains NaN value(s)")
        if np.isinf(lower)  == True:
            raise ParticipantVisibleError("lower interval value contains inf values(s)")
        if np.isnan(upper)  == True:
            raise ParticipantVisibleError("upper interval value contains NaN value(s)")
        if np.isinf(upper)  == True:
            raise ParticipantVisibleError("upper interval value contains inf values(s)")
        # These should not occur in a competition setting
        if np.isnan(y_true) == True:
            raise ParticipantVisibleError("y_true contains NaN value(s)")
        if np.isinf(y_true) == True:
            raise ParticipantVisibleError("y_true contains inf values(s)")

        # WIS for a single interval
        score = np.abs(upper-lower)
        # print(np.minimum(upper,lower) - y_true)
        if y_true < np.minimum(upper,lower):
            score += ((2/alpha) * (np.minimum(upper,lower) - y_true))
        if y_true > np.maximum(upper,lower):
            score += ((2/alpha) * (y_true - np.maximum(upper,lower)))
        # coverage for one single row
        coverage  = 1
        if (y_true < np.minimum(upper,lower)) or (y_true > np.maximum(upper,lower)):
            coverage = 0
        return score,coverage

v_WIS_and_coverage = np.vectorize(WIS_and_coverage)

def score(y_true,lower,upper,alpha):

        # y_true = y_true.astype(float)
        # lower  = lower.astype(float)
        # upper  = upper.astype(float)

        WIS_score,coverage = v_WIS_and_coverage(y_true,lower,upper,alpha)
        MWIS     = np.mean(WIS_score)
        coverage = coverage.sum()/coverage.shape[0]

        MWIS      = float(MWIS)
        coverage  = float(coverage)

        return MWIS,coverage