"""
A decision tree that predicts how happy someone rates themselves
on a 1-10 scale given their responses on the European Social Survey.
"""

import pandas as pd
import numpy as np

if __name__ == "__main__":
  # Load ess data
  ess = pd.read_csv('data/ess.csv')

  # Choose which variables we will use for the predictor
  variables = [
    'netustm',      # Internet use, how much time on typical day, in minutes
    'sclmeet',      # How often socially meet with friends, relatives or colleagues
    'inprdsc',      # How many people with whom you can discuss intimate and personal matters
    'aesfdrk',      # Feeling of safety of walking alone in local area after dark
    'health',       # Subjective general health
    'rlgdgr',       # How religious are you
    'lkuemp',       # How likely unemployed and looking for work next 12 months
    'lknemny',      # How likely not enough money for household necessities next 12 months
    'hhmmb',        # Number of people living regularly as member of household
    'wkdcorga',     # Allowed to decide how daily work is organised
    'iorgact',      # Allowed to influence policy decisions about activities of organisation
    'ipcrtiv',      # Important to think new ideas and being creative
    'imprich',      # Important to be rich, have money and expensive things
    'ipshabt',      # Important to show abilities and be admired
    'impdiff',      # Important to try new and different things in life
    'ipfrule',      # Important to do what is told and follow rules
    'ipmodst',      # Important to be humble and modest, not draw attention
    'ipgdtim',      # Important to have a good time
    'ipbhprp',      # Important to behave properly
    'iplylfr',      # Important to be loyal to friends and devote to people close
    'impfun',       # Important to seek fun and things that give pleasure
  ]

  # Filter out entries with invalid values
  ess = ess.loc[ess['netustm'] <= 1140, :]
  ess = ess.loc[ess['sclmeet'] <= 10, :]
  ess = ess.loc[ess['inprdsc'] <= 10, :]
  ess = ess.loc[ess['rlgdgr'] <= 10, :]
  ess = ess.loc[ess['lkuemp'] <= 10, :]
  ess = ess.loc[ess['lknemny'] <= 10, :]
  ess = ess.loc[ess['hhmmb'] <= 20, :]
  ess = ess.loc[ess['wkdcorga'] <= 10, :]
  ess = ess.loc[ess['iorgact'] <= 10, :]
  ess = ess.loc[ess['happy'] <= 10, :]

  response_count = ess.shape[0]

  # Find predicted happiness by checking whether use uses more or less than 3 hours of internet per day
  split_point = 180
  net_usage = list(ess.loc[:, 'netustm'])
  happy = list(ess.loc[:, 'happy'])
  happ_low_usage = [happ for happ, usage in zip(happy, net_usage) if usage <= split_point]
  happ_high_usage = [happ for happ, usage in zip(happy, net_usage) if usage > split_point]

  mean_low = np.mean(happ_low_usage)
  mean_high = np.mean(happ_high_usage)

  # Calcualte error
  low_err = [abs(happ - mean_low) for happ in happ_low_usage]
  high_err = [abs(happ - mean_high) for happ in happ_high_usage]

  total_err = sum(low_err) + sum(high_err)
  print('Error: ', total_err)
  print('Avg error: ', total_err / response_count)