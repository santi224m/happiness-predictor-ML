"""
A decision tree that predicts how happy someone rates themselves
on a 1-10 scale given their responses on the European Social Survey.
"""

import pandas as pd
import numpy as np

def get_split_point(input_scores, happiness_scores):
  """Calculate the best split point"""
  lowest_err = float('inf')
  best_split_point = None
  best_mean_low = np.mean(happiness_scores)
  best_mean_high = np.mean(happiness_scores)

  # Possible split points are all values in input score
  possible_split_points = set(input_scores)
  for possible_split_point in possible_split_points:
    happ_low_inputs = [happ for happ, input_score in zip(happiness_scores, input_scores) if input_score <= possible_split_point]
    happ_high_inputs = [happ for happ, input_score in zip(happiness_scores, input_scores) if input_score > possible_split_point]

    if len(happ_low_inputs) > 0 and len(happ_high_inputs) > 0:
      mean_low = np.mean(happ_low_inputs)
      mean_high = np.mean(happ_high_inputs)

      # Calcualte error
      low_err = [abs(happ - mean_low) for happ in happ_low_inputs]
      high_err = [abs(happ - mean_high) for happ in happ_high_inputs]

      total_err = sum(low_err) + sum(high_err)
      if total_err < lowest_err:
        lowest_err = total_err
        best_split_point = possible_split_point
        best_mean_low = mean_low
        best_mean_high = mean_high
  return (best_split_point, float(lowest_err), float(best_mean_low), float(best_mean_high))

def generate_tree(happiness_scores, variables):
  """Find the best metric to use for predicting happiness"""
  best_metric = None
  lowest_err = float('inf')
  best_split_point = None
  best_mean_low = np.mean(happiness_scores)
  best_mean_high = np.mean(happiness_scores)

  for var in variables:
    input_scores = list(ess.loc[:, var])
    split_point, total_error, mean_low, mean_high = get_split_point(input_scores, happiness_scores)
    if total_error < lowest_err:
      best_metric = var
      lowest_err = total_error
      best_split_point = split_point
      best_mean_low = mean_low
      best_mean_high = mean_high
  
  return (best_metric, best_split_point, lowest_err, best_mean_low, best_mean_high)
    

if __name__ == "__main__":
  # Load ess data
  ess = pd.read_csv('data/ess.csv', low_memory=False)

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
  net_usage = list(ess.loc[:, 'netustm'])
  happy = list(ess.loc[:, 'happy'])
  # best_metric, best_split_point, lowest_err, best_mean_low, best_mean_high = generate_tree(happy, variables)
  res = generate_tree(happy, variables)
  print(res)
  print(res[2] / response_count)