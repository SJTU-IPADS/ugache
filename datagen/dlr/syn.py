import sys,os
import numpy as np
sys.path.append(os.getcwd()+'/../../eval/dlr/common')
from ds_generator import generate_random_samples as generate_random_samples


max_vocabulary_size = 800000000
num_slot = 100
dense_dim = 13
'''
alpha here indicates that the probability distribution function (PDF) is proportional to (1/r)^alpha, where r is rank.
This converts to a CDF that is proportional to (1/r)^(alpha-1).
The similar 'alpha' term is also used in numpy.zipf, and ycsb's zipf generation(it's ZIPFIAN_CONSTANT).
We follow https://en.wikipedia.org/wiki/Pareto_distribution#Generating_bounded_Pareto_random_variables to generate samples, which requests alpha as CDF's exponent, so here we minus PDF alpha by 1.
'''
alpha = 1.2
dtype = np.int32
num_entries = (1000 * 8 + 1000) * 32768

def to_readable_scale(integer):
  if integer > 1000000000:
    return str(integer // 1000000000) + 'b'
  if integer > 1000000:
    return str(integer // 1000000) + 'm'
  if integer > 1000:
    return str(integer // 1000) + 'k'
  return str(integer)

FINAL_PATH = f'/datasets_dlr/processed/syn_a{int(round(alpha * 10))}_s{num_slot}_c{to_readable_scale(max_vocabulary_size)}'


if __name__ == "__main__":
  feature_spec = {"cat_" + str(i) + ".bin" : {'cardinality' : max_vocabulary_size // num_slot} for i in range(num_slot)}
  ranges = [[0, 0] for i in range(num_slot)]
  max_range = 0
  for i in range(num_slot):
      feature_cat = feature_spec['cat_' + str(i) + '.bin']['cardinality']
      ranges[i][0] = max_range
      ranges[i][1] = max_range + feature_cat
      max_range += feature_cat

  os.system(f'mkdir -p {FINAL_PATH}')
  sparse_keys, dense_features, labels = generate_random_samples(num_entries, ranges, dense_dim, dtype, alpha-1)
  sparse_keys.astype('int32').tofile(f'{FINAL_PATH}/syn.sparse')
  dense_features.astype('float32').tofile(f'{FINAL_PATH}/syn.dense')
  labels.astype('int32').tofile(f'{FINAL_PATH}/syn.label')