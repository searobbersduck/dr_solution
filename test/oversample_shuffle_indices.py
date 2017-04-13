import numpy as np

BALANCE_WEIGHTS = np.array([1.3609453700116234,  14.378223495702006,
                            6.637566137566138, 40.235967926689575,
                            49.612994350282484])

weight = BALANCE_WEIGHTS

print('weight: {}'.format(weight))


y = [22636, 2146, 4648, 746, 610]

y = np.array(y)

print('y: {}'.format(y))

weights =np.array(weight, dtype=float)

print('weight: {}'.format(weights))

from collections import Counter

counter = Counter(y)

print('counter: {}'.format(counter))

max_count = np.max(counter.values())

print('max_count: {}'.format(max_count))

indices = []

idx = [1, 2, 3, 4, 5, 6, 7]
idx = np.tile(idx , 3)
print(idx)
shuffle_idx = np.random.shuffle(idx)
print(idx)
