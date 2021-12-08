from generators.independent import Independent
from stats import gc

data = Independent(9, 0).generate(1000)
dim = 6
set_of_dims_1st = frozenset(range(0, int(dim / 3)))
set_of_dims_2nd = frozenset(range(int(dim / 3), int(dim / 3 * 2)))
set_of_dims_3nd = frozenset(range(int(dim / 3 * 2), int(dim)))
sets = {set_of_dims_1st, set_of_dims_2nd, set_of_dims_3nd}
print(gc.generalized_contrast("dhsic", data, sets))
