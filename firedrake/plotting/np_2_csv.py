import numpy as np



inverse_results = '/fenics/shared/inverse_codes/results_deterministic/numpy_results/'
inverse_var = '/fenics/shared/inverse_codes/results_deterministic/numpy_variables/'

u0load = np.load(inverse_results + 'u0_slab_array.npy')
size_u = int(u0load.shape[0] / 3)
u0load = u0load.reshape((size_u, 3), order='F')
XYZslab = np.load(inverse_var + 'XYZ_slab.npy')[0:size_u, :]



np.savetxt("u0_inversion.csv", u0load, delimiter=",")
np.savetxt("u0_XYZ.csv", XYZslab, delimiter=",")