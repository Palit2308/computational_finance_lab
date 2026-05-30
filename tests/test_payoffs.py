from comp_fin_lab.payoffs import vcall, vput, vopt
import numpy as np


print("Test call S=100, K=90:", vcall(100, 90))
print("Test put S=100, K=90:", vput(100, 90))
print("Test option S=100, K=90 (call):", vopt(100, 90, call=True))
print("Test option S=100, K=90 (put):", vopt(100, 90, call=False))

S = np.arange(50, 151, 10)

print("Test array of S:", S)

print("Test call with array of S: 50 - 150, K=80:", vcall(S, 80))
print("Test put with array of S: 50 - 150, K=80:", vput(S, 80))
print("Test option with array of S: 50 - 150, K=80 (call):", vopt(S, 80, call=True))
print("Test option with array of S: 50 - 150, K=80 (put):", vopt(S, 80, call=False))