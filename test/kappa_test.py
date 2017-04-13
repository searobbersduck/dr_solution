import ml_metrics

a = [2,2,2,2,5,6,7,8,9,3]
b = [2,2,2,2,2,6,5,7,8,3]

res = ml_metrics.quadratic_weighted_kappa(a,b,2,9)
print(res)