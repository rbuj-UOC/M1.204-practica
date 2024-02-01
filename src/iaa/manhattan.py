# Distancia manhattan entre dos vectors amv 2 components
def dist_manhattan(x, y):
    return (((float(x[0])-float(y[0])) ** 2) ** 0.5) + (((float(x[1])-float(y[1])) ** 2) ** 0.5)
