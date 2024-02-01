from math import sqrt

# Distancia euclidiana entre dos diccionaris
# amb el mateix nombres d'elements als dos diccionaris
def dist_euclidiana_iguals(x, y):
    return float(sqrt(sum(map(lambda a, b: pow(float(a) - float(b), 2), x, y))))

# Distancia euclidiana entre dos diccionaris
# amb nombres d'elements diferent entre ambdos diccionaris
def dist_euclidiana_diferents(dic1, dic2):
    suma2 = sum([pow(dic1[elem]-dic2[elem], 2)
        for elem in dic1 if elem in dic2])
    return sqrt(suma2)

# Similitud euclidiana entre dos diccionaris
def simul_euclidiana_diferents(dic1, dic2):
    return 1/(1+dist_euclidiana_diferents(dic1, dic2))
