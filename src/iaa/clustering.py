from math import sqrt

# Calcula el enlace completo (maximo) entre dos grupos
# (distancia maxima entre dos puntos de cada grupo)
def enllacComplet(puntos, g1, g2, dist):
    # Busca el maximo en las combinaciones de puntos
    maxima = 0.0
    for p1 in g1:
        for p2 in g2:
            d = dist(puntos[p1], puntos[p2])
            if d > maxima:
                maxima = d
    return maxima

# Calcula el enlace simple (minimo) entre dos grupos
# (distancia minima entre dos puntos de cada grupo)
def enllacSimple(puntos, g1, g2, dist):
    # Busca el minimo en las combinaciones de puntos
    minima = float("inf")
    for p1 in g1:
        for p2 in g2:
            d = dist(puntos[p1], puntos[p2])
            if d < minima:
                minima = d
    return minima

def enllacMitja(puntos, g1, g2, dist):
    s = []
    for p in g1:
        s.append(sum(dist(puntos[p], puntos[x]) for x in g2))
    return sum(s)/float((len(g1)*len(g2)))

# Dado un conjunto de puntos y un agrupamiento, fusiona
# los dos grupos mas cercanos con el criterio indicado.
# "grups" debe contener al menos dos grupos, y vuelve
# modificado, con los grupos elegidos fusionados.
def fusionaGrups(puntos, grupos, criterio, dist):
    if len(grupos) < 1: return

    # Busca el par de grupos mas adecuados (valor minimo
    # del criterio utilizado).
    minimo  = float("inf")
    nombres = grupos.keys()
    for i in range(len(nombres)-1):
        for j in range(i+1, len(nombres)):
            d = criterio(puntos, grupos[nombres[i]], grupos[nombres[j]], dist)
            if d < minimo:
                minimo    = d
                candidato = (nombres[i], nombres[j])

    # El nombre del nuevo grupo sera el mas bajo de los dos
    nombreGrupo = min(candidato)
    grupoBorrar = max(candidato)

    # Fusiona los dos grupos: anyade los elementos a uno de
    # ellos y elimina el otro del diccionario "grupos".
    grupos[nombreGrupo].extend(grupos[grupoBorrar])
    del(grupos[grupoBorrar])


# Agrupament jerarquic aglomeratiu: fusiona els grups
# fins obtenir n grups
def agrupamentAglomeratiu(diccio, ngrups, criteri, dist):
    # Generacio de l'agrupament inicial (cada punt del grup)
    grups = {x:[x] for x in diccio}
    # Fusiona els grups fins obtindre un determinat nombre de grups
    while len(grups) > ngrups:
        fusionaGrups(diccio, grups, criteri, dist)
    return grups