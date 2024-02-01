from PIL import Image
from PIL import ImageOps
from iaa import clustering as UOC_C
from iaa import euclidean as UOC_E
import matplotlib.pyplot as P
import numpy as np
import os
from random import randrange
from types import *


def get_imList():
    # obtenir el cami fins a les imatges
    path = os.getcwd()
    dirList = os.listdir(path + '/cares_practica')
    # construir una llista amb el cami complet als fitxers amb extensio jpg
    imList = []
    for fname in dirList:
        file_name, file_ext = fname.split('.')
        if file_ext == 'jpg':
            imList.append(path + '/cares_practica/' + file_name + '.' + file_ext)
    return imList

def get_imgSize(fitxer):
    # obrir la primera imatge i obtenir la grandaria en pixels
    z = Image.open(fitxer)
    # convertir la imatge a escala de grisos
    ImageOps.grayscale(z)
    im = np.array(z)
    return im.shape[0:2]

def get_cetroides(indexs, M):
    data = np.zeros((1, M.shape[1]))
    for i in range(len(indexs)):
        data += M[indexs[i]]
    return data[0] / len(indexs)

def get_cardinalitat(diccionari):
    resultat = []
    for key in diccionari.iterkeys():
        resultat.append(len(diccionari[key]))
    return resultat

def kmeans(k, maxit, entrenament, numelem):
    # taula amb la informacio d'entrenament per als centroides
    conj = list(map(lambda x: [float(v) for c, v in entrenament[x[1]].items() if c <= numelem], enumerate(entrenament)))
    # generar 4 centres inicials de forma aleatoria amb les dades del vector d'entrenament
    centr = [conj[randrange(len(conj))] for i in range(k)]
    for i in range(k):
        for j in range(k):
            if j != i:
                while centr[i] == centr[j]:
                    centr[i] = conj[randrange(len(conj))]
    # Repetir la cerca k centres tenint en compte la proximitat als centres
    anteriors = None
    # per a cada punt d'entrenamanet determinar a quin grup esta mes aprop
    distancies  = list(map(lambda x: map(lambda y: UOC_E.dist_euclidiana_iguals(conj[x], centr[y]), range(len(centr))), range(len(conj))))
    # pertinensa, llista amb les pertinenses dels elements d'entrenament als grups
    pertinensa = list(map(lambda x: distancies[x].index(min(distancies[x])), range(len(distancies))))
    for it in range(maxit):
        # Recalcular els nous centroides dels k grups, que estaran en el centre geometric del conjunt de punts del grup.
        for i in range(k):
            # conj_grup conjunt amb els indexs dels elements d'entrenament que formen part d'un grup
            conj_grup = filter(lambda x: pertinensa[x] == i, range(len(pertinensa)))
            if type(conj_grup) is BooleanType:
                centr[i] = conj[randrange(len(conj))]
                print "fix 1: volta", it
            else:
                if len(conj_grup) == 0:
                    centr[i] = conj[randrange(len(conj))]
                    print "fix 2: volta", it
                else:
                    # calcular la mitja aritmetica de cada variable
                    aux = np.zeros([numelem], float)
                    for j in range(len(conj_grup)):
                        for p in range(numelem):
                            aux[p] = aux[p] + conj[conj_grup[j]][p]
                    aux = aux * (1.0 / float(len(conj_grup)))
                    # el centre de cada grup es el nou centre geometric
                    centr[i] = aux
        # per a cada punt d'entrenamanet determinar a quin grup esta mes aprop
        distancies  = list(map(lambda x: map(lambda y: UOC_E.dist_euclidiana_iguals(conj[x], centr[y]), range(len(centr))), range(len(conj))))
        # pertinensa, llista amb les pertinenses dels elements d'entrenament als grups
        pertinensa = list(map(lambda x: distancies[x].index(min(distancies[x])), range(len(distancies))))
        # Aturar en cas que no hi hagin canvis en la pertinensa
        if pertinensa == anteriors: break
        anteriors = pertinensa
    return centr, pertinensa


print "Activitat 2"
print ""

# llista amb les imatges
imList = get_imList()
NIM = len(imList)

# dimensio de les imatges, primera imatge
m, n = get_imgSize(imList[0])
NPIX = m * n

# crear una matriu de grandaria NIM x NPIX:
A = np.array([np.array(ImageOps.grayscale(Image.open(imList[i]))).flatten() for i in range(NIM)], 'f')

#######################
# PCA
#######################

# centrar les dades restant la mitjana
im_mitjana = A.mean(axis=0)
for i in range(NIM):
    A[i] -= im_mitjana
im_mitjana = im_mitjana.reshape(m, n)

# calcular PCA a partir de la descomposicio SVD de A
U, S, Vt = np.linalg.svd(A) # Vt.T = VEPS ordenats
SS = S * S.T # SS = VAPS ordenats

# Projeccio M-dimensional amb M=2 (matriu de resultats)
R_M2 = np.dot(A, Vt.T[:,:2])

#######################
# clustering
#######################
num_grups = 2
diccio_R_M2 = {}
for observacio in range(R_M2.shape[0]):
    diccio_R_M2[observacio] = {}
    for component in range(R_M2.shape[1]):
        diccio_R_M2[observacio][component] = R_M2[observacio][component]

#######################
# Enllac simple
#######################
grup_L1 = UOC_C.agrupamentAglomeratiu(diccio_R_M2, num_grups, UOC_C.enllacSimple, dist=UOC_E.dist_euclidiana_diferents)
print "Enllac simple:"
print "> Num. elements x grup:", get_cardinalitat(grup_L1)
for key in grup_L1.iterkeys():
    print "> Grup:",key
    for idx_fitxer in grup_L1[key]:
        print imList[idx_fitxer]
# centroides
centroides_L1 = []
for key in grup_L1.iterkeys():
    centroides_L1.append(list(get_cetroides(grup_L1[key], R_M2)))
print "> Centroide G1: ", centroides_L1[0]
print "> Centroide G2: ", centroides_L1[1]
print ""

#######################
# Enllac complex
#######################
grup_L2 = UOC_C.agrupamentAglomeratiu(diccio_R_M2, num_grups, UOC_C.enllacComplet, dist=UOC_E.dist_euclidiana_diferents)
print "Enllac complex:"
print "> Num. elements x grup:", get_cardinalitat(grup_L2)
for key in grup_L2.iterkeys():
    print "> Grup:",key
    for idx_fitxer in grup_L2[key]:
        print imList[idx_fitxer]
# centroides
centroides_L2 = []
for key in grup_L2.iterkeys():
    centroides_L2.append(list(get_cetroides(grup_L2[key], R_M2)))
print "> Centroide G1: ", centroides_L2[0]
print "> Centroide G2: ", centroides_L2[1]
print ""

#######################
# Enllac mitjana
#######################
grup_L3 = UOC_C.agrupamentAglomeratiu(diccio_R_M2, num_grups, UOC_C.enllacMitja, dist=UOC_E.dist_euclidiana_diferents)
print "Enllac mitjana:"
print "> Num. elements x grup:", get_cardinalitat(grup_L3)
for key in grup_L3.iterkeys():
    print "> Grup:",key
    for idx_fitxer in grup_L3[key]:
        print imList[idx_fitxer]
# centroides
centroides_L3 = []
for key in grup_L3.iterkeys():
    centroides_L3.append(list(get_cetroides(grup_L3[key], R_M2)))
print "> Centroide G1: ", centroides_L3[0]
print "> Centroide G2: ", centroides_L3[1]
print ""

#######################
# k-means
#######################
k = num_grups
maxit = 10
centroides_L4, grup_L4 = kmeans(k, maxit, diccio_R_M2, 2)
grup_L4 = list(grup_L4)
centroides_L4 = list(centroides_L4)
print "k-means:"
grup_L4_dict={}
grup_L4_dict[0]=[]
grup_L4_dict[1]=[]
for i in range(len(grup_L4)):
    grup_L4_dict[grup_L4[i]].append(i)
print "> Num. elements x grup:", get_cardinalitat(grup_L4_dict)
for key in grup_L4_dict.iterkeys():
    print "> Grup:",key
    for idx_fitxer in grup_L4_dict[key]:
        print imList[idx_fitxer]
print "> Centroide G1: ", centroides_L4[0]
print "> Centroide G2: ", centroides_L4[1]
print ""

######################
# Grafiques
######################
fig = P.figure()
fig.suptitle("Clustering", fontsize=20)
fig_M1 = fig.add_subplot(2, 2, 1)
fig_M1.set_title("Dendrograma Link simple")
fig_M2 = fig.add_subplot(2, 2, 2)
fig_M2.set_title("Dendrograma Link complex")
fig_M3 = fig.add_subplot(2, 2, 3)
fig_M3.set_title("Dendrograma Link mitjana")
fig_M4 = fig.add_subplot(2, 2, 4)
fig_M4.set_title("k-means")

colors = ['r', 'b']
# Dendrograma link simple
c = 0
for key in grup_L1.iterkeys():
    for i in grup_L1[key]:
        fig_M1.plot(R_M2[i][0], R_M2[i][1], 'x' + colors[c])
    c += 1
c = 0
for centroide in centroides_L1:
    fig_M1.plot(centroide[0], centroide[1], 'o' + colors[c])
    c += 1

# Dendrograma link complex
c = 0
for key in grup_L2.iterkeys():
    for i in grup_L2[key]:
        fig_M2.plot(R_M2[i][0], R_M2[i][1], 'x' + colors[c])
    c += 1
c = 0
for centroide in centroides_L2:
    fig_M2.plot(centroide[0], centroide[1], 'o' + colors[c])
    c += 1

# Dendrograma link mitjana
c = 0
for key in grup_L3.iterkeys():
    for i in grup_L3[key]:
        fig_M3.plot(R_M2[i][0], R_M2[i][1], 'x' + colors[c])
    c += 1
c = 0
for centroide in centroides_L3:
    fig_M3.plot(centroide[0], centroide[1], 'o' + colors[c])
    c += 1

# k-means
for i in range(len(grup_L4)):
    fig_M4.plot(R_M2[i][0], R_M2[i][1], 'x' + colors[grup_L4[i]])
c = 0
for centroide in list(centroides_L4):
    fig_M4.plot(centroide[0], centroide[1], 'o' + colors[c])
    c += 1

P.show()
