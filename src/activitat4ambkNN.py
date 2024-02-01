from PIL import Image
from PIL import ImageOps
from iaa import euclidean as UOC_E
import matplotlib.pyplot as P
import numpy as np
import os
from types import *

k = 2

def get_imList():
    # obtenir el cami fins a les imatges
    path = os.getcwd()
    dirList = ['cara8.jpg', 'cara9.jpg', 'cara79.jpg', 'cara80.jpg', 'cara81.jpg',
        'cara84.jpg', 'cara130.jpg', 'cara131.jpg', 'cara132.jpg', 'cara133.jpg',
        'cara142.jpg', 'cara157.jpg', 'cara161.jpg', 'cara166.jpg', 'cara178.jpg',
        'cara180.jpg', 'cara183.jpg', 'cara184.jpg', 'cara185.jpg', 'cara196.jpg',
        'cara202.jpg', 'cara207.jpg', 'cara208.jpg', 'cara209.jpg', 'cara241.jpg',
        'cara246.jpg', 'cara247.jpg', 'cara250.jpg', 'cara269.jpg', 'cara274.jpg',
        'cara275.jpg', 'cara287.jpg', 'cara289.jpg', 'cara298.jpg', 'cara299.jpg',
        'cara300.jpg', 'cara301.jpg', 'cara306.jpg', 'cara309.jpg', 'cara312.jpg']

    # construir una llista amb el cami complet als fitxers amb extensio jpg
    imList = []
    for fname in dirList:
        imList.append(path + '/cares_practica/' + fname)
    return imList

def get_imListNew(carpeta):
    # obtenir el cami fins a les imatges
    path = os.getcwd()
    dirList = os.listdir(path + carpeta)
    # construir una llista amb el cami complet als fitxers amb extensio jpg
    imList = []
    for fname in dirList:
        file_name, file_ext = fname.split('.')
        if file_ext == 'jpg':
            imList.append(path + carpeta + file_name + '.' + file_ext)
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

def contar(l):
    p = {}
    for x in l:
        p.setdefault(x, 0)
        p[x] += 1
    return p

def classify(t, distacia):
    ds = list(map(distacia, train, [t for x in range(len(train))]))
    kcl = contar([sorted([(ds[i], clasesTrain[i]) for i in range(len(train))], key=lambda x: x[0])[i][1] for i in range(k)])
    return max([(x, kcl[x]) for x in kcl.keys()], key=lambda x: x[1])[0]

def classify_euclidian(t, distacia=UOC_E.dist_euclidiana_iguals):
    return classify(t, distacia)


print "Activitat 4 amb knn d.euclidiana"
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
# kNN
#######################
print "kNN"
print ("> formatant dades per al classificador...")
clasesTrain = np.concatenate((np.ones((1, 10), int)[0], np.zeros((1, 10), int)[0]))
clasesTest = np.zeros((1, 20), int)[0]
train  = []
for i in range(0, len(clasesTrain), 1):
    train.append(list(R_M2[i]))
test  = []
for i in range(len(clasesTrain), len(clasesTest) + len(clasesTrain), 1):
    test.append(list(R_M2[i-len(clasesTrain)]))
print

print "kNN amb d.euclidiana conneixent el sexe de les primeres 20 cares conegudes"
# Entrenamiento
print ("> entrenament...")
clases = contar(clasesTrain)
# Clasificacion
predicciones = list(map(classify_euclidian, test))
# Numero de correctos
print "> prec.:", format(float(len(list(filter(lambda x: x[0] == x[1], zip(*[predicciones, clasesTest]))))) / float(len(test)) * 100.0, '.2f'), '%'
# combinar grups
grup = {}
grup['homes'] = []
grup['dones'] = []
for i in range(0, len(clasesTrain), 1):
    if clasesTrain[i] == 1:
        grup['homes'].append(i)
    if clasesTrain[i] == 0:
        grup['dones'].append(i)
for i in range(0, len(predicciones), 1):
    if predicciones[i] == 1:
        grup['homes'].append(i + len(clasesTrain))
    if predicciones[i] == 0:
        grup['dones'].append(i + len(clasesTrain))
print "> Num. elements x grup:", get_cardinalitat(grup)
for key in grup.iterkeys():
    print "> Grup:", key
    for idx_fitxer in grup[key]:
        print imList[idx_fitxer]
# centroides
centroides = []
for key in grup.iterkeys():
    centroides.append(list(get_cetroides(grup[key], R_M2)))
print "> Centroide G1: ", centroides[0]
print "> Centroide G2: ", centroides[1]
print ""

#######################
# classificacio
#######################
print "Classificacio minima distancia euleriana als centroides:"
# llista amb les imatges de prova
imListTest = get_imListNew('/cares_noves_practica/')
NIM_TEST = len(imListTest)
# crear una matriu de grandaria NIM_TEST x NPIX:
B = np.array([np.array(ImageOps.grayscale(Image.open(imListTest[i]))).flatten() for i in range(NIM_TEST)], 'f')

# PCA
# centrar les dades restant la mitjana
im_mitjana_test = B.mean(axis=0)
for i in range(NIM_TEST):
    B[i] -= im_mitjana_test
im_mitjana_test = im_mitjana_test.reshape(m, n)
# calcular PCA a partir de la descomposicio SVD de B
U_B, S_B, Vt_B = np.linalg.svd(B) # Vt.T = VEPS ordenats
SS_B = S_B * S_B.T # SS = VAPS ordenats
# Projeccio M-dimensional amb M=2 (matriu de resultats)
R_M2_B = np.dot(B, Vt_B.T[:,:2])

classificacio = {}
classes=['dones', 'homes']
classificacio[classes[0]] = []
classificacio[classes[1]] = []
for obserbacio in range(len(R_M2_B)):
    D = {x: UOC_E.dist_euclidiana_iguals(R_M2_B[obserbacio], centroides[x]) for x in range(2)}
    classificacio[classes[min(D, key=D.get)]].append(obserbacio)
#print "Grups:",classificacio
for key in classificacio.iterkeys():
    print "> Grup:", key
    for idx_fitxer in classificacio[key]:
        print imListTest[idx_fitxer]

# Representacio grafica de les dades
fig = P.figure()
fig.suptitle("Classificacio", fontsize=20)
fig_M1 = fig.add_subplot(1, 1, 1)
fig_M1.set_title("centoides grups kNN distancia euclidiana")

colors = ['r', 'b']
c = 0
for key in classificacio.iterkeys():
    for i in classificacio[key]:
        fig_M1.plot(R_M2_B[i][0], R_M2_B[i][1], 'x' + colors[c])
    c += 1
c = 0
for centroide in centroides:
    fig_M1.plot(centroide[0], centroide[1], 'o' + colors[c])
    c += 1
P.show()
