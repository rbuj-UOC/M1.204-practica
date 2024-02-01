from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as P
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

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

print "Activitat 1"
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

# Projeccio M-dimensional (matriu de resultats)
R_M1 = np.dot(A, Vt.T[:, :1])
R_M2 = np.dot(A, Vt.T[:, :2])
R_M3 = np.dot(A, Vt.T[:, :3])

# Representacio grafica de les dades
fig = P.figure()
fig.suptitle("R", fontsize=20)
fig_M1 = fig.add_subplot(2, 2, 1)
fig_M1.set_title("M=1")
fig_M2 = fig.add_subplot(2, 2, 2)
fig_M2.set_title("M=2")
fig_M3 = fig.add_subplot(2, 2, 3, projection='3d')
fig_M3.set_title("M=3")
for i in range(NIM):
    fig_M1.plot(R_M1[i], 1, 'o')
    fig_M2.plot(R_M2[i][0], R_M2[i][1], 'o')
    fig_M3.scatter(R_M3[i][0], R_M2[i][1], R_M3[i][2], 'o')
    fig_M1.annotate(str(i+1), xy=(R_M1[i], 1), xycoords='data', xytext=(-5, 10), textcoords='offset points')
    fig_M2.annotate(str(i+1), xy=(R_M2[i][0], R_M2[i][1]), xycoords='data', xytext=(-5, 10), textcoords='offset points')

P.show()