from PIL import Image
from PIL import ImageOps
import matplotlib.pyplot as P
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

print "Activitat 3"
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

print "3 VAPS amb mes VAP:", SS[:3]
print "> Percentatge recontruccio:", format((sum(SS[:3]) / sum(SS)) * 100.0, '.2f'), '%'
print
print "2 VAPS amb mes VAP:", SS[:2]
print "> Percentatge recontruccio:", format((sum(SS[:2]) / sum(SS)) * 100.0, '.2f'), '%'

fig = P.figure()
fig.suptitle("Eigenfaces", fontsize=20)
fig_M1 = fig.add_subplot(1, 1, 1)
fig_M1.set_title("3 VAPS amb mes VAP")
for i in range(3):
    fig_M1.plot(i + 1, SS[i], 'o')
    fig_M1.annotate(format((SS[i] / sum(SS)) * 100.0, '.2f') + '%', xy=(i + 1, SS[i]), xycoords='data', xytext=(-5, 10), textcoords='offset points')

P.show()
