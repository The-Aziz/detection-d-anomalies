def similarite(X1, X2, sigma=1):
    assert X1.shape == X2.shape, "Shapes of X1 and X2 must match"
    assert X1.shape[1] == 1, "X1 and X2 must be column vectors"
    diff = X1 - X2
    dist_sq = np.sum(diff ** 2)
    K = np.exp(-dist_sq / (2 * sigma ** 2))
    return K

def moyenne(M):
    assert len(M.shape) == 2, "Input must be a 2D matrix"
    moy = np.mean(M, axis=1)
    C = moy[:, np.newaxis]
    return C

def matrice(M):
    assert len(M.shape) == 2, "Input must be a 2D matrix"
    somme = np.sum(M, axis=0)
    r = M / somme
    return r

def connectivite(M, d=0.85, mas=100, tol=1e-6):
    n = M.shape[0]
    c = np.ones(n) / n
    teleport = (1 - d) / n
    for _ in range(mas):
        c_new = teleport + d * M @ c
        if np.linalg.norm(c_new - c, 1) < tol:
            break
        c = c_new
    return c

def les_moyennes(l1):
    l = []
    for m in l1:
        X = moyenne(m)
        l.append(X)
    return l

def graphe(l):
    n = len(l)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                S[i, j] = similarite(l[i], l[j])
    return S

import numpy as np

#données des joueurs
mat=[np.array([[44,44,44,44,44],[0.71,2.31,2.09,2.25,1.47],[0,2,4,1.27,4.5]]),
     np.array([[47,47,47,47,47],[2.44,2.1,3.2,2.4,2.26],[0.54,1.57,3.22,0.6,17]]),
     np.array([[45,45,45,45,45],[0.85,1.42,1.02,1.44,1.30],[2.67,0.44,4.33,1.22,0.86]]),
     np.array([[44,44,44,44,44],[2.65,1.52,2.27,1.85,2.23],[5,2.5,1.71,0.91,6.5]]),
     np.array([[48,48,48,48,48],[2.28,2.82,2.78,2.45,2.74],[1.15,3.17,0.8,2.88,0.5]]),
     np.array([[50,50,50,50,50],[1.92,1.72,1.9,1.21,1.44],[4.17,2.22,2.78,4,2.6]]),
     np.array([[48,48,48,48,48],[1.41,1.18,1.92,2.78,0.97],[2.17,3.75,4.2,1.23,2.67]]),
     np.array([[44,44,44,44,44],[2,2.44,2.04,2.72,2.88],[0.9,1.89,0.91,3.5,1.6]]),
     np.array([[54,54,54,54,54],[3.34,3.79,2.83,1.48,2.78],[2.14,3.57,1.86,2.33,2.82]]),
     np.array([[50,50,50,50,50],[0.55,3.15,2.03,3.55,2.16],[0.33,2.89,3,3.09,1.15]]),
     np.array([[52,52,52,52,52],[1.43,1.16,1.19,1.3,1.76],[0.83,3.33,6,3.14,1.15]]),
     np.array([[49,49,49,49,49],[1.74,0.98,1.33,1.27,1.37],[7.76,1,2.8,1.38,3]]),
     np.array([[47,47,47,47,47],[1.56,1.94,1.32,1.7,1.49],[0.5,1,17,1.64,2.33]]),
     np.array([[45,45,45,45,45],[1.89,1.97,3.22,1.35,1.75],[4.4,1.71,2.33,0.45,1.33]]),
     np.array([[51,51,51,51,51],[2.13,2.59,1.61,1.45,1.18],[3.86,1.4,1.75,0.89,2]]),
     np.array([[47,47,47,47,47],[0.45,1.94,2.17,2.48,2.11],[0.5,0.78,0.77,0.67,1.2]]),
     np.array([[58,58,58,58,58],[1.71,3.22,2.92,2.49,2.49],[0.8,1.16,2.29,0.33,1.14]]),
     np.array([[48,48,48,48,48],[2.5,2.03,1.93,2.58,2.13],[2.5,1.71,3,17,0.88]]),
     np.array([[50,50,50,50,50],[1.81,2.4,2.9,2.64,2.28],[3,5.67,4.8,3.1,0.75]]),
     np.array([[46,46,46,46,46],[1.03,1.43,0.94,2.21,2.02],[2,0.71,3,3,1.86]]),
     np.array([[47,47,47,47,47],[0.73,1.97,1.93,2.93,1.68],[2,1.83,1.25,2.73,1.63]]),
     np.array([[48,48,48,48,48],[1.57,1.38,1.47,1.94,1.34],[3,0.53,8.5,2.9,1.5]]),
     np.array([[49,49,49,49,49],[1.67,1.7,2.57,2.14,2.83],[6,0.83,1.07,7.5,3.75]]),
     np.array([[50,50,50,50,50],[2.96,1.8,2.65,3.14,2.33],[6,2.75,3,3.75,2.09]]),
     np.array([[58,58,58,58,58],[1.71,3.22,2.92,2.49,1.89],[0.8,1.16,2.29,0.33,2.5]]),
     np.array([[44,44,44,44,44],[1.16,0.65,1.65,2.33,2.04],[0.33,4,3.5,6.25,3.4]]),
     np.array([[47,47,47,47,47],[1.4,1.35,1.75,1.36,1.65],[2.13,1.09,2.78,6.67,1.17]]),
     np.array([[45,45,45,45,45],[1.38,1.28,2.02,2.29,1.88],[0.29,4.67,0.6,3.6,3.17]]),
     np.array([[48,48,48,48,48],[1.49,0.73,2.85,2.23,1.45],[0.64,2,2.58,4.29,0.77]]),
     np.array([[48,48,48,48,48],[2.84,2.17,1.43,2.91,1.05],[5.33,3.25,4.5,3.22,0.62]])
     ]
def anomalie(l):
    indices = [indice+1 for indice, valeur in enumerate(l) if valeur < 0.015]
    return indices
l=connectivite(matrice(graphe(les_moyennes(mat))))
print(l)
print(anomalie(l))
#courbe
import matplotlib.pyplot as plt
def plot(values):
    x = list(range(1, len(values) + 1))
    plt.plot(x, values, marker='o')
    plt.title('courbe des valeurs de connectivité')
    plt.xlabel('joueur')
    plt.ylabel('connectivité')
    plt.show()
plot(l)