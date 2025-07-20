import numpy as np
from sklearn.ensemble import IsolationForest

def eliminar_outliers(X, Y):
    datos = np.column_stack((X, Y))
    modelo = IsolationForest(contamination=0.1, random_state=42)
    etiquetas = modelo.fit_predict(datos)
    return X[etiquetas == 1], Y[etiquetas == 1]

def validar_puntos(X, Y):
    if len(X) < 4 or len(np.unique(X)) < 4 or len(np.unique(Y)) < 4:
        raise ValueError("Se necesitan al menos 4 puntos únicos en X y Y para la interpolación.")

def derivada(T, D, z):
    d, x, y, w = T
    D1 = D[0] / ((d - x)*(d - y)*(d - w))
    D2 = D[1] / ((x - d)*(x - y)*(x - w))
    D3 = D[2] / ((y - d)*(y - x)*(y - w))
    D4 = D[3] / ((w - d)*(w - x)*(w - y))
    return (
        D1 * (3*z**2 - 2*z*(y+x+w) + x*y + (y+x)*w) +
        D2 * (3*z**2 - 2*z*(y+d+w) + y*d + (y+d)*w) +
        D3 * (3*z**2 - 2*z*(x+d+w) + x*d + (x+d)*w) +
        D4 * (3*z**2 - 2*z*(x+d+y) + d*x + (x+d)*y)
    )

def segundaderivada(T, D, z):
    d, x, y, w = T
    DD1 = D[0] / ((d - x)*(d - y)*(d - w))
    DD2 = D[1] / ((x - d)*(x - y)*(x - w))
    DD3 = D[2] / ((y - d)*(y - x)*(y - w))
    DD4 = D[3] / ((w - d)*(w - x)*(w - y))
    return (
        DD1 * (6*z - 2*(y+x+w)) +
        DD2 * (6*z - 2*(y+d+w)) +
        DD3 * (6*z - 2*(x+d+w)) +
        DD4 * (6*z - 2*(x+d+y))
    )

def mi_interpolacion(D, numero_des=35):
    valor_div = numero_des - 1
    T = np.arange(1, len(D) + 1, dtype=float)
    M = len(T)
    nc = int(M - (M + 2) % 3)

    interpolado, derivada1, derivada2 = [], [], []

    for b in range(0, nc - 3, 3):
        Ts = T[b:b + 4]
        Ds = D[b:b + 4]
        z_vals = np.linspace(Ts[0], Ts[3], valor_div + 1)

        for z in z_vals:
            denom0 = (Ts[0] - Ts[1]) * (Ts[0] - Ts[2]) * (Ts[0] - Ts[3])
            denom1 = (Ts[1] - Ts[0]) * (Ts[1] - Ts[2]) * (Ts[1] - Ts[3])
            denom2 = (Ts[2] - Ts[0]) * (Ts[2] - Ts[1]) * (Ts[2] - Ts[3])
            denom3 = (Ts[3] - Ts[0]) * (Ts[3] - Ts[1]) * (Ts[3] - Ts[2])

            L0 = Ds[0] * ((z - Ts[1]) * (z - Ts[2]) * (z - Ts[3])) / denom0
            L1 = Ds[1] * ((z - Ts[0]) * (z - Ts[2]) * (z - Ts[3])) / denom1
            L2 = Ds[2] * ((z - Ts[0]) * (z - Ts[1]) * (z - Ts[3])) / denom2
            L3 = Ds[3] * ((z - Ts[0]) * (z - Ts[1]) * (z - Ts[2])) / denom3

            interpolado.append(L0 + L1 + L2 + L3)
            derivada1.append(derivada(Ts, Ds, z))
            derivada2.append(segundaderivada(Ts, Ds, z))

    return np.array(interpolado), np.array(derivada1), np.array(derivada2)
