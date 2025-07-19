#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpolación cúbica + cálculo de curvatura con exportación PDF y CSV
Autor: Miguel Eduardo (2025-07-19)
Versión con detección de outliers, mejoras de nombres y colormap actualizado
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime
import os
import mplcursors
from concurrent.futures import ProcessPoolExecutor

# -------------------- FUNCIONES AUXILIARES --------------------

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

# -------------------- PROGRAMA PRINCIPAL --------------------

def main():
    plt.close('all')

    # Captura de puntos
    plt.figure("Captura de puntos", figsize=(10, 8))
    plt.title("Haz clic en puntos (cierra la ventana para usar datos demo)")
    try:
        puntos = plt.ginput(31, timeout=-1)
        if len(puntos) < 4:
            raise RuntimeError("Se necesitan al menos 4 puntos")
        X, Y = map(np.array, zip(*puntos))
    except Exception:
        t = np.linspace(0, 2 * np.pi, 31)
        X = np.cos(t) + 0.2 * np.sin(5 * t)
        Y = np.sin(t) + 0.2 * np.cos(3 * t)
    plt.close()

    # Eliminar outliers
    X, Y = eliminar_outliers(X, Y)

    try:
        validar_puntos(X, Y)
    except Exception as e:
        print(f"[❌] Validación fallida: {e}")
        return

    with ProcessPoolExecutor() as executor:
        fx = executor.submit(mi_interpolacion, X)
        fy = executor.submit(mi_interpolacion, Y)
        int_x, dx, ddx = fx.result()
        int_y, dy, ddy = fy.result()

    # Cálculo de curvatura
    try:
        K = np.abs((dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)
        K = np.log(K + 1e-9)
        K -= K.min()
        Kmax = K.max() if K.max() > 0 else 1.0
    except Exception as e:
        print(f"[❌] Error en curvatura: {e}")
        return

    # Visualización
    fig, ax = plt.subplots(num="Resultado", figsize=(12, 10))
    cmap = plt.get_cmap('turbo')
    for i in range(len(K) - 1):
        ax.plot([int_x[i], int_x[i+1]], [int_y[i], int_y[i+1]],
                color=cmap(K[i] / Kmax), linewidth=4, solid_capstyle='round')

    ax.scatter(X, Y, s=80, facecolors='white', edgecolors='red', linewidths=1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title("Interpolación cúbica con curvatura (sin outliers)", fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=Kmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Curvatura (log)', rotation=270, labelpad=20)

    cursor = mplcursors.cursor(ax, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        idx = np.argmin(np.hypot(int_x - x, int_y - y))
        sel.annotation.set_text(f"({int_x[idx]:.2f}, {int_y[idx]:.2f})\nK={K[idx]:.2f}")

    try:
        carpeta = os.path.dirname(os.path.abspath(__file__))
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(carpeta, f"curvatura_{ts}.pdf"), format='pdf', dpi=300)
        np.savetxt(os.path.join(carpeta, f"datos_{ts}.csv"),
                   np.column_stack([int_x, int_y, K]),
                   delimiter=",", header="X,Y,Curvatura", comments='')
        print(f"[✅] PDF y CSV exportados correctamente.")
    except Exception as e:
        print(f"[❌] Error al guardar archivos: {e}")

    plt.tight_layout()
    plt.show()

# -------------------- EJECUCIÓN --------------------

if __name__ == "__main__":
    main()
