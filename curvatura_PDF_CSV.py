#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpolación cúbica + cálculo de curvatura con exportación PDF y CSV
Autor: Miguel Eduardo (2025-07-15)
Versión con mejoras (CSV input, validación, paralelización, exportación, interactividad corregida)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from datetime import datetime
import os
import mplcursors
from concurrent.futures import ThreadPoolExecutor

# --------------------------------------------------------------------- #
#                       FUNCIONES AUXILIARES                            #
# --------------------------------------------------------------------- #

def leer_puntos_desde_csv(ruta_csv):
    try:
        datos = np.loadtxt(ruta_csv, delimiter=',', skiprows=1)
        if datos.shape[1] != 2:
            raise ValueError("El archivo debe tener exactamente dos columnas: X,Y.")
        return datos[:, 0], datos[:, 1]
    except Exception as e:
        print(f"[⚠️] Error al leer CSV: {e}")
        return None, None

def validar_puntos(X, Y):
    if len(X) < 4 or len(np.unique(X)) < 4 or len(np.unique(Y)) < 4:
        raise ValueError("Se necesitan al menos 4 puntos únicos en X y Y para la interpolación.")

def derivada(T, D, z):
    d, x, y, w = T
    D1 = D[0] / ((d-x)*(d-y)*(d-w))
    D2 = D[1] / ((x-d)*(x-y)*(x-w))
    D3 = D[2] / ((y-d)*(y-x)*(y-w))
    D4 = D[3] / ((w-d)*(w-x)*(w-y))
    return (
        D1*(3*z**2 - 2*z*(y + x + w) + x*y + (y + x)*w) +
        D2*(3*z**2 - 2*z*(y + d + w) + y*d + (y + d)*w) +
        D3*(3*z**2 - 2*z*(x + d + w) + x*d + (x + d)*w) +
        D4*(3*z**2 - 2*z*(x + d + y) + d*x + (x + d)*y)
    )

def segundaderivada(T, D, z):
    d, x, y, w = T
    DD1 = D[0] / ((d-x)*(d-y)*(d-w))
    DD2 = D[1] / ((x-d)*(x-y)*(x-w))
    DD3 = D[2] / ((y-d)*(y-x)*(y-w))
    DD4 = D[3] / ((w-d)*(w-x)*(w-y))
    return (
        DD1*(6*z - 2*(y + x + w)) +
        DD2*(6*z - 2*(y + d + w)) +
        DD3*(6*z - 2*(x + d + w)) +
        DD4*(6*z - 2*(x + d + y))
    )

def mi_interpolacion(D, numero_des=35):
    valor_div = numero_des - 1
    T = np.arange(1, len(D)+1, dtype=float)
    M = len(T)
    nc = int(M - (M+2) % 3)

    aa, pd, sd = [], [], []

    for b in range(0, nc-3, 3):
        resto = T[b+3] - T[b]
        inc = resto / valor_div
        c_flag = 0 if b == (nc-4) else 1
        z_vals = np.arange(T[b], T[b+3] - c_flag*inc + 1e-12, inc)

        Ts = T[b:b+4]
        Ds = D[b:b+4]

        for z in z_vals:
            denom0 = (Ts[0]-Ts[1])*(Ts[0]-Ts[2])*(Ts[0]-Ts[3])
            denom1 = (Ts[1]-Ts[0])*(Ts[1]-Ts[2])*(Ts[1]-Ts[3])
            denom2 = (Ts[2]-Ts[0])*(Ts[2]-Ts[1])*(Ts[2]-Ts[3])
            denom3 = (Ts[3]-Ts[0])*(Ts[3]-Ts[1])*(Ts[3]-Ts[2])

            L0 = Ds[0]*((z-Ts[1])*(z-Ts[2])*(z-Ts[3])) / denom0
            L1 = Ds[1]*((z-Ts[0])*(z-Ts[2])*(z-Ts[3])) / denom1
            L2 = Ds[2]*((z-Ts[0])*(z-Ts[1])*(z-Ts[3])) / denom2
            L3 = Ds[3]*((z-Ts[0])*(z-Ts[1])*(z-Ts[2])) / denom3

            aa.append(L0 + L1 + L2 + L3)
            pd.append(derivada(Ts, Ds, z))
            sd.append(segundaderivada(Ts, Ds, z))

    return np.array(aa), np.array(pd), np.array(sd)

# --------------------------------------------------------------------- #
#                            PROGRAMA PRINCIPAL                         #
# --------------------------------------------------------------------- #

def main():
    plt.close('all')

    usar_csv = input("¿Deseas usar un archivo CSV? (s/n): ").lower().startswith("s")
    if usar_csv:
        ruta = input("Ruta del archivo CSV: ").strip()
        X, Y = leer_puntos_desde_csv(ruta)
        if X is None:
            return
    else:
        plt.figure("Captura de puntos", figsize=(10, 8))
        plt.title("Haz clic en 31 puntos (cierra la ventana si quieres usar datos demo)")
        try:
            puntos = plt.ginput(31, timeout=-1)
            if len(puntos) < 4:
                raise RuntimeError("Se necesitan al menos 4 puntos")
            X, Y = map(np.array, zip(*puntos))
        except Exception as e:
            print(f"[⚠️] {e}. Usando datos de ejemplo.")
            t = np.linspace(0, 2*np.pi, 31)
            X = np.cos(t) + 0.2*np.sin(5*t)
            Y = np.sin(t) + 0.2*np.cos(3*t)
        plt.close("Captura de puntos")

    try:
        validar_puntos(X, Y)
    except Exception as e:
        print(f"[❌] Validación fallida: {e}")
        return

    with ThreadPoolExecutor() as executor:
        future_x = executor.submit(mi_interpolacion, X, 35)
        future_y = executor.submit(mi_interpolacion, Y, 35)
        int_x, dx, ddx = future_x.result()
        int_y, dy, ddy = future_y.result()

    try:
        K = np.abs((dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5)
        K = np.log(K + 1e-12)
        K -= K.min()
        Kmax = K.max() if K.max() > 0 else 1.0
    except Exception as e:
        print(f"[❌] Error en curvatura: {e}")
        return

    fig, ax = plt.subplots(num="Resultado", figsize=(12, 10))
    cmap = colormaps['turbo']
    for i in range(len(K)-1):
        ax.plot([int_x[i], int_x[i+1]],
                [int_y[i], int_y[i+1]],
                color=cmap(K[i]/Kmax),
                linewidth=4,
                solid_capstyle='round')

    ax.scatter(X, Y, s=80, facecolors='white', edgecolors='red', linewidths=1.5, label="Puntos originales")
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title("Interpolación cúbica con visualización de curvatura", fontsize=14)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    x_margin = (np.max(int_x) - np.min(int_x)) * 0.1
    y_margin = (np.max(int_y) - np.min(int_y)) * 0.1
    ax.set_xlim(np.min(int_x)-x_margin, np.max(int_x)+x_margin)
    ax.set_ylim(np.min(int_y)-y_margin, np.max(int_y)+y_margin)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=Kmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Curvatura (log)', rotation=270, labelpad=20)

    # Interactividad con mplcursors corregida
    cursor = mplcursors.cursor(ax, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        x, y = sel.target
        distancias = np.hypot(int_x - x, int_y - y)
        idx = np.argmin(distancias)
        sel.annotation.set_text(f"({int_x[idx]:.2f}, {int_y[idx]:.2f})\nK={K[idx]:.2f}")

    plt.tight_layout()

    # Guardar PDF y CSV
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(script_dir, f"curvatura_{timestamp}.pdf"), format='pdf', dpi=300)

        np.savetxt(os.path.join(script_dir, f"datos_{timestamp}.csv"),
                   np.column_stack([int_x, int_y, K]),
                   delimiter=",", header="X,Y,Curvatura", comments='')

        print(f"[✅] Resultados guardados en PDF y CSV")
    except Exception as e:
        print(f"[❌] Error al guardar archivos: {e}")

    plt.show()

# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
