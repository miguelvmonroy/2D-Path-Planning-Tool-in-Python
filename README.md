#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interpolación cúbica + cálculo de curvatura con exportación PDF
Autor: Miguel Eduardo (2025-07-15)
Versión corregida
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from datetime import datetime
import os

# --------------------------------------------------------------------- #
#                       FUNCIONES AUXILIARES                            #
# --------------------------------------------------------------------- #

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
#                    PROGRAMA PRINCIPAL                                 #
# --------------------------------------------------------------------- #

def main():
    # Configuración inicial
    plt.close('all')
    
    # Captura de puntos
    plt.figure("Captura de puntos", figsize=(10, 8))
    plt.title("Haz clic en 31 puntos (cierra la ventana si quieres usar datos demo)")
    try:
        puntos = plt.ginput(31, timeout=-1)
        if len(puntos) < 4:
            raise RuntimeError("Se necesitan al menos 4 puntos para la interpolación")
        X, Y = map(np.array, zip(*puntos))
    except Exception as e:
        print(f"[⚠️] {e}. Usando datos de ejemplo.")
        t = np.linspace(0, 2*np.pi, 31)
        X = np.cos(t) + 0.2*np.sin(5*t)
        Y = np.sin(t) + 0.2*np.cos(3*t)

    plt.close("Captura de puntos")
    
    # Crear figura para resultados
    fig, ax = plt.subplots(num="Resultado", figsize=(12, 10))

    # Interpolación
    try:
        int_x, dx, ddx = mi_interpolacion(X, 35)
        int_y, dy, ddy = mi_interpolacion(Y, 35)
    except Exception as e:
        print(f"[❌] Error en la interpolación: {e}")
        return

    # Cálculo de curvatura
    try:
        K = np.abs((dx*ddy - dy*ddx) / (dx**2 + dy**2)**1.5)
        K = np.log(K + 1e-12)  # Evitar log(0)
        K -= K.min()
        Kmax = K.max() if K.max() > 0 else 1.0  # Evitar división por cero
    except Exception as e:
        print(f"[❌] Error en el cálculo de curvatura: {e}")
        return

    # Configuración de colores
    cmap = colormaps['turbo']
    
    # Dibujo de la curva interpolada
    grosor = 4
    for i in range(len(K)-1):
        ax.plot([int_x[i], int_x[i+1]],
                [int_y[i], int_y[i+1]],
                color=cmap(K[i]/Kmax),
                linewidth=grosor,
                solid_capstyle='round')

    # Puntos originales
    ax.scatter(X, Y, s=80, facecolors='white', edgecolors='red', 
               linewidths=1.5, label="Puntos originales", zorder=10)
    
    # Configuración del gráfico
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title("Interpolación cúbica con visualización de curvatura", fontsize=14, pad=20)
    ax.set_xlabel("Coordenada X", fontsize=12)
    ax.set_ylabel("Coordenada Y", fontsize=12)
    
    # Ajustar límites con un margen del 10%
    x_margin = (np.max(int_x) - np.min(int_x)) * 0.1
    y_margin = (np.max(int_y) - np.min(int_y)) * 0.1
    ax.set_xlim(np.min(int_x)-x_margin, np.max(int_x)+x_margin)
    ax.set_ylim(np.min(int_y)-y_margin, np.max(int_y)+y_margin)
    
    # Barra de color
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=Kmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Logaritmo de curvatura normalizada', rotation=270, labelpad=20)
    
    plt.tight_layout()

    # Guardar PDF
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(script_dir, exist_ok=True)
        nombre_pdf = os.path.join(
            script_dir, 
            f"curvatura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        plt.savefig(nombre_pdf, format='pdf', bbox_inches='tight', dpi=300)
        print(f"[✅] Figura guardada como '{nombre_pdf}'")
    except Exception as e:
        print(f"[❌] Error al guardar el PDF: {e}")

    plt.show()

# --------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
