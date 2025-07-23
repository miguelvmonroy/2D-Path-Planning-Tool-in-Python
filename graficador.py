import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import os

def graficar_resultado(X, Y, int_x, int_y, K, Kmax, carpeta, ts):
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
        plt.savefig(os.path.join(carpeta, f"curvatura_{ts}.pdf"), format='pdf', dpi=300)
        np.savetxt(os.path.join(carpeta, f"datos_{ts}.csv"),
                   np.column_stack([int_x, int_y, K]),
                   delimiter=",", header="X,Y,Curvatura", comments='')
        print(f"[✅] PDF y CSV exportados correctamente.")
    except Exception as e:
        print(f"[❌] Error al guardar archivos: {e}")

    plt.tight_layout()
    plt.show()
