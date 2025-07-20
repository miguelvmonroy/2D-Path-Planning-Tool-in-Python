from interpolador import eliminar_outliers, validar_puntos, mi_interpolacion
from graficador import graficar_resultado
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import os

def main():
    plt.close('all')

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

    try:
        K = np.abs((dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5)
        K = np.log(K + 1e-9)
        K -= K.min()
        Kmax = K.max() if K.max() > 0 else 1.0
    except Exception as e:
        print(f"[❌] Error en curvatura: {e}")
        return

    carpeta = os.path.dirname(os.path.abspath(__file__))
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    graficar_resultado(X, Y, int_x, int_y, K, Kmax, carpeta, ts)

if __name__ == "__main__":
    main()
