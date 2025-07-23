import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from interpolador import eliminar_outliers, validar_puntos, mi_interpolacion
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
import os
import matplotlib.pyplot as plt


class InterpolacionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Interpolador Cúbico")
        self.master.geometry("1280x720")  # Inicio widescreen

        self.notebook = ttk.Notebook(master)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # Hacer el root responsivo
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        self.crear_pestana_no_name()
        self.crear_pestana_interpolacion()

    def crear_pestana_no_name(self):
        tab1 = ttk.Frame(self.notebook)
        self.notebook.add(tab1, text="No Name")

    def crear_pestana_interpolacion(self):
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Interpolacion")

        # Diseño responsivo
        self.tab2.rowconfigure(0, weight=1)
        self.tab2.rowconfigure(1, weight=0)
        self.tab2.columnconfigure(0, weight=1)

        # Gráfica widescreen principal
        self.figura = Figure(figsize=(14, 6), dpi=100)  # Proporción 14:6 = widescreen
        self.ax = self.figura.add_subplot(111)
        self._configurar_grafico()

        self.canvas = FigureCanvasTkAgg(self.figura, master=self.tab2)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky='nsew')
        self.canvas.mpl_connect("button_press_event", self.agregar_punto)

        # Botones
        frame_botones = ttk.Frame(self.tab2)
        frame_botones.grid(row=1, column=0, pady=8)

        btn_reset = tk.Button(frame_botones, text="Resetear", command=self.resetear)
        btn_reset.pack(side="left", padx=10)

        btn_finalizar = tk.Button(frame_botones, text="Finalizar Interpolación", command=self.ejecutar_interpolacion)
        btn_finalizar.pack(side="left", padx=10)

        self.puntos = []

    def _configurar_grafico(self):
        self.ax.clear()
        self.ax.set_title("Haz clic para capturar puntos")
        self.ax.set_xlim(-2, 2)
        self.ax.set_ylim(-2, 2)
        self.ax.grid(True)
        self.ax.set_aspect('auto')  # Clave: responsivo y widescreen

    def agregar_punto(self, event):
        if event.inaxes != self.ax:
            return
        self.puntos.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, 'ro')
        self.canvas.draw()

    def resetear(self):
        self.puntos = []
        self._configurar_grafico()
        self.canvas.draw()

    def ejecutar_interpolacion(self):
        if len(self.puntos) < 4:
            messagebox.showwarning("Insuficientes puntos", "Se necesitan al menos 4 puntos únicos.")
            return

        X, Y = map(np.array, zip(*self.puntos))
        X, Y = eliminar_outliers(X, Y)

        try:
            validar_puntos(X, Y)
        except Exception as e:
            messagebox.showerror("Error de validación", str(e))
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
            messagebox.showerror("Error en curvatura", str(e))
            return

        centroide_x, centroide_y = np.mean(X), np.mean(Y)
        radios = np.hypot(X - centroide_x, Y - centroide_y)
        radio_max = np.max(radios) * 1.1

        self._configurar_grafico()
        cmap = plt.get_cmap('turbo')
        for i in range(len(K) - 1):
            self.ax.plot([int_x[i], int_x[i+1]], [int_y[i], int_y[i+1]],
                         color=cmap(K[i] / Kmax), linewidth=4, solid_capstyle='round')

        self.ax.scatter(X, Y, s=80, facecolors='white', edgecolors='red', linewidths=1.5)

        buffer = Circle((centroide_x, centroide_y), radio_max,
                        color='blue', linewidth=1.5, linestyle='--', fill=False, alpha=0.6)
        self.ax.add_patch(buffer)

        self.ax.set_title("Interpolación cúbica con curvatura + buffer")
        self.ax.set_aspect('auto')  # Clave aquí también
        self.ax.grid(True)
        self.canvas.draw()

        try:
            carpeta = os.path.dirname(os.path.abspath(__file__))
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_save, ax_save = plt.subplots(figsize=(14, 7))
            for i in range(len(K) - 1):
                ax_save.plot([int_x[i], int_x[i+1]], [int_y[i], int_y[i+1]],
                             color=cmap(K[i] / Kmax), linewidth=4, solid_capstyle='round')
            ax_save.scatter(X, Y, s=80, facecolors='white', edgecolors='red', linewidths=1.5)
            ax_save.add_patch(Circle((centroide_x, centroide_y), radio_max,
                                     color='blue', linestyle='--', linewidth=1.5, fill=False, alpha=0.6))
            ax_save.set_aspect('auto')
            ax_save.grid(True, linestyle='--', alpha=0.7)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=Kmax))
            sm.set_array([])
            fig_save.colorbar(sm, ax=ax_save, shrink=0.8, label='Curvatura (log)')
            fig_save.tight_layout()
            fig_save.savefig(os.path.join(carpeta, f"curvatura_{ts}.pdf"), format='pdf', dpi=300)
            np.savetxt(os.path.join(carpeta, f"datos_{ts}.csv"),
                       np.column_stack([int_x, int_y, K]),
                       delimiter=",", header="X,Y,Curvatura", comments='')
            plt.close(fig_save)
            print(f"[✅] curvatura_{ts}.pdf y datos_{ts}.csv exportados correctamente.")
        except Exception as e:
            print(f"[❌] Error al guardar archivos: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = InterpolacionApp(root)
    root.mainloop()
