
# Interpolación Cúbica y Curvatura

Este proyecto realiza una interpolación cúbica sobre puntos definidos por el usuario,
calcula la curvatura en cada punto y exporta los resultados en archivos PDF y CSV.

## 📦 Estructura del Proyecto

```
interpolacion_modular/
├── main.py            # Script principal
├── interpolador.py    # Funciones matemáticas y validaciones
└── graficador.py      # Visualización y exportación de resultados
```

## 🛠️ Requisitos

- Python 3.8 o superior
- Paquetes necesarios:

```bash
pip install numpy matplotlib mplcursors scikit-learn
```

## 🚀 Uso

1. Ejecuta `main.py`
2. Selecciona puntos con el mouse (mínimo 4)
3. Visualiza la interpolación y curvatura
4. Se generarán:
   - Un archivo PDF con la curva coloreada por curvatura
   - Un archivo CSV con los valores interpolados y su curvatura

## 🧠 Créditos

Autor: Miguel Eduardo Venegas Monroy  
Fecha: Julio 2025





# 2D-Path-Planning-Tool-in-Python

![image](https://raw.githubusercontent.com/miguelvmonroy/python-patch/refs/heads/main/Pantalla.png)




This program was developed in PYTHON and aims to visualize a curved trajectory generated from points selected by the user. Through a mathematical interpolation process, the code smooths the trajectory and calculates its curvature, that is, it measures how much it bends or changes direction in each segment.

The selected points are visually displayed, along with a line that changes color according to the curvature: the more intense the curvature, the more the color changes. This can be useful, for example, in trajectory analysis for robots, vehicles, or any system that must follow a specific path.

The program was designed to be interactive, visual, and efficient, graphically showing how a trajectory behaves with different levels of curvature. Essentially, it is a tool that helps interpret movements and shapes intuitively, with potential applications in the design, simulation, and analysis of trajectories in real-world environments.

<p align="center">
  <img src="https://raw.githubusercontent.com/miguelvmonroy/omnidirectional-mobile-robot/refs/heads/main/FotosVehiculo.jpg" alt="Vehículo Omnidireccional" />
</p>



## Table of Contents

🛠 Technologies Used

Python
