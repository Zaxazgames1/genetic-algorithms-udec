"""
Herramientas de visualización para algoritmos genéticos
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def plot_fitness_evolution(history: Dict, title: str = "Evolución del Fitness"):
    """
    Grafica la evolución del fitness a lo largo de las generaciones.
    
    Args:
        history: Historial del algoritmo genético
        title: Título del gráfico
    """
    generations = history['generations']
    best_fitness = history['best_fitness']
    avg_fitness = history['average_fitness']
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'b-', linewidth=2, label='Mejor Fitness')
    plt.plot(generations, avg_fitness, 'g--', linewidth=2, label='Fitness Promedio')
    
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def plot_population_diversity(population, generation: int = None):
    """
    Visualiza la diversidad de la población.
    
    Args:
        population: Población actual
        generation: Número de generación (opcional)
    """
    # Extraer genes de todos los individuos
    all_genes = np.array([ind.genes for ind in population.individuals])
    
    if all_genes.shape[1] == 1:
        # Para una sola dimensión
        plt.figure(figsize=(8, 6))
        plt.hist(all_genes.flatten(), bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Valor del Gen')
        plt.ylabel('Frecuencia')
        title = 'Distribución de la Población'
        if generation is not None:
            title += f' - Generación {generation}'
        plt.title(title)
        plt.grid(True, alpha=0.3)
    else:
        # Para múltiples dimensiones, mostrar scatter plot de las primeras 2
        plt.figure(figsize=(8, 6))
        plt.scatter(all_genes[:, 0], all_genes[:, 1], alpha=0.6)
        plt.xlabel('Gen 1')
        plt.ylabel('Gen 2')
        title = 'Distribución de la Población (Primeros 2 genes)'
        if generation is not None:
            title += f' - Generación {generation}'
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt.gcf()


def plot_parameter_adaptation(history: Dict):
    """
    Grafica la adaptación de parámetros durante la ejecución.
    
    Args:
        history: Historial del algoritmo genético
    """
    generations = history['generations']
    parameters = history['parameters']
    
    crossover_rates = [p['crossover_rate'] for p in parameters]
    mutation_rates = [p['mutation_rate'] for p in parameters]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Tasa de cruce
    ax1.plot(generations, crossover_rates, 'b-', linewidth=2)
    ax1.set_ylabel('Tasa de Cruce')
    ax1.set_title('Adaptación de Parámetros')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Tasa de mutación
    ax2.plot(generations, mutation_rates, 'r-', linewidth=2)
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Tasa de Mutación')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 0.3)
    
    plt.tight_layout()
    return fig


def visualize_panels(distance: float, 
                    inclination: float,
                    panel_length: float = 1.476,
                    panel_width: float = 0.659):
    """
    Visualiza la disposición de los paneles solares.
    
    Args:
        distance: Distancia entre paneles
        inclination: Ángulo de inclinación en grados
        panel_length: Longitud del panel en metros
        panel_width: Ancho del panel en metros
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convertir inclinación a radianes
    incl_rad = np.radians(inclination)
    
    # Panel 1
    x1 = [0, panel_length * np.cos(incl_rad)]
    y1 = [0, panel_length * np.sin(incl_rad)]
    
    # Panel 2
    x2 = [distance, distance + panel_length * np.cos(incl_rad)]
    y2 = [0, panel_length * np.sin(incl_rad)]
    
    # Dibujar paneles
    ax.plot(x1, y1, 'b-', linewidth=3, label='Panel 1')
    ax.plot(x2, y2, 'g-', linewidth=3, label='Panel 2')
    
    # Rellenar paneles
    ax.fill([0, x1[1], x1[1], 0], [0, y1[1], y1[1], 0], 
            'blue', alpha=0.3)
    ax.fill([x2[0], x2[1], x2[1], x2[0]], [y2[0], y2[1], y2[1], y2[0]], 
            'green', alpha=0.3)
    
    # Línea de suelo
    ax.plot([-0.5, distance + panel_length + 0.5], [0, 0], 'k-', linewidth=2)
    
    # Flecha de distancia
    ax.annotate('', xy=(distance, -0.15), xytext=(0, -0.15),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(distance/2, -0.25, f'Distancia: {distance:.2f}m', 
            ha='center', va='top', color='red', fontsize=10)
    
    # Configuración del gráfico
    ax.set_xlim(-0.5, distance + panel_length + 0.5)
    ax.set_ylim(-0.5, max(y1[1], y2[1]) + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Distancia (m)')
    ax.set_ylabel('Altura (m)')
    ax.set_title(f'Disposición de Paneles Solares - Inclinación: {inclination}°')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_animation(history: Dict, save_path: str = None):
    """
    Crea una animación de la evolución del algoritmo.
    
    Args:
        history: Historial del algoritmo genético
        save_path: Ruta para guardar la animación (opcional)
    """
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = history['generations']
    best_fitness = history['best_fitness']
    avg_fitness = history['average_fitness']
    
    ax.set_xlim(0, max(generations))
    ax.set_ylim(min(avg_fitness) * 0.9, max(best_fitness) * 1.1)
    ax.set_xlabel('Generación')
    ax.set_ylabel('Fitness')
    ax.set_title('Evolución del Algoritmo Genético')
    ax.grid(True, alpha=0.3)
    
    line_best, = ax.plot([], [], 'b-', linewidth=2, label='Mejor Fitness')
    line_avg, = ax.plot([], [], 'g--', linewidth=2, label='Fitness Promedio')
    ax.legend()
    
    def init():
        line_best.set_data([], [])
        line_avg.set_data([], [])
        return line_best, line_avg
    
    def animate(frame):
        x_data = generations[:frame+1]
        y_best = best_fitness[:frame+1]
        y_avg = avg_fitness[:frame+1]
        
        line_best.set_data(x_data, y_best)
        line_avg.set_data(x_data, y_avg)
        
        return line_best, line_avg
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(generations), interval=50,
                        blit=True, repeat=True)
    
    if save_path:
        anim.save(save_path, writer='pillow')
    
    return anim