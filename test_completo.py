"""
Script de prueba mejorado
"""

import numpy as np
from genetic_algorithms import SolarPanelOptimizer
import matplotlib.pyplot as plt

# Configurar matplotlib
plt.style.use('default')

# Crear optimizador
print("=== Optimizador de Paneles Solares ===")
print("Universidad de Cundinamarca - 2025\n")

optimizer = SolarPanelOptimizer(
    population_size=30,
    max_generations=50,
    crossover_rate=0.75,
    mutation_rate=0.1
)

# Configuración del problema
latitude = 41.0
inclination = 45.0
season = 'winter'

print(f"Configuración:")
print(f"- Latitud: {latitude}°")
print(f"- Inclinación: {inclination}°")
print(f"- Temporada: {season}")
print(f"\nEjecutando optimización...")

# Ejecutar optimización
try:
    results = optimizer.optimize(
        latitude=latitude,
        inclination=inclination,
        season=season
    )
    
    print(f"\n=== RESULTADOS ===")
    print(f"Distancia óptima encontrada: {results['optimal_distance']:.3f} metros")
    print(f"Distancia teórica mínima: {results['theoretical_distance']:.3f} metros")
    print(f"Diferencia: {results['optimal_distance'] - results['theoretical_distance']:.3f} metros")
    print(f"Eficiencia de espacio: {results['efficiency']:.2f}%")
    print(f"Ángulo solar crítico: {results['solar_angle']:.1f}°")
    print(f"Generaciones totales: {results['generations']}")
    print(f"Tiempo de ejecución: {results['execution_time']:.2f} segundos")
    
    # Visualizar resultados
    print("\nGenerando visualizaciones...")
    
    # 1. Evolución del fitness
    plt.figure(figsize=(10, 6))
    history = results['history']
    plt.plot(history['generations'], history['best_fitness'], 'b-', linewidth=2, label='Mejor Fitness')
    plt.plot(history['generations'], history['average_fitness'], 'g--', linewidth=1.5, label='Fitness Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Fitness')
    plt.title('Evolución del Algoritmo Genético')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 2. Configuración de paneles
    plt.figure(figsize=(8, 6))
    
    # Dibujar paneles
    panel_length = 1.476
    incl_rad = np.radians(inclination)
    distance = results['optimal_distance']
    
    # Panel 1
    x1 = [0, panel_length * np.cos(incl_rad)]
    y1 = [0, panel_length * np.sin(incl_rad)]
    
    # Panel 2
    x2 = [distance, distance + panel_length * np.cos(incl_rad)]
    y2 = [0, panel_length * np.sin(incl_rad)]
    
    plt.plot(x1, y1, 'b-', linewidth=4, label='Panel 1')
    plt.plot(x2, y2, 'g-', linewidth=4, label='Panel 2')
    
    # Área de los paneles
    plt.fill([0, x1[1], x1[1], 0], [0, y1[1], y1[1], 0], 'blue', alpha=0.3)
    plt.fill([x2[0], x2[1], x2[1], x2[0]], [y2[0], y2[1], y2[1], y2[0]], 'green', alpha=0.3)
    
    # Línea del suelo
    plt.plot([-0.5, distance + panel_length + 0.5], [0, 0], 'k-', linewidth=2)
    
    # Indicador de distancia
    plt.annotate('', xy=(distance, -0.2), xytext=(0, -0.2),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    plt.text(distance/2, -0.3, f'Distancia: {distance:.3f}m', 
            ha='center', va='top', color='red', fontsize=12, fontweight='bold')
    
    plt.xlim(-0.5, distance + panel_length + 0.5)
    plt.ylim(-0.5, max(y1[1], y2[1]) + 0.5)
    plt.gca().set_aspect('equal')
    plt.xlabel('Distancia (m)')
    plt.ylabel('Altura (m)')
    plt.title(f'Configuración Óptima de Paneles - Inclinación: {inclination}°')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n¡Prueba completada exitosamente!")
    
except Exception as e:
    print(f"\nError durante la ejecución: {e}")
    import traceback
    traceback.print_exc()