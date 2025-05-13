"""
Ejemplo completo de optimización de paneles solares
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithms import SolarPanelOptimizer
from genetic_algorithms.utils import (
    plot_fitness_evolution,
    generate_report,
    save_results,
    visualize_panels
)
import matplotlib.pyplot as plt


def main():
    """
    Ejemplo principal de uso del optimizador de paneles solares.
    """
    print("=== Optimizador de Paneles Solares ===")
    print("Algoritmos Genéticos - Universidad de Cundinamarca\n")
    
    # Configuración del problema
    latitude = 41.0  # Latitud en grados
    inclination = 45.0  # Inclinación de los paneles
    season = 'winter'  # Temporada crítica
    
    print(f"Configuración:")
    print(f"- Latitud: {latitude}°")
    print(f"- Inclinación: {inclination}°")
    print(f"- Temporada: {season}")
    print()
    
    # Crear optimizador
    optimizer = SolarPanelOptimizer(
        population_size=50,
        max_generations=75,
        crossover_rate=0.75,
        mutation_rate=0.1
    )
    
    # Ejecutar optimización
    print("Ejecutando optimización...")
    results = optimizer.optimize(latitude, inclination, season)
    
    # Mostrar resultados
    print("\n=== RESULTADOS ===")
    print(f"Distancia óptima: {results['optimal_distance']:.3f} metros")
    print(f"Distancia teórica: {results['theoretical_distance']:.3f} metros")
    print(f"Eficiencia: {results['efficiency']:.2f}%")
    print(f"Ángulo solar crítico: {results['solar_angle']:.1f}°")
    print(f"Generaciones: {results['generations']}")
    print(f"Tiempo de ejecución: {results['execution_time']:.2f} segundos")
    
    # Generar visualizaciones
    print("\nGenerando visualizaciones...")
    
    # 1. Evolución del fitness
    fig1 = plot_fitness_evolution(results['history'])
    plt.savefig('results/fitness_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Visualización de paneles
    fig2 = visualize_panels(
        results['optimal_distance'],
        inclination
    )
    plt.savefig('results/panel_layout.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Gráfico personalizado de optimización
    optimizer.plot_optimization(results)
    plt.savefig('results/optimization_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Guardar resultados
    save_results(results, format='json')
    
    # Generar informe
    report = generate_report(results, save_path='results/optimization_report.md')
    print("\nInforme generado: results/optimization_report.md")
    
    return results


def run_comparative_analysis():
    """
    Ejecuta un análisis comparativo con diferentes configuraciones.
    """
    configurations = [
        {
            'name': 'Estándar',
            'population_size': 50,
            'max_generations': 75,
            'crossover_rate': 0.75,
            'mutation_rate': 0.1
        },
        {
            'name': 'Población Grande',
            'population_size': 100,
            'max_generations': 50,
            'crossover_rate': 0.8,
            'mutation_rate': 0.05
        },
        {
            'name': 'Alta Mutación',
            'population_size': 50,
            'max_generations': 75,
            'crossover_rate': 0.7,
            'mutation_rate': 0.2
        }
    ]
    
    results_comparison = {}
    
    for config in configurations:
        print(f"\nEjecutando configuración: {config['name']}")
        
        optimizer = SolarPanelOptimizer(
            population_size=config['population_size'],
            max_generations=config['max_generations'],
            crossover_rate=config['crossover_rate'],
            mutation_rate=config['mutation_rate']
        )
        
        results = optimizer.optimize(41.0, 45.0, 'winter')
        results_comparison[config['name']] = results
    
    # Comparar resultados
    print("\n=== COMPARACIÓN DE RESULTADOS ===")
    for name, results in results_comparison.items():
        print(f"\n{name}:")
        print(f"  - Distancia óptima: {results['optimal_distance']:.3f} m")
        print(f"  - Fitness final: {results['best_fitness']:.4f}")
        print(f"  - Tiempo: {results['execution_time']:.2f} s")
    
    return results_comparison


def interactive_demo():
    """
    Demo interactivo para el usuario.
    """
    print("=== Demo Interactivo - Optimizador de Paneles Solares ===")
    
    try:
        # Solicitar parámetros al usuario
        latitude = float(input("Ingrese la latitud (grados, -90 a 90): "))
        inclination = float(input("Ingrese la inclinación de los paneles (grados, 0 a 90): "))
        
        print("\nSeleccione la temporada:")
        print("1. Invierno (crítico)")
        print("2. Verano")
        season_choice = input("Opción (1 o 2): ")
        season = 'winter' if season_choice == '1' else 'summer'
        
        # Configuración del algoritmo
        print("\n¿Usar configuración predeterminada? (s/n): ")
        use_default = input().lower() == 's'
        
        if use_default:
            optimizer = SolarPanelOptimizer()
        else:
            pop_size = int(input("Tamaño de población (default: 50): ") or "50")
            max_gen = int(input("Máximo de generaciones (default: 75): ") or "75")
            cross_rate = float(input("Tasa de cruce (0-1, default: 0.75): ") or "0.75")
            mut_rate = float(input("Tasa de mutación (0-1, default: 0.1): ") or "0.1")
            
            optimizer = SolarPanelOptimizer(
                population_size=pop_size,
                max_generations=max_gen,
                crossover_rate=cross_rate,
                mutation_rate=mut_rate
            )
        
        # Ejecutar optimización
        print("\nEjecutando optimización...")
        results = optimizer.optimize(latitude, inclination, season)
        
        # Mostrar resultados
        print("\n=== RESULTADOS ===")
        print(f"Distancia óptima encontrada: {results['optimal_distance']:.3f} metros")
        print(f"Distancia teórica mínima: {results['theoretical_distance']:.3f} metros")
        print(f"Diferencia: {results['optimal_distance'] - results['theoretical_distance']:.3f} metros")
        print(f"Eficiencia de espacio: {results['efficiency']:.2f}%")
        
        # Guardar resultados
        save_choice = input("\n¿Desea guardar los resultados? (s/n): ")
        if save_choice.lower() == 's':
            save_results(results)
            print("Resultados guardados exitosamente.")
        
        # Mostrar gráficos
        show_plots = input("\n¿Desea ver los gráficos? (s/n): ")
        if show_plots.lower() == 's':
            plot_fitness_evolution(results['history'])
            plt.show()
            
            visualize_panels(results['optimal_distance'], inclination)
            plt.show()
    
    except Exception as e:
        print(f"\nError: {e}")
        print("Por favor, verifique los valores ingresados.")


if __name__ == "__main__":
    # Ejecutar ejemplo principal
    main()
    
    # Descomentar para ejecutar otros ejemplos:
    # run_comparative_analysis()
    # interactive_demo()