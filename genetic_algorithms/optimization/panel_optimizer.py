"""
Optimizador de Paneles Solares usando Algoritmos Genéticos
"""

import numpy as np
import math
from ..core.individual import Individual
from ..core.genetic_algorithm import GeneticAlgorithm


class SolarPanelIndividual(Individual):
    """
    Individuo específico para optimización de paneles solares.
    
    Representa una configuración de distancia entre paneles.
    """
    
    def __init__(self, genes=None, min_distance=1.0, max_distance=6.0):
        """
        Inicializa un individuo de panel solar.
        
        Args:
            genes: Gen que representa la distancia
            min_distance: Distancia mínima permitida
            max_distance: Distancia máxima permitida
        """
        self.min_distance = min_distance
        self.max_distance = max_distance
        super().__init__(genes)
    
    def generate_random_genes(self):
        """Genera una distancia aleatoria dentro de los límites"""
        return np.array([np.random.uniform(self.min_distance, self.max_distance)])
    
    def calculate_fitness(self, latitude, inclination, season='winter'):
        """
        Calcula el fitness basado en la configuración del panel.
        
        Args:
            latitude: Latitud en grados
            inclination: Inclinación del panel en grados
            season: Temporada ('winter' o 'summer')
            
        Returns:
            Valor de fitness
        """
        distance = self.genes[0]
        
        # Calcular distancia mínima teórica
        theoretical_distance = self._calculate_theoretical_distance(
            latitude, inclination, season
        )
        
        if theoretical_distance == float('inf'):
            return -1000  # Penalización severa
        
        # Penalizar si hay sombras
        if distance < theoretical_distance:
            return -100 * (theoretical_distance - distance)
        else:
            # Premiar distancias cercanas a la mínima teórica
            return 10 - (distance - theoretical_distance)
    
    def _calculate_theoretical_distance(self, latitude, inclination, season):
        """Calcula la distancia teórica mínima sin sombras"""
        panel_length = 1.476  # metros
        
        # Ángulo solar crítico
        if season == 'winter':
            alpha_min = (90 - latitude) - 23
        else:
            alpha_min = (90 - latitude) + 23
        
        if alpha_min <= 0:
            return float('inf')
        
        # Convertir a radianes
        inclination_rad = math.radians(inclination)
        alpha_min_rad = math.radians(alpha_min)
        
        # Fórmula de distancia
        distance = (panel_length * math.cos(inclination_rad) + 
                   (panel_length * math.sin(inclination_rad)) / 
                   math.tan(alpha_min_rad))
        
        return distance
    
    def _mutate_gene(self, gene):
        """Muta el gen dentro de los límites permitidos"""
        mutated = gene + np.random.normal(0, 0.1)
        return np.clip(mutated, self.min_distance, self.max_distance)


class SolarPanelOptimizer:
    """
    Optimizador especializado para configuraciones de paneles solares.
    
    Utiliza algoritmos genéticos adaptativos para encontrar la distancia
    óptima entre paneles solares minimizando sombras.
    """
    
    def __init__(self, 
                 population_size=50,
                 max_generations=75,
                 crossover_rate=0.75,
                 mutation_rate=0.1):
        """
        Inicializa el optimizador de paneles solares.
        
        Args:
            population_size: Tamaño de la población
            max_generations: Máximo de generaciones
            crossover_rate: Tasa inicial de cruce
            mutation_rate: Tasa inicial de mutación
        """
        self.ga = GeneticAlgorithm(
            individual_class=SolarPanelIndividual,
            population_size=population_size,
            max_generations=max_generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            adaptive_params=True,
            verbose=True
        )
        
        # Parámetros del panel
        self.panel_length = 1.476  # metros
        self.panel_width = 0.659   # metros
        self.panel_thickness = 0.035  # metros
    
    def optimize(self, latitude, inclination, season='winter'):
        """
        Optimiza la distancia entre paneles.
        
        Args:
            latitude: Latitud en grados
            inclination: Inclinación de los paneles en grados
            season: Temporada ('winter' o 'summer')
            
        Returns:
            Diccionario con resultados de la optimización
        """
        # Validar parámetros
        if not -90 <= latitude <= 90:
            raise ValueError("La latitud debe estar entre -90 y 90 grados")
        
        if not 0 <= inclination <= 90:
            raise ValueError("La inclinación debe estar entre 0 y 90 grados")
        
        # Función de fitness específica
        def fitness_function(individual):
            return individual.calculate_fitness(latitude, inclination, season)
        
        # Inicializar población
        self.ga.initialize()
        
        # Ejecutar optimización
        results = self.ga.run(fitness_function)
        
        # Calcular resultados adicionales
        best_distance = results['best_solution'].genes[0]
        theoretical_distance = self._calculate_min_distance(latitude, inclination, season)
        
        # Calcular eficiencia
        if theoretical_distance > 0:
            efficiency = ((best_distance - theoretical_distance) / theoretical_distance) * 100
        else:
            efficiency = 0
        
        # Añadir información adicional
        results['optimal_distance'] = best_distance
        results['theoretical_distance'] = theoretical_distance
        results['efficiency'] = efficiency
        results['solar_angle'] = self._calculate_solar_angle(latitude, season)
        results['configuration'] = {
            'latitude': latitude,
            'inclination': inclination,
            'season': season
        }
        
        return results
    
    def _calculate_min_distance(self, latitude, inclination, season):
        """Calcula la distancia mínima teórica sin sombras"""
        if season == 'winter':
            alpha_min = (90 - latitude) - 23
        else:
            alpha_min = (90 - latitude) + 23
        
        if alpha_min <= 0:
            return float('inf')
        
        inclination_rad = math.radians(inclination)
        alpha_min_rad = math.radians(alpha_min)
        
        distance = (self.panel_length * math.cos(inclination_rad) + 
                   (self.panel_length * math.sin(inclination_rad)) / 
                   math.tan(alpha_min_rad))
        
        return distance
    
    def _calculate_solar_angle(self, latitude, season):
        """Calcula el ángulo solar crítico"""
        if season == 'winter':
            return (90 - latitude) - 23
        else:
            return (90 - latitude) + 23
    
    def plot_optimization(self, results):
        """
        Genera visualizaciones de los resultados de optimización.
        
        Args:
            results: Diccionario con los resultados
        """
        import matplotlib.pyplot as plt
        
        history = results['history']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Gráfico de evolución del fitness
        ax1.plot(history['generations'], history['best_fitness'], 'b-', linewidth=2)
        ax1.set_xlabel('Generación')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Evolución del Algoritmo Genético - Optimización de Paneles Solares')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de distancia óptima
        best_distances = []
        for gen in range(len(history['generations'])):
            # Extraer distancia del mejor individuo en cada generación
            best_distances.append(results['final_population'].individuals[0].genes[0])
        
        ax2.plot(history['generations'], best_distances, 'g-', linewidth=2)
        ax2.axhline(y=results['theoretical_distance'], color='r', linestyle='--', 
                   label=f'Distancia teórica: {results["theoretical_distance"]:.3f}m')
        ax2.set_xlabel('Generación')
        ax2.set_ylabel('Distancia (m)')
        ax2.set_title('Convergencia de la Distancia Óptima')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()