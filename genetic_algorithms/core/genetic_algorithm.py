"""
Clase principal del Algoritmo Genético
"""

import numpy as np
from typing import Callable, Type, Dict, Any
import time
from .population import Population
from .individual import Individual


class GeneticAlgorithm:
    """
    Implementación del Algoritmo Genético con parámetros adaptativos.
    
    Esta clase proporciona una implementación completa y configurable
    de un algoritmo genético para problemas de optimización.
    """
    
    def __init__(self, 
                 individual_class: Type[Individual],
                 population_size: int = 50,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elitism_rate: float = 0.1,
                 selection_method: str = 'tournament',
                 adaptive_params: bool = True,
                 verbose: bool = True):
        """
        Inicializa el algoritmo genético.
        
        Args:
            individual_class: Clase de Individual a usar
            population_size: Tamaño de la población
            max_generations: Número máximo de generaciones
            crossover_rate: Probabilidad inicial de cruce
            mutation_rate: Probabilidad inicial de mutación
            elitism_rate: Proporción de elite a mantener
            selection_method: Método de selección ('tournament', 'roulette', 'rank')
            adaptive_params: Si usar parámetros adaptativos
            verbose: Si mostrar información durante la ejecución
        """
        self.individual_class = individual_class
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.selection_method = selection_method
        self.adaptive_params = adaptive_params
        self.verbose = verbose
        
        # Estado del algoritmo
        self.population = None
        self.generation = 0
        self.best_solution = None
        self.execution_time = 0
        
        # Historial
        self.history = {
            'generations': [],
            'best_fitness': [],
            'average_fitness': [],
            'parameters': []
        }
    
    def initialize(self, **kwargs):
        """
        Inicializa la población.
        
        Args:
            **kwargs: Argumentos para la creación de individuos
        """
        self.population = Population(
            self.individual_class, 
            self.population_size,
            **kwargs
        )
        self.generation = 0
        
        if self.verbose:
            print(f"Población inicial creada con {self.population_size} individuos")
    
    def run(self, fitness_function: Callable, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta el algoritmo genético.
        
        Args:
            fitness_function: Función de fitness a optimizar
            **kwargs: Argumentos adicionales para la función de fitness
            
        Returns:
            Diccionario con los resultados de la optimización
        """
        if self.population is None:
            raise ValueError("Debe inicializar la población antes de ejecutar")
        
        start_time = time.time()
        
        # Parámetros adaptativos
        current_crossover_rate = self.crossover_rate
        current_mutation_rate = self.mutation_rate
        
        # Bucle principal
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluar fitness
            self.population.evaluate_fitness(fitness_function, **kwargs)
            
            # Obtener estadísticas
            stats = self.population.get_statistics()
            
            # Registrar historial
            self._update_history(stats, current_crossover_rate, current_mutation_rate)
            
            # Mostrar progreso
            if self.verbose and generation % 10 == 0:
                self._print_progress(generation, stats)
            
            # Verificar criterio de parada temprana
            if self._should_stop_early():
                if self.verbose:
                    print(f"\nConvergencia alcanzada en generación {generation}")
                break
            
            # Adaptación de parámetros
            if self.adaptive_params:
                current_crossover_rate, current_mutation_rate = self._adapt_parameters(
                    stats, current_crossover_rate, current_mutation_rate
                )
            
            # Selección
            parents = self.population.select_parents(
                selection_method=self.selection_method
            )
            
            # Crear nueva generación
            self.population.create_next_generation(
                parents, 
                current_crossover_rate, 
                current_mutation_rate
            )
        
        # Evaluación final
        self.population.evaluate_fitness(fitness_function, **kwargs)
        self.best_solution = self.population.best_individual
        self.execution_time = time.time() - start_time
        
        return self._get_results()
    
    def _adapt_parameters(self, stats, crossover_rate, mutation_rate):
        """
        Adapta los parámetros del algoritmo basándose en el progreso.
        
        Args:
            stats: Estadísticas actuales
            crossover_rate: Tasa de cruce actual
            mutation_rate: Tasa de mutación actual
            
        Returns:
            Tupla con las nuevas tasas (crossover_rate, mutation_rate)
        """
        # Si hay mejora, reducir mutación y aumentar cruce
        if len(self.history['best_fitness']) > 1:
            if stats['best_fitness'] > self.history['best_fitness'][-2]:
                mutation_rate = max(0.01, mutation_rate * 0.95)
                crossover_rate = min(0.95, crossover_rate * 1.02)
            else:
                # Sin mejora, aumentar exploración
                mutation_rate = min(0.25, mutation_rate * 1.05)
                crossover_rate = max(0.6, crossover_rate * 0.98)
        
        return crossover_rate, mutation_rate
    
    def _should_stop_early(self):
        """
        Verifica si se debe detener el algoritmo tempranamente.
        
        Returns:
            True si se debe detener, False en caso contrario
        """
        if len(self.history['best_fitness']) < 20:
            return False
        
        # Verificar si no hay mejora en las últimas 20 generaciones
        recent_fitness = self.history['best_fitness'][-20:]
        if max(recent_fitness) == min(recent_fitness):
            return True
        
        # Verificar si la mejora es mínima
        improvement = (max(recent_fitness) - min(recent_fitness)) / abs(min(recent_fitness))
        if improvement < 0.001:  # Menos del 0.1% de mejora
            return True
        
        return False
    
    def _update_history(self, stats, crossover_rate, mutation_rate):
        """Actualiza el historial del algoritmo"""
        self.history['generations'].append(self.generation)
        self.history['best_fitness'].append(stats['best_fitness'])
        self.history['average_fitness'].append(stats['average_fitness'])
        self.history['parameters'].append({
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate
        })
    
    def _print_progress(self, generation, stats):
        """Imprime el progreso del algoritmo"""
        print(f"Generación {generation:3d} | "
              f"Mejor: {stats['best_fitness']:10.4f} | "
              f"Promedio: {stats['average_fitness']:10.4f} | "
              f"Peor: {stats['worst_fitness']:10.4f}")
    
    def _get_results(self):
        """Prepara y retorna los resultados de la optimización"""
        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_solution.fitness,
            'generations': self.generation,
            'execution_time': self.execution_time,
            'history': self.history,
            'final_population': self.population
        }
    
    def plot_evolution(self):
        """Genera un gráfico de la evolución del algoritmo"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Gráfico de fitness
        plt.subplot(1, 2, 1)
        plt.plot(self.history['generations'], self.history['best_fitness'], 
                'b-', label='Mejor fitness')
        plt.plot(self.history['generations'], self.history['average_fitness'], 
                'g--', label='Fitness promedio')
        plt.xlabel('Generación')
        plt.ylabel('Fitness')
        plt.title('Evolución del Fitness')
        plt.legend()
        plt.grid(True)
        
        # Gráfico de parámetros
        plt.subplot(1, 2, 2)
        crossover_rates = [p['crossover_rate'] for p in self.history['parameters']]
        mutation_rates = [p['mutation_rate'] for p in self.history['parameters']]
        
        plt.plot(self.history['generations'], crossover_rates, 
                'r-', label='Tasa de cruce')
        plt.plot(self.history['generations'], mutation_rates, 
                'm--', label='Tasa de mutación')
        plt.xlabel('Generación')
        plt.ylabel('Valor del parámetro')
        plt.title('Parámetros Adaptativos')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()