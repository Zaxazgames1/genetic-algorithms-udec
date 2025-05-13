"""
Clase Population - Maneja una población de individuos
"""

import numpy as np
from typing import List, Type
from .individual import Individual


class Population:
    """
    Representa una población de individuos en el algoritmo genético.
    
    Attributes:
        individuals: Lista de individuos en la población
        size: Tamaño de la población
        generation: Número de generación actual
    """
    
    def __init__(self, individual_class: Type[Individual], size: int, **kwargs):
        """
        Inicializa una población.
        
        Args:
            individual_class: Clase de Individual a usar
            size: Tamaño de la población
            **kwargs: Argumentos adicionales para la creación de individuos
        """
        self.individual_class = individual_class
        self.size = size
        self.generation = 0
        self.individuals = []
        self.best_individual = None
        self.fitness_history = []
        
        # Crear población inicial
        self._initialize_population(**kwargs)
    
    def _initialize_population(self, **kwargs):
        """Crea la población inicial de individuos aleatorios"""
        for _ in range(self.size):
            individual = self.individual_class(**kwargs)
            self.individuals.append(individual)
    
    def evaluate_fitness(self, fitness_function, *args, **kwargs):
        """
        Evalúa el fitness de todos los individuos.
        
        Args:
            fitness_function: Función para calcular fitness
            *args, **kwargs: Argumentos para la función de fitness
        """
        fitness_values = []
        
        for individual in self.individuals:
            if individual.fitness is None:
                individual.fitness = fitness_function(individual, *args, **kwargs)
            fitness_values.append(individual.fitness)
        
        # Actualizar mejor individuo
        best_idx = np.argmax(fitness_values)
        self.best_individual = self.individuals[best_idx]
        
        # Registrar estadísticas
        self.fitness_history.append({
            'generation': self.generation,
            'best_fitness': max(fitness_values),
            'average_fitness': np.mean(fitness_values),
            'worst_fitness': min(fitness_values)
        })
    
    def select_parents(self, selection_method='tournament', tournament_size=3):
        """
        Selecciona padres para la siguiente generación.
        
        Args:
            selection_method: Método de selección ('tournament', 'roulette', 'rank')
            tournament_size: Tamaño del torneo (si aplica)
            
        Returns:
            Lista de padres seleccionados
        """
        parents = []
        
        if selection_method == 'tournament':
            parents = self._tournament_selection(tournament_size)
        elif selection_method == 'roulette':
            parents = self._roulette_wheel_selection()
        elif selection_method == 'rank':
            parents = self._rank_selection()
        else:
            raise ValueError(f"Método de selección no reconocido: {selection_method}")
        
        return parents
    
    def _tournament_selection(self, tournament_size):
        """Implementa selección por torneo"""
        parents = []
        
        for _ in range(self.size):
            tournament = np.random.choice(self.individuals, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _roulette_wheel_selection(self):
        """Implementa selección por ruleta"""
        fitness_values = [ind.fitness for ind in self.individuals]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return self.individuals.copy()
        
        probabilities = [f / total_fitness for f in fitness_values]
        parents = np.random.choice(self.individuals, self.size, p=probabilities)
        
        return list(parents)
    
    def _rank_selection(self):
        """Implementa selección por rango"""
        # Ordenar individuos por fitness
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness)
        
        # Asignar probabilidades basadas en rango
        ranks = range(1, len(sorted_individuals) + 1)
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        parents = np.random.choice(sorted_individuals, self.size, p=probabilities)
        return list(parents)
    
    def create_next_generation(self, parents, crossover_rate=0.8, mutation_rate=0.1):
        """
        Crea la siguiente generación a partir de los padres.
        
        Args:
            parents: Lista de padres seleccionados
            crossover_rate: Probabilidad de cruce
            mutation_rate: Probabilidad de mutación
        """
        new_individuals = []
        
        # Elitismo: mantener el mejor individuo
        if self.best_individual:
            new_individuals.append(self.best_individual)
        
        while len(new_individuals) < self.size:
            # Seleccionar dos padres
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            
            # Aplicar cruce
            if np.random.random() < crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.__class__(parent1.genes.copy()), parent2.__class__(parent2.genes.copy())
            
            # Aplicar mutación
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            
            new_individuals.extend([child1, child2])
        
        # Ajustar tamaño si es necesario
        self.individuals = new_individuals[:self.size]
        self.generation += 1
    
    def get_statistics(self):
        """Obtiene estadísticas de la población actual"""
        fitness_values = [ind.fitness for ind in self.individuals]
        
        return {
            'generation': self.generation,
            'best_fitness': max(fitness_values),
            'average_fitness': np.mean(fitness_values),
            'worst_fitness': min(fitness_values),
            'std_fitness': np.std(fitness_values)
        }
    
    def __str__(self):
        """Representación en string de la población"""
        stats = self.get_statistics()
        return (f"Population(generation={self.generation}, "
                f"size={self.size}, "
                f"best_fitness={stats['best_fitness']:.4f})")