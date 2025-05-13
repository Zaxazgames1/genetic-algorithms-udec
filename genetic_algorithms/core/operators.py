"""
Operadores genéticos para el algoritmo
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from .individual import Individual


class GeneticOperator(ABC):
    """Clase base para operadores genéticos"""
    
    @abstractmethod
    def apply(self, *args, **kwargs):
        """Aplica el operador"""
        pass


class SelectionOperator(GeneticOperator):
    """Operador de selección"""
    
    def __init__(self, method='tournament'):
        """
        Inicializa el operador de selección.
        
        Args:
            method: Método de selección ('tournament', 'roulette', 'rank')
        """
        self.method = method
    
    def apply(self, population: List[Individual], num_parents: int, **kwargs) -> List[Individual]:
        """
        Aplica la selección.
        
        Args:
            population: Lista de individuos
            num_parents: Número de padres a seleccionar
            **kwargs: Argumentos adicionales según el método
            
        Returns:
            Lista de padres seleccionados
        """
        if self.method == 'tournament':
            return self._tournament_selection(population, num_parents, 
                                            kwargs.get('tournament_size', 3))
        elif self.method == 'roulette':
            return self._roulette_selection(population, num_parents)
        elif self.method == 'rank':
            return self._rank_selection(population, num_parents)
        else:
            raise ValueError(f"Método desconocido: {self.method}")
    
    def _tournament_selection(self, population, num_parents, tournament_size):
        """Selección por torneo"""
        parents = []
        
        for _ in range(num_parents):
            tournament = np.random.choice(population, tournament_size, replace=False)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _roulette_selection(self, population, num_parents):
        """Selección por ruleta"""
        fitness_values = [ind.fitness for ind in population]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return np.random.choice(population, num_parents).tolist()
        
        probabilities = [f / total_fitness for f in fitness_values]
        parents = np.random.choice(population, num_parents, p=probabilities)
        
        return list(parents)
    
    def _rank_selection(self, population, num_parents):
        """Selección por rango"""
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        ranks = range(1, len(sorted_pop) + 1)
        total_rank = sum(ranks)
        probabilities = [r / total_rank for r in ranks]
        
        parents = np.random.choice(sorted_pop, num_parents, p=probabilities)
        return list(parents)


class CrossoverOperator(GeneticOperator):
    """Operador de cruce"""
    
    def __init__(self, method='single_point'):
        """
        Inicializa el operador de cruce.
        
        Args:
            method: Método de cruce ('single_point', 'two_point', 'uniform')
        """
        self.method = method
    
    def apply(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Aplica el cruce.
        
        Args:
            parent1: Primer padre
            parent2: Segundo padre
            
        Returns:
            Tupla con dos hijos
        """
        if self.method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        elif self.method == 'two_point':
            return self._two_point_crossover(parent1, parent2)
        elif self.method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        else:
            raise ValueError(f"Método desconocido: {self.method}")
    
    def _single_point_crossover(self, parent1, parent2):
        """Cruce de un punto"""
        point = np.random.randint(1, len(parent1.genes))
        
        genes1 = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        genes2 = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
        
        child1 = parent1.__class__(genes=genes1)
        child2 = parent2.__class__(genes=genes2)
        
        return child1, child2
    
    def _two_point_crossover(self, parent1, parent2):
        """Cruce de dos puntos"""
        points = sorted(np.random.choice(range(1, len(parent1.genes)), 2, replace=False))
        
        genes1 = np.concatenate([
            parent1.genes[:points[0]],
            parent2.genes[points[0]:points[1]],
            parent1.genes[points[1]:]
        ])
        
        genes2 = np.concatenate([
            parent2.genes[:points[0]],
            parent1.genes[points[0]:points[1]],
            parent2.genes[points[1]:]
        ])
        
        child1 = parent1.__class__(genes=genes1)
        child2 = parent2.__class__(genes=genes2)
        
        return child1, child2
    
    def _uniform_crossover(self, parent1, parent2):
        """Cruce uniforme"""
        mask = np.random.rand(len(parent1.genes)) < 0.5
        
        genes1 = np.where(mask, parent1.genes, parent2.genes)
        genes2 = np.where(mask, parent2.genes, parent1.genes)
        
        child1 = parent1.__class__(genes=genes1)
        child2 = parent2.__class__(genes=genes2)
        
        return child1, child2


class MutationOperator(GeneticOperator):
    """Operador de mutación"""
    
    def __init__(self, method='gaussian'):
        """
        Inicializa el operador de mutación.
        
        Args:
            method: Método de mutación ('gaussian', 'uniform', 'swap')
        """
        self.method = method
    
    def apply(self, individual: Individual, mutation_rate: float, **kwargs):
        """
        Aplica la mutación.
        
        Args:
            individual: Individuo a mutar
            mutation_rate: Probabilidad de mutación
            **kwargs: Parámetros adicionales según el método
        """
        if self.method == 'gaussian':
            return self._gaussian_mutation(individual, mutation_rate, 
                                         kwargs.get('sigma', 0.1))
        elif self.method == 'uniform':
            return self._uniform_mutation(individual, mutation_rate, 
                                        kwargs.get('min_val', 0), 
                                        kwargs.get('max_val', 1))
        elif self.method == 'swap':
            return self._swap_mutation(individual, mutation_rate)
        else:
            raise ValueError(f"Método desconocido: {self.method}")
    
    def _gaussian_mutation(self, individual, mutation_rate, sigma):
        """Mutación gaussiana"""
        for i in range(len(individual.genes)):
            if np.random.random() < mutation_rate:
                individual.genes[i] += np.random.normal(0, sigma)
    
    def _uniform_mutation(self, individual, mutation_rate, min_val, max_val):
        """Mutación uniforme"""
        for i in range(len(individual.genes)):
            if np.random.random() < mutation_rate:
                individual.genes[i] = np.random.uniform(min_val, max_val)
    
    def _swap_mutation(self, individual, mutation_rate):
        """Mutación por intercambio"""
        if np.random.random() < mutation_rate:
            idx1, idx2 = np.random.choice(len(individual.genes), 2, replace=False)
            individual.genes[idx1], individual.genes[idx2] = \
                individual.genes[idx2], individual.genes[idx1]