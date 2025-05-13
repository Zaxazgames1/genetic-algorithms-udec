"""
Clase Individual - Representa un individuo en la población
"""

import numpy as np
from abc import ABC, abstractmethod


class Individual(ABC):
    """
    Clase base abstracta para representar un individuo en el algoritmo genético.
    
    Attributes:
        genes: Array de genes que representa la solución
        fitness: Valor de aptitud del individuo
    """
    
    def __init__(self, genes=None):
        """
        Inicializa un individuo.
        
        Args:
            genes: Array de genes o None para generación aleatoria
        """
        self.genes = genes if genes is not None else self.generate_random_genes()
        self.fitness = None
    
    @abstractmethod
    def generate_random_genes(self):
        """Genera genes aleatorios para el individuo"""
        pass
    
    @abstractmethod
    def calculate_fitness(self, *args, **kwargs):
        """Calcula y retorna el valor de fitness del individuo"""
        pass
    
    def mutate(self, mutation_rate):
        """
        Aplica mutación al individuo.
        
        Args:
            mutation_rate: Probabilidad de mutación
        """
        for i in range(len(self.genes)):
            if np.random.random() < mutation_rate:
                self.genes[i] = self._mutate_gene(self.genes[i])
    
    @abstractmethod
    def _mutate_gene(self, gene):
        """Muta un gen específico"""
        pass
    
    def crossover(self, other):
        """
        Realiza cruce con otro individuo.
        
        Args:
            other: Otro individuo para el cruce
            
        Returns:
            Tuple de dos nuevos individuos
        """
        if len(self.genes) != len(other.genes):
            raise ValueError("Los individuos deben tener el mismo número de genes")
        
        # Cruce de un punto
        crossover_point = np.random.randint(1, len(self.genes))
        
        genes1 = np.concatenate([
            self.genes[:crossover_point],
            other.genes[crossover_point:]
        ])
        
        genes2 = np.concatenate([
            other.genes[:crossover_point],
            self.genes[crossover_point:]
        ])
        
        child1 = self.__class__(genes=genes1)
        child2 = self.__class__(genes=genes2)
        
        return child1, child2
    
    def __str__(self):
        """Representación en string del individuo"""
        return f"Individual(genes={self.genes}, fitness={self.fitness})"
    
    def __repr__(self):
        """Representación para debugging"""
        return self.__str__()