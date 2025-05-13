"""
Pruebas unitarias para el algoritmo genético
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithms.core import Individual, Population, GeneticAlgorithm
from genetic_algorithms.optimization import SolarPanelOptimizer


class TestIndividual(Individual):
    """Individuo de prueba para las pruebas unitarias"""
    
    def generate_random_genes(self):
        return np.random.uniform(-10, 10, 5)
    
    def calculate_fitness(self):
        # Función esfera simple
        return -np.sum(self.genes ** 2)
    
    def _mutate_gene(self, gene):
        return gene + np.random.normal(0, 0.1)


class TestGeneticAlgorithm(unittest.TestCase):
    """Pruebas para el algoritmo genético"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.ga = GeneticAlgorithm(
            individual_class=TestIndividual,
            population_size=20,
            max_generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1,
            verbose=False
        )
    
    def test_initialization(self):
        """Prueba la inicialización del algoritmo"""
        self.assertEqual(self.ga.population_size, 20)
        self.assertEqual(self.ga.max_generations, 10)
        self.assertIsNone(self.ga.population)
    
    def test_population_creation(self):
        """Prueba la creación de la población"""
        self.ga.initialize()
        self.assertIsNotNone(self.ga.population)
        self.assertEqual(len(self.ga.population.individuals), 20)
    
    def test_fitness_evaluation(self):
        """Prueba la evaluación del fitness"""
        self.ga.initialize()
        
        def fitness_function(individual):
            return individual.calculate_fitness()
        
        self.ga.population.evaluate_fitness(fitness_function)
        
        for individual in self.ga.population.individuals:
            self.assertIsNotNone(individual.fitness)
    
    def test_selection(self):
        """Prueba los métodos de selección"""
        self.ga.initialize()
        
        def fitness_function(individual):
            return individual.calculate_fitness()
        
        self.ga.population.evaluate_fitness(fitness_function)
        
        # Probar diferentes métodos
        methods = ['tournament', 'roulette', 'rank']
        
        for method in methods:
            parents = self.ga.population.select_parents(selection_method=method)
            self.assertEqual(len(parents), self.ga.population_size)
    
    def test_crossover(self):
        """Prueba el operador de cruce"""
        ind1 = TestIndividual()
        ind2 = TestIndividual()
        
        child1, child2 = ind1.crossover(ind2)
        
        self.assertEqual(len(child1.genes), len(ind1.genes))
        self.assertEqual(len(child2.genes), len(ind2.genes))
    
    def test_mutation(self):
        """Prueba el operador de mutación"""
        ind = TestIndividual()
        original_genes = ind.genes.copy()
        
        ind.mutate(1.0)  # 100% de probabilidad de mutación
        
        # Al menos un gen debería cambiar
        self.assertFalse(np.array_equal(original_genes, ind.genes))
    
    def test_evolution(self):
        """Prueba la evolución completa"""
        self.ga.initialize()
        
        def fitness_function(individual):
            return individual.calculate_fitness()
        
        results = self.ga.run(fitness_function)
        
        self.assertIn('best_solution', results)
        self.assertIn('best_fitness', results)
        self.assertIn('generations', results)
        self.assertIn('execution_time', results)
        
        # El mejor fitness debería mejorar o mantenerse
        history = results['history']
        self.assertGreaterEqual(
            history['best_fitness'][-1],
            history['best_fitness'][0]
        )


class TestSolarPanelOptimizer(unittest.TestCase):
    """Pruebas para el optimizador de paneles solares"""
    
    def setUp(self):
        """Configuración inicial"""
        self.optimizer = SolarPanelOptimizer(
            population_size=10,
            max_generations=5
        )
    
    def test_optimization(self):
        """Prueba la optimización básica"""
        results = self.optimizer.optimize(
            latitude=41.0,
            inclination=45.0,
            season='winter'
        )
        
        self.assertIn('optimal_distance', results)
        self.assertIn('theoretical_distance', results)
        self.assertIn('efficiency', results)
        
        # La distancia óptima debe ser mayor o igual a la teórica
        self.assertGreaterEqual(
            results['optimal_distance'],
            results['theoretical_distance']
        )
    
    def test_invalid_parameters(self):
        """Prueba con parámetros inválidos"""
        with self.assertRaises(ValueError):
            self.optimizer.optimize(latitude=100, inclination=45)
        
        with self.assertRaises(ValueError):
            self.optimizer.optimize(latitude=41, inclination=-10)
    
    def test_different_seasons(self):
        """Prueba con diferentes temporadas"""
        winter_results = self.optimizer.optimize(41, 45, 'winter')
        summer_results = self.optimizer.optimize(41, 45, 'summer')
        
        # Las distancias deberían ser diferentes
        self.assertNotEqual(
            winter_results['theoretical_distance'],
            summer_results['theoretical_distance']
        )


class TestPopulation(unittest.TestCase):
    """Pruebas para la clase Population"""
    
    def test_statistics(self):
        """Prueba el cálculo de estadísticas"""
        pop = Population(TestIndividual, size=10)
        
        def fitness_function(individual):
            return individual.calculate_fitness()
        
        pop.evaluate_fitness(fitness_function)
        stats = pop.get_statistics()
        
        self.assertIn('best_fitness', stats)
        self.assertIn('average_fitness', stats)
        self.assertIn('worst_fitness', stats)
        
        # Verificar orden correcto
        self.assertGreaterEqual(stats['best_fitness'], stats['average_fitness'])
        self.assertGreaterEqual(stats['average_fitness'], stats['worst_fitness'])


if __name__ == '__main__':
    unittest.main()