"""
Genetic Algorithms Library - Universidad de Cundinamarca
======================================================

Librería profesional para implementación de Algoritmos Genéticos
desarrollada como parte del programa de Ingeniería de Sistemas y Computación.

Módulos principales:
- core: Implementación base del algoritmo genético
- optimization: Aplicaciones específicas de optimización
- utils: Utilidades y herramientas de visualización
"""

__version__ = "1.0.0"
__author__ = "Johan [Tu Nombre]"
__email__ = "tu_email@email.com"

# Importaciones principales para acceso directo
from .core.genetic_algorithm import GeneticAlgorithm
from .core.population import Population
from .core.individual import Individual
from .optimization.panel_optimizer import SolarPanelOptimizer

__all__ = [
    "GeneticAlgorithm",
    "Population", 
    "Individual",
    "SolarPanelOptimizer",
]