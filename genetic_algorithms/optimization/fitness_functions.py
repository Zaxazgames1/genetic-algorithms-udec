"""
Funciones de fitness comunes para problemas de optimización
"""

import numpy as np
import math


def sphere_function(individual):
    """
    Función esfera - problema de minimización simple.
    
    f(x) = sum(x_i^2)
    
    Args:
        individual: Individuo con genes
        
    Returns:
        Valor de fitness (negativo del valor de la función)
    """
    return -np.sum(individual.genes ** 2)


def rastrigin_function(individual):
    """
    Función de Rastrigin - problema multimodal.
    
    f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))
    
    Args:
        individual: Individuo con genes
        
    Returns:
        Valor de fitness (negativo del valor de la función)
    """
    n = len(individual.genes)
    A = 10
    value = A * n
    
    for gene in individual.genes:
        value += gene**2 - A * np.cos(2 * np.pi * gene)
    
    return -value


def rosenbrock_function(individual):
    """
    Función de Rosenbrock - problema de optimización difícil.
    
    f(x) = sum(100*(x_(i+1) - x_i^2)^2 + (1 - x_i)^2)
    
    Args:
        individual: Individuo con genes
        
    Returns:
        Valor de fitness (negativo del valor de la función)
    """
    value = 0
    for i in range(len(individual.genes) - 1):
        value += 100 * (individual.genes[i+1] - individual.genes[i]**2)**2
        value += (1 - individual.genes[i])**2
    
    return -value


def panel_fitness_function(individual, panel_config):
    """
    Función de fitness para optimización de paneles solares.
    
    Args:
        individual: Individuo con genes representando configuración
        panel_config: Diccionario con configuración del panel
        
    Returns:
        Valor de fitness
    """
    distance = individual.genes[0]
    latitude = panel_config['latitude']
    inclination = panel_config['inclination']
    season = panel_config.get('season', 'winter')
    panel_length = panel_config.get('panel_length', 1.476)
    
    # Calcular distancia mínima teórica
    if season == 'winter':
        alpha_min = (90 - latitude) - 23
    else:
        alpha_min = (90 - latitude) + 23
    
    if alpha_min <= 0:
        return -1000
    
    # Convertir a radianes
    inclination_rad = math.radians(inclination)
    alpha_min_rad = math.radians(alpha_min)
    
    # Distancia mínima sin sombras
    min_distance = (panel_length * math.cos(inclination_rad) + 
                   (panel_length * math.sin(inclination_rad)) / 
                   math.tan(alpha_min_rad))
    
    # Función de fitness
    if distance < min_distance:
        # Penalización por sombras
        return -100 * (min_distance - distance)
    else:
        # Premiar distancias cercanas a la mínima
        return 10 - (distance - min_distance)


def tsp_fitness_function(individual, distance_matrix):
    """
    Función de fitness para el Problema del Viajante (TSP).
    
    Args:
        individual: Individuo con genes representando el orden de ciudades
        distance_matrix: Matriz de distancias entre ciudades
        
    Returns:
        Valor de fitness (negativo de la distancia total)
    """
    total_distance = 0
    path = individual.genes
    
    for i in range(len(path)):
        from_city = int(path[i])
        to_city = int(path[(i + 1) % len(path)])
        total_distance += distance_matrix[from_city][to_city]
    
    return -total_distance


def knapsack_fitness_function(individual, items):
    """
    Función de fitness para el problema de la mochila.
    
    Args:
        individual: Individuo con genes binarios (0 o 1)
        items: Lista de tuplas (valor, peso)
        
    Returns:
        Valor de fitness
    """
    total_value = 0
    total_weight = 0
    max_weight = items.get('max_weight', 50)
    
    for i, gene in enumerate(individual.genes):
        if gene == 1:
            value, weight = items['items'][i]
            total_value += value
            total_weight += weight
    
    # Penalizar si excede el peso máximo
    if total_weight > max_weight:
        return -1000 * (total_weight - max_weight)
    
    return total_value