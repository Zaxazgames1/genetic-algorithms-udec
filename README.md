# Genetic Algorithms Library - Universidad de Cundinamarca

![Universidad de Cundinamarca](https://img.shields.io/badge/UDEC-Ingenier%C3%ADa_de_Sistemas-green)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📚 Descripción

Librería profesional para implementación de Algoritmos Genéticos desarrollada como proyecto académico del programa de Ingeniería de Sistemas y Computación de la Universidad de Cundinamarca.

**Instructor**: Ing. John M. Alvarez Cely  
**Estudiantes**: Johan Sebastian Rojas Ramirez, Julian Lara Beltran  
**Año**: 2025

## 🚀 Características

- Implementación completa de Algoritmos Genéticos
- Parámetros adaptativos automáticos
- Múltiples operadores de selección, cruce y mutación
- Aplicación específica para optimización de paneles solares
- Herramientas de visualización integradas
- Benchmarking y análisis de rendimiento

## 📦 Instalación

### Opción 1: Instalar desde PyPI (recomendado)
```bash
pip install genetic-algorithms-udec
```

### Opción 2: Instalar desde GitHub
```bash
pip install git+https://github.com/Zaxazgames1/genetic-algorithms-udec.git
```

### Opción 3: Instalar en modo desarrollo
```bash
git clone https://github.com/Zaxazgames1/genetic-algorithms-udec.git
cd genetic-algorithms-udec
pip install -e .
```

## 🎯 Uso Rápido

### Ejemplo básico: Optimización de función esfera
```python
import numpy as np
from genetic_algorithms import GeneticAlgorithm
from genetic_algorithms.core import Individual

class SimpleIndividual(Individual):
    def generate_random_genes(self):
        return np.random.uniform(-10, 10, 5)
    
    def calculate_fitness(self):
        return -np.sum(self.genes ** 2)
    
    def _mutate_gene(self, gene):
        return gene + np.random.normal(0, 0.1)

# Crear y ejecutar el algoritmo genético
ga = GeneticAlgorithm(
    individual_class=SimpleIndividual,
    population_size=50,
    max_generations=100
)

ga.initialize()
results = ga.run(lambda ind: ind.calculate_fitness())

print(f"Mejor solución: {results['best_solution'].genes}")
print(f"Mejor fitness: {results['best_fitness']}")
```

### Ejemplo aplicado: Optimización de paneles solares
```python
from genetic_algorithms import SolarPanelOptimizer

# Crear optimizador
optimizer = SolarPanelOptimizer(
    population_size=50,
    max_generations=75,
    crossover_rate=0.75,
    mutation_rate=0.1
)

# Ejecutar optimización
results = optimizer.optimize(
    latitude=41.0,      # Latitud en grados
    inclination=45.0,   # Inclinación de los paneles
    season='winter'     # Temporada crítica
)

print(f"Distancia óptima: {results['optimal_distance']:.3f} metros")
print(f"Eficiencia: {results['efficiency']:.2f}%")

# Visualizar resultados
optimizer.plot_optimization(results)
```

## 📋 Documentación

### Componentes principales

1. **Core**: Implementación base del algoritmo genético
   - `Individual`: Clase base para representar soluciones
   - `Population`: Manejo de poblaciones
   - `GeneticAlgorithm`: Algoritmo principal

2. **Optimization**: Aplicaciones específicas
   - `SolarPanelOptimizer`: Optimización de paneles solares
   - Funciones de fitness predefinidas

3. **Utils**: Utilidades y herramientas
   - Guardado/carga de resultados
   - Visualización y gráficos
   - Benchmarking y análisis

### Operadores disponibles

- **Selección**: Tournament, Roulette, Rank
- **Cruce**: Single-point, Two-point, Uniform
- **Mutación**: Gaussian, Uniform, Swap

## 🛠️ Desarrollo

### Requisitos del proyecto

- Python 3.6+
- NumPy
- Matplotlib 
- Pillow

### Instalar dependencias de desarrollo
```bash
pip install -e .[dev]
```

### Ejecutar pruebas
```bash
pytest tests/
```

### Formato de código
```bash
black genetic_algorithms/
flake8 genetic_algorithms/
```

## 📊 Resultados esperados

Los algoritmos genéticos convergen hacia soluciones óptimas mediante:
- Evolución adaptativa de parámetros
- Preservación de elite
- Balance entre exploración y explotación
- Criterios de parada inteligentes

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el repositorio
2. Crea una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 👥 Autores

- **Johan Sebastian Rojas Ramirez** - *Desarrollo principal* - [johansrojas@ucundinamarca.edu.co](mailto:johansrojas@ucundinamarca.edu.co)
- **Julian Lara Beltran** - *Colaborador*

## 🎓 Agradecimientos

- **Ing. John M. Alvarez Cely** - Instructor y guía del proyecto
- Universidad de Cundinamarca - Por el apoyo académico
- Comunidad de Python - Por las herramientas y librerías base

## 📚 Referencias

- Holland, J. H. (1975). Adaptation in Natural and Artificial Systems
- Goldberg, D. E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning
- Mitchell, M. (1998). An Introduction to Genetic Algorithms

---

*Desarrollado con ❤️ en la Universidad de Cundinamarca*