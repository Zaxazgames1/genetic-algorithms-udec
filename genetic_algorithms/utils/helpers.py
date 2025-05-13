"""
Funciones auxiliares y utilidades
"""

import json
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Any, List  # ← Añadir List aquí
import os


def save_results(results: Dict, filename: str = None, format: str = 'json'):
    """
    Guarda los resultados del algoritmo genético.
    
    Args:
        results: Diccionario con los resultados
        filename: Nombre del archivo (se genera automáticamente si es None)
        format: Formato de salida ('json', 'pickle')
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ga_results_{timestamp}"
    
    # Asegurar que el directorio existe
    os.makedirs('results', exist_ok=True)
    
    if format == 'json':
        # Convertir numpy arrays a listas para JSON
        json_results = _convert_for_json(results)
        
        with open(f"results/{filename}.json", 'w') as f:
            json.dump(json_results, f, indent=4)
    
    elif format == 'pickle':
        with open(f"results/{filename}.pkl", 'wb') as f:
            pickle.dump(results, f)
    
    print(f"Resultados guardados en: results/{filename}.{format}")


def load_results(filename: str, format: str = 'json') -> Dict:
    """
    Carga resultados previamente guardados.
    
    Args:
        filename: Nombre del archivo
        format: Formato del archivo ('json', 'pickle')
        
    Returns:
        Diccionario con los resultados
    """
    if format == 'json':
        with open(f"results/{filename}.json", 'r') as f:
            return json.load(f)
    
    elif format == 'pickle':
        with open(f"results/{filename}.pkl", 'rb') as f:
            return pickle.load(f)


def _convert_for_json(obj):
    """Convierte objetos numpy a formato compatible con JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return _convert_for_json(obj.__dict__)
    else:
        return obj


def generate_report(results: Dict, save_path: str = None) -> str:
    """
    Genera un informe en formato markdown de los resultados.
    
    Args:
        results: Diccionario con los resultados
        save_path: Ruta donde guardar el informe (opcional)
        
    Returns:
        String con el informe en formato markdown
    """
    report = f"""# Informe de Resultados - Algoritmo Genético

## Información General
- **Fecha de ejecución**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Tiempo de ejecución**: {results.get('execution_time', 0):.2f} segundos
- **Generaciones**: {results.get('generations', 0)}

## Mejor Solución
- **Fitness**: {results.get('best_fitness', 0):.6f}
- **Genes**: {results.get('best_solution', {}).get('genes', [])}

## Configuración del Algoritmo"""

    if 'configuration' in results:
        config = results['configuration']
        for key, value in config.items():
            report += f"\n- **{key}**: {value}"
    
    report += "\n\n## Estadísticas Finales"
    
    if 'history' in results:
        history = results['history']
        final_stats = {
            'Mejor fitness': max(history['best_fitness']),
            'Fitness promedio final': history['average_fitness'][-1],
            'Mejora total': max(history['best_fitness']) - history['best_fitness'][0],
            'Convergencia': len(history['generations'])
        }
        
        for key, value in final_stats.items():
            report += f"\n- **{key}**: {value:.6f}" if isinstance(value, float) else f"\n- **{key}**: {value}"
    
    report += "\n\n## Parámetros Específicos del Problema"
    
    if 'configuration' in results:
        problem_config = results['configuration']
        for key, value in problem_config.items():
            report += f"\n- **{key}**: {value}"
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Informe guardado en: {save_path}")
    
    return report


def calculate_statistics(population) -> Dict[str, float]:
    """
    Calcula estadísticas detalladas de una población.
    
    Args:
        population: Población a analizar
        
    Returns:
        Diccionario con estadísticas
    """
    fitness_values = [ind.fitness for ind in population.individuals 
                     if ind.fitness is not None]
    
    if not fitness_values:
        return {}
    
    genes_matrix = np.array([ind.genes for ind in population.individuals])
    
    stats = {
        'población_size': len(population.individuals),
        'fitness_max': np.max(fitness_values),
        'fitness_min': np.min(fitness_values),
        'fitness_mean': np.mean(fitness_values),
        'fitness_std': np.std(fitness_values),
        'fitness_median': np.median(fitness_values),
        'genes_mean': np.mean(genes_matrix, axis=0),
        'genes_std': np.std(genes_matrix, axis=0),
        'diversidad': np.mean(np.std(genes_matrix, axis=0))
    }
    
    return stats


def validate_parameters(params: Dict[str, Any]) -> bool:
    """
    Valida los parámetros del algoritmo genético.
    
    Args:
        params: Diccionario con parámetros
        
    Returns:
        True si los parámetros son válidos
        
    Raises:
        ValueError: Si algún parámetro es inválido
    """
    required_params = ['population_size', 'max_generations', 
                      'crossover_rate', 'mutation_rate']
    
    # Verificar parámetros requeridos
    for param in required_params:
        if param not in params:
            raise ValueError(f"Parámetro requerido faltante: {param}")
    
    # Validar rangos
    if params['population_size'] < 10:
        raise ValueError("El tamaño de población debe ser al menos 10")
    
    if params['max_generations'] < 1:
        raise ValueError("El número de generaciones debe ser al menos 1")
    
    if not 0 <= params['crossover_rate'] <= 1:
        raise ValueError("La tasa de cruce debe estar entre 0 y 1")
    
    if not 0 <= params['mutation_rate'] <= 1:
        raise ValueError("La tasa de mutación debe estar entre 0 y 1")
    
    return True


def benchmark_algorithms(algorithms: Dict[str, Any], 
                        test_problems: List[Dict],
                        num_runs: int = 10) -> Dict:
    """
    Compara el rendimiento de diferentes configuraciones.
    
    Args:
        algorithms: Diccionario con configuraciones a comparar
        test_problems: Lista de problemas de prueba
        num_runs: Número de ejecuciones por configuración
        
    Returns:
        Diccionario con resultados comparativos
    """
    results = {}
    
    for algo_name, algo_config in algorithms.items():
        results[algo_name] = {}
        
        for problem in test_problems:
            problem_name = problem['name']
            fitness_function = problem['fitness_function']
            
            run_results = []
            
            for _ in range(num_runs):
                ga = algo_config['algorithm']
                ga.initialize(**problem.get('init_params', {}))
                
                result = ga.run(fitness_function, **problem.get('run_params', {}))
                run_results.append({
                    'best_fitness': result['best_fitness'],
                    'generations': result['generations'],
                    'execution_time': result['execution_time']
                })
            
            # Calcular estadísticas
            results[algo_name][problem_name] = {
                'mean_fitness': np.mean([r['best_fitness'] for r in run_results]),
                'std_fitness': np.std([r['best_fitness'] for r in run_results]),
                'mean_time': np.mean([r['execution_time'] for r in run_results]),
                'mean_generations': np.mean([r['generations'] for r in run_results])
            }
    
    return results


def create_parameter_grid(param_ranges: Dict[str, List]) -> List[Dict]:
    """
    Crea una grilla de parámetros para búsqueda de hiperparámetros.
    
    Args:
        param_ranges: Diccionario con rangos de parámetros
        
    Returns:
        Lista de combinaciones de parámetros
    """
    from itertools import product
    
    keys = param_ranges.keys()
    values = param_ranges.values()
    
    combinations = []
    for combination in product(*values):
        param_dict = dict(zip(keys, combination))
        combinations.append(param_dict)
    
    return combinations


def export_to_csv(results: Dict, filename: str):
    """
    Exporta resultados a formato CSV.
    
    Args:
        results: Diccionario con resultados
        filename: Nombre del archivo de salida
    """
    import csv
    
    os.makedirs('exports', exist_ok=True)
    filepath = f"exports/{filename}.csv"
    
    # Preparar datos para CSV
    history = results.get('history', {})
    
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Encabezados
        headers = ['Generation', 'Best_Fitness', 'Average_Fitness', 
                  'Crossover_Rate', 'Mutation_Rate']
        writer.writerow(headers)
        
        # Datos
        for i in range(len(history.get('generations', []))):
            row = [
                history['generations'][i],
                history['best_fitness'][i],
                history['average_fitness'][i],
                history['parameters'][i]['crossover_rate'],
                history['parameters'][i]['mutation_rate']
            ]
            writer.writerow(row)
    
    print(f"Resultados exportados a: {filepath}")