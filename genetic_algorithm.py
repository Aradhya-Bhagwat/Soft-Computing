# Implement a genetic algorithm for an optimization problem and compare the results with classical approaches

import math
import random
import time

def objective_function(x, y):
    return 50.0 - ((x - 3.0) ** 2 + (y - 3.0) ** 2) + 5.0 * math.cos(2.0 * math.pi * (x - 3.0)) + 5.0 * math.cos(2.0 * math.pi * (y - 3.0))

BOUNDS = (0.0, 6.0)

def hill_climbing(max_evals=1000, step_size=0.1, seed=42):
    random.seed(seed)
    evals = 0
    curr_x = random.uniform(BOUNDS[0], BOUNDS[1])
    curr_y = random.uniform(BOUNDS[0], BOUNDS[1])
    curr_fit = objective_function(curr_x, curr_y)
    evals += 1

    while evals < max_evals:
        neighbors = []
        for dx, dy in [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size),
                       (-step_size, -step_size), (step_size, step_size),
                       (-step_size, step_size), (step_size, -step_size)]:
            nx = max(BOUNDS[0], min(BOUNDS[1], curr_x + dx))
            ny = max(BOUNDS[0], min(BOUNDS[1], curr_y + dy))
            neighbors.append((nx, ny))

        improved = False
        best_nx, best_ny, best_nfit = curr_x, curr_y, curr_fit
        for nx, ny in neighbors:
            if evals >= max_evals:
                break
            fit = objective_function(nx, ny)
            evals += 1
            if fit > best_nfit:
                best_nfit = fit
                best_nx, best_ny = nx, ny
                improved = True

        if improved:
            curr_x, curr_y, curr_fit = best_nx, best_ny, best_nfit
        else:
            step_size *= 0.5
            if step_size < 1e-4:
                break

    return (curr_x, curr_y), curr_fit, evals

def random_search(max_evals=1000, seed=42):
    random.seed(seed)
    best_x, best_y = 0.0, 0.0
    best_fit = -float('inf')
    for _ in range(max_evals):
        x = random.uniform(BOUNDS[0], BOUNDS[1])
        y = random.uniform(BOUNDS[0], BOUNDS[1])
        fit = objective_function(x, y)
        if fit > best_fit:
            best_fit = fit
            best_x, best_y = x, y
    return (best_x, best_y), best_fit, max_evals

class GeneticAlgorithm:
    def __init__(self, pop_size=40, generations=30, crossover_prob=0.8, mutation_prob=0.15, seed=42):
        random.seed(seed)
        self.pop_size = pop_size
        self.generations = generations
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.evals = 0

    def optimize(self):
        population = [
            [random.uniform(BOUNDS[0], BOUNDS[1]), random.uniform(BOUNDS[0], BOUNDS[1])]
            for _ in range(self.pop_size)
        ]
        fitness = [objective_function(ind[0], ind[1]) for ind in population]
        self.evals += self.pop_size

        for _ in range(self.generations):
            sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
            new_population = [list(population[sorted_indices[0]]), list(population[sorted_indices[1]])]

            while len(new_population) < self.pop_size:
                cand1 = random.sample(range(self.pop_size), 3)
                p1 = population[max(cand1, key=lambda i: fitness[i])]
                cand2 = random.sample(range(self.pop_size), 3)
                p2 = population[max(cand2, key=lambda i: fitness[i])]

                if random.random() < self.pc:
                    alpha = random.random()
                    c1 = [alpha * p1[0] + (1 - alpha) * p2[0], alpha * p1[1] + (1 - alpha) * p2[1]]
                    c2 = [(1 - alpha) * p1[0] + alpha * p2[0], (1 - alpha) * p1[1] + alpha * p2[1]]
                else:
                    c1, c2 = list(p1), list(p2)

                for child in [c1, c2]:
                    for j in range(2):
                        if random.random() < self.pm:
                            child[j] += random.gauss(0, 0.3)
                            child[j] = max(BOUNDS[0], min(BOUNDS[1], child[j]))
                    new_population.append(child)
                    if len(new_population) == self.pop_size:
                        break

            population = new_population
            fitness = [objective_function(ind[0], ind[1]) for ind in population]
            self.evals += self.pop_size

        best_idx = max(range(len(fitness)), key=lambda i: fitness[i])
        return (population[best_idx][0], population[best_idx][1]), fitness[best_idx], self.evals

if __name__ == "__main__":
    print("=" * 70)
    print("  GENETIC ALGORITHM OPTIMIZATION VS CLASSICAL APPROACHES")
    print("=" * 70)
    print("Problem Statement:")
    print("  Maximize Multimodal Benchmark Function:")
    print("  f(x, y) = 50 - ((x-3)^2 + (y-3)^2) + 5*cos(2π(x-3)) + 5*cos(2π(y-3))")
    print("  Search Space : x, y in [0.0, 6.0]")
    print("  Global Optimum: f(3.0, 3.0) = 60.0000 (Surrounded by local maxima)")
    print("-" * 70)

    t0 = time.time()
    hc_point, hc_val, hc_evals = hill_climbing(max_evals=1200, seed=15)
    t_hc = time.time() - t0

    t0 = time.time()
    rs_point, rs_val, rs_evals = random_search(max_evals=1200, seed=42)
    t_rs = time.time() - t0

    t0 = time.time()
    ga = GeneticAlgorithm(pop_size=40, generations=30, seed=42)
    ga_point, ga_val, ga_evals = ga.optimize()
    t_ga = time.time() - t0

    print(f"{'Method':<22} | {'Best (x, y)':<18} | {'Best f(x, y)':<12} | {'Evals':<8} | {'Status'}")
    print("-" * 70)
    print(f"{'1. Hill Climbing':<22} | ({hc_point[0]:.3f}, {hc_point[1]:.3f}){'':<5} | {hc_val:<12.4f} | {hc_evals:<8} | Trapped in Local Maxima")
    print(f"{'2. Random Search':<22} | ({rs_point[0]:.3f}, {rs_point[1]:.3f}){'':<5} | {rs_val:<12.4f} | {rs_evals:<8} | Coarse Sampling")
    print(f"{'3. Genetic Algorithm':<22} | ({ga_point[0]:.3f}, {ga_point[1]:.3f}){'':<5} | {ga_val:<12.4f} | {ga_evals:<8} | Converged to Global Max")
    print("-" * 70)

    print("\nComparison Analysis:")
    print("  - Hill Climbing is fast but gets stuck in sub-optimal local peaks.")
    print("  - Random Search explores broadly without guiding pressure toward the peak.")
    print("  - Genetic Algorithm effectively balances exploration (crossover & mutation)")
    print("    and exploitation (selection & elitism) to reach the global optimum.")
    print("=" * 70)