import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import copy
import math
from functools import lru_cache
import time
from typing import List, Tuple, Dict, Optional, Set, Callable
import pickle
import os
from scipy.spatial import cKDTree

# Define rectangle type as (width, height, value)
RectangleType = Tuple[float, float, float]

# Define solution as list of (rectangle_type_index, center_x, center_y)
Solution = List[Tuple[int, float, float]]

class CircleRectanglePacker:
    def __init__(self, rectangle_types: List[RectangleType],
                 population_size: int = 100,
                 initial_mutation_rate: float = 0.2,
                 initial_crossover_rate: float = 0.7,
                 elite_size: int = 5,
                 initial_tournament_size: int = 5,
                 max_generations: int = 100,
                 diversity_threshold: float = 0.2,
                 cache_size: int = 1024):
        """
        Initialize the genetic algorithm for rectangle packing in a unit circle.

        Args:
            rectangle_types: List of available rectangle types as (width, height, value)
            population_size: Size of the population
            initial_mutation_rate: Initial probability of mutation
            initial_crossover_rate: Probability of crossover
            elite_size: Number of best solutions to carry to next generation
            initial_tournament_size: Initial size of tournament for selection
            max_generations: Maximum number of generations
            diversity_threshold: Minimum population diversity before triggering injection
            cache_size: Size of LRU cache for validation checks
        """
        self.rectangle_types = rectangle_types
        self.population_size = population_size
        self.mutation_rate = initial_mutation_rate
        self.initial_mutation_rate = initial_mutation_rate
        self.crossover_rate = initial_crossover_rate
        self.elite_size = elite_size
        self.tournament_size = initial_tournament_size
        self.initial_tournament_size = initial_tournament_size
        self.max_generations = max_generations
        self.circle_radius = 1.0  # Unit circle
        self.circle_area = np.pi * (self.circle_radius ** 2)
        self.diversity_threshold = diversity_threshold
        self.validation_cache = {}
        self.collision_cache = {}

        # Adaptive algorithm parameters
        self.stagnation_counter = 0
        self.improvement_streak = 0
        self.best_fitness_history = []
        self.diversity_history = []

        # Precompute value-to-area ratios and areas for each rectangle type
        self.value_area_ratios = []
        self.rectangle_areas = []
        for width, height, value in rectangle_types:
            area = width * height
            self.rectangle_areas.append(area)
            self.value_area_ratios.append(value / area if area > 0 else 0)

        # Cache validation and collision detection results
        self.is_valid_solution = lru_cache(maxsize=cache_size)(self._is_valid_solution)
        self.rectangles_overlap = lru_cache(maxsize=cache_size)(self._rectangles_overlap)

        # Initialize population
        self.population = self.generate_initial_population()

    def generate_single_specimen(self) -> Solution:
        """Generate a single valid solution."""
        solution = []

        # Try to add random rectangles until we fail multiple times
        consecutive_failures = 0
        max_failures = 10

        while consecutive_failures < max_failures:
            # Pick a rectangle type with preference for high value/area ratio
            rect_type_idx = self.select_rectangle_type_for_addition(solution)
            width, height, _ = self.rectangle_types[rect_type_idx]

            # Generate a random position that keeps the rectangle mostly inside the circle
            max_r = self.circle_radius - max(width, height) / 2.0
            if max_r <= 0:
                # This rectangle is too big for the circle
                consecutive_failures += 1
                continue

            # Use polar coordinates for more uniform distribution
            r = random.uniform(0, max_r)
            theta = random.uniform(0, 2 * np.pi)

            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Create a candidate solution by adding this rectangle
            candidate = solution + [(rect_type_idx, x, y)]

            # Check if candidate is valid (no overlaps)
            if self.is_valid_solution(tuple(candidate)):
                solution = candidate
                consecutive_failures = 0
            else:
                consecutive_failures += 1

        return solution

    def generate_initial_population(self) -> List[Solution]:
        """Generate initial random population with diversity."""
        population = []

        # Generate initial population
        for _ in range(self.population_size):
            solution = self.generate_single_specimen()
            population.append(solution)

        # Ensure minimal diversity in starting population
        diversity = self.calculate_population_diversity(population)
        if diversity < self.diversity_threshold:
            # Replace some specimens to increase diversity
            num_replacements = int(self.population_size * 0.3)
            for _ in range(num_replacements):
                idx = random.randint(0, len(population) - 1)
                # Don't replace the best solution
                if idx == 0:
                    continue
                population[idx] = self.generate_single_specimen()

        return population

    def calculate_filled_ratio(self, solution: Solution) -> float:
        """Calculate the ratio of circle area that is filled."""
        if not solution:
            return 0.0

        total_rect_area = 0.0
        for rect_type_idx, _, _ in solution:
            width, height, _ = self.rectangle_types[rect_type_idx]
            total_rect_area += width * height

        return min(1.0, total_rect_area / self.circle_area)

    def select_rectangle_type_for_addition(self, solution: Solution) -> int:
        """
        Select a rectangle type to add based on the current filled ratio of the circle.
        - For empty circles: favor high value/area ratio
        - For filled circles: favor small area and shapes that fit better
        """
        filled_ratio = self.calculate_filled_ratio(solution)

        # Calculate weights for each rectangle type
        weights = []
        for i in range(len(self.rectangle_types)):
            width, height, value = self.rectangle_types[i]

            # Dynamic weighting factors
            value_area_weight = self.value_area_ratios[i]
            small_area_weight = 1.0 / max(self.rectangle_areas[i], 0.001)

            # Calculate fitness for shape aspect ratio (squares fit better in later stages)
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 999
            shape_fitness = 1.0 / (1.0 + abs(aspect_ratio - 1.0))

            # Linear blend between the weights based on filled_ratio
            # As circle fills up, favor smaller rectangles with better aspect ratios
            weight = (1 - filled_ratio) * value_area_weight + \
                     filled_ratio * (0.7 * small_area_weight + 0.3 * shape_fitness)

            weights.append(weight)

        # Normalize weights to get probabilities
        total_weight = sum(weights)
        if total_weight <= 0:
            # Fallback to uniform selection if all weights are zero
            return random.randint(0, len(self.rectangle_types) - 1)

        probs = [w / total_weight for w in weights]

        # Select rectangle type using weighted random choice
        return random.choices(range(len(self.rectangle_types)), probs, k=1)[0]

    def _rectangles_overlap(self, rect1: Tuple[int, float, float], rect2: Tuple[int, float, float]) -> bool:
        """Check if two rectangles overlap."""
        rect_type_idx1, x1, y1 = rect1
        rect_type_idx2, x2, y2 = rect2

        width1, height1, _ = self.rectangle_types[rect_type_idx1]
        width2, height2, _ = self.rectangle_types[rect_type_idx2]

        # Calculate rectangle boundaries
        left1, right1 = x1 - width1/2, x1 + width1/2
        bottom1, top1 = y1 - height1/2, y1 + height1/2

        left2, right2 = x2 - width2/2, x2 + width2/2
        bottom2, top2 = y2 - height2/2, y2 + height2/2

        # Check for overlap
        return not (right1 <= left2 or right2 <= left1 or top1 <= bottom2 or top2 <= bottom1)

    def _is_valid_solution(self, solution: Tuple[Tuple[int, float, float], ...]) -> bool:
        """Check if a solution is valid (all rectangles within circle and no overlaps)."""
        if not solution:
            return True

        # For cache effectiveness, take solution as tuple
        solution = list(solution)

        # Build spatial index for faster collision detection when solution is large
        if len(solution) > 10:
            return self._is_valid_with_spatial_index(solution)

        # Check each rectangle
        for i, (rect_type_idx1, x1, y1) in enumerate(solution):
            width1, height1, _ = self.rectangle_types[rect_type_idx1]

            # Check if rectangle is within the circle
            rect_corners = [
                (x1 - width1/2, y1 - height1/2),  # Bottom-left
                (x1 + width1/2, y1 - height1/2),  # Bottom-right
                (x1 - width1/2, y1 + height1/2),  # Top-left
                (x1 + width1/2, y1 + height1/2)   # Top-right
            ]

            for corner_x, corner_y in rect_corners:
                if corner_x**2 + corner_y**2 > self.circle_radius**2:
                    return False  # Corner outside circle

            # Check for overlap with other rectangles
            for j in range(i):
                if self.rectangles_overlap(solution[i], solution[j]):
                    return False  # Overlap detected

        return True

    def _is_valid_with_spatial_index(self, solution: List[Tuple[int, float, float]]) -> bool:
        """
        Check if a solution is valid using spatial indexing for faster collision detection.
        This is more efficient for solutions with many rectangles.
        """
        rectangle_data = []

        for i, (rect_type_idx, x, y) in enumerate(solution):
            width, height, _ = self.rectangle_types[rect_type_idx]

            # Check if rectangle is within the circle
            rect_corners = [
                (x - width/2, y - height/2),  # Bottom-left
                (x + width/2, y - height/2),  # Bottom-right
                (x - width/2, y + height/2),  # Top-left
                (x + width/2, y + height/2)   # Top-right
            ]

            for corner_x, corner_y in rect_corners:
                if corner_x**2 + corner_y**2 > self.circle_radius**2:
                    return False  # Corner outside circle

            # Store rectangle data for spatial indexing
            half_width, half_height = width/2, height/2
            rectangle_data.append({
                'index': i,
                'position': (x, y),
                'bounds': (x - half_width, y - half_height, x + half_width, y + half_height)
            })

        # Build kdtree with rectangle centers
        if len(rectangle_data) > 1:
            centers = np.array([rect['position'] for rect in rectangle_data])
            tree = cKDTree(centers)

            # For each rectangle, find potential neighbors and check for collisions
            max_diagonal = max(
                math.sqrt(self.rectangle_types[rect_idx][0]**2 + self.rectangle_types[rect_idx][1]**2)
                for rect_idx, _, _ in solution
            )

            for i, rect in enumerate(rectangle_data):
                # Query the tree for potential neighbors
                # Search radius is sum of max diagonals to ensure we don't miss any collisions
                indices = tree.query_ball_point(rect['position'], 2 * max_diagonal)

                # Remove self from indices
                indices = [j for j in indices if j != i]

                # Check for collisions with potential neighbors
                for j in indices:
                    if self.rectangles_overlap(solution[i], solution[j]):
                        return False

        return True

    def calculate_fitness(self, solution: Solution) -> float:
        """Calculate fitness (total value) of a solution."""
        if not self.is_valid_solution(tuple(solution)):
            return 0.0

        total_value = 0.0
        for rect_type_idx, _, _ in solution:
            _, _, value = self.rectangle_types[rect_type_idx]
            total_value += value

        return total_value

    def select_parent(self, fitness_scores: List[float]) -> Solution:
        """Select a parent using tournament selection."""
        tournament_indices = random.sample(range(len(self.population)), self.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return self.population[winner_idx]

    def crossover(self, parent1: Solution, parent2: Solution) -> Tuple[Solution, Solution]:
        """
        Perform spatial crossover between two parents.
        Uses a random line through the circle to divide rectangles between children.
        """
        if random.random() > self.crossover_rate or len(parent1) == 0 or len(parent2) == 0:
            return parent1, parent2  # No crossover

        # Generate a random line through the circle
        # Represented as ax + by + c = 0
        theta = random.uniform(0, np.pi)  # Random angle for the line direction

        # Unit vector for line direction
        dx, dy = np.cos(theta), np.sin(theta)

        # Random point on the line within the circle
        r = random.uniform(0, self.circle_radius)
        point_x = r * np.cos(random.uniform(0, 2 * np.pi))
        point_y = r * np.sin(random.uniform(0, 2 * np.pi))

        # Line coefficients (ax + by + c = 0)
        a, b = dy, -dx
        c = -a * point_x - b * point_y

        # Initialize children
        child1, child2 = [], []

        # Assign rectangles from parent1
        for rect in parent1:
            rect_type_idx, x, y = rect
            # Check which side of the line the rectangle center is on
            side = a * x + b * y + c
            if side > 0:
                child1.append(rect)
            else:
                child2.append(rect)

        # Assign rectangles from parent2
        for rect in parent2:
            rect_type_idx, x, y = rect
            # Check which side of the line the rectangle center is on
            side = a * x + b * y + c
            if side <= 0:  # Note the <= instead of > for diversity
                child1.append(rect)
            else:
                child2.append(rect)

        # Verify and repair children
        child1 = self.repair_solution(child1)
        child2 = self.repair_solution(child2)

        return child1, child2

    def repair_solution(self, solution: Solution) -> Solution:
        """Repair an invalid solution by removing conflicting rectangles."""
        if not solution:
            return solution

        if self.is_valid_solution(tuple(solution)):
            return solution

        # Sort rectangles by value (highest first) to prioritize keeping valuable rectangles
        sorted_rects = sorted(
            [(i, rect) for i, rect in enumerate(solution)],
            key=lambda item: self.rectangle_types[item[1][0]][2],
            reverse=True
        )

        valid_solution = []
        for _, rect in sorted_rects:
            # Try adding this rectangle to our valid solution
            candidate = valid_solution + [rect]
            if self.is_valid_solution(tuple(candidate)):
                valid_solution = candidate

        return valid_solution

    def mutate(self, solution: Solution) -> Solution:
        """Apply mutation to a solution with adaptive strategies."""
        if not solution:
            return solution

        mutated = copy.deepcopy(solution)

        # Calculate filled ratio to inform mutation strategies
        filled_ratio = self.calculate_filled_ratio(mutated)

        # 1. Rectangle addition mutation - adaptive probability based on fill ratio
        addition_prob = self.mutation_rate * (1.0 - filled_ratio * 0.5)  # Less likely to add when full
        if random.random() < addition_prob:
            # Select rectangle type based on current solution state
            rect_type_idx = self.select_rectangle_type_for_addition(mutated)
            width, height, _ = self.rectangle_types[rect_type_idx]

            # Generate random position with adaptive radius
            max_r = self.circle_radius - max(width, height) / 2.0
            if max_r > 0:
                # More precise positioning when circle is fuller
                precision_factor = 1.0 + 2.0 * filled_ratio
                attempts = max(3, int(5 * precision_factor))

                best_candidate = None
                best_fitness = -1

                for _ in range(attempts):
                    r = random.uniform(0, max_r)
                    theta = random.uniform(0, 2 * np.pi)
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)

                    candidate = mutated + [(rect_type_idx, x, y)]
                    if self.is_valid_solution(tuple(candidate)):
                        fitness = self.calculate_fitness(candidate)
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_candidate = candidate

                if best_candidate:
                    mutated = best_candidate

        # 2. Rectangle removal mutation - more likely to remove when full
        removal_prob = self.mutation_rate * (0.5 + filled_ratio * 0.5)
        if random.random() < removal_prob and len(mutated) > 0:
            # Preferentially remove lower-value rectangles
            weights = []
            for rect_type_idx, _, _ in mutated:
                _, _, value = self.rectangle_types[rect_type_idx]
                # Inverse value gives higher weight to lower value rectangles
                weights.append(1.0 / (value + 0.1))

            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                probs = [w / total_weight for w in weights]
                idx_to_remove = random.choices(range(len(mutated)), probs, k=1)[0]
            else:
                idx_to_remove = random.randint(0, len(mutated) - 1)

            mutated = mutated[:idx_to_remove] + mutated[idx_to_remove+1:]

        # 3. Position mutation - adaptive movement magnitude
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                rect_type_idx, x, y = mutated[i]
                width, height, _ = self.rectangle_types[rect_type_idx]

                # Smaller movements when circle is more filled
                scale = 0.2 * (1 - filled_ratio) + 0.05 * filled_ratio

                # Try multiple positions and pick the best valid one
                attempts = 5
                best_candidate = None
                best_fitness = self.calculate_fitness(mutated)

                for _ in range(attempts):
                    dx = random.uniform(-scale, scale)
                    dy = random.uniform(-scale, scale)
                    new_x = x + dx
                    new_y = y + dy

                    candidate = mutated[:i] + [(rect_type_idx, new_x, new_y)] + mutated[i+1:]
                    if self.is_valid_solution(tuple(candidate)):
                        fitness = self.calculate_fitness(candidate)
                        if fitness >= best_fitness:
                            best_fitness = fitness
                            best_candidate = candidate

                if best_candidate:
                    mutated = best_candidate

        # 4. Rectangle type mutation
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                rect_type_idx, x, y = mutated[i]

                # Filter current rectangle from solution
                solution_without_current = mutated[:i] + mutated[i+1:]

                # Either select adaptively or choose randomly with small probability for exploration
                if random.random() < 0.8:
                    new_rect_type_idx = self.select_rectangle_type_for_addition(solution_without_current)
                else:
                    new_rect_type_idx = random.randint(0, len(self.rectangle_types) - 1)

                if new_rect_type_idx != rect_type_idx:
                    candidate = mutated[:i] + [(new_rect_type_idx, x, y)] + mutated[i+1:]
                    if self.is_valid_solution(tuple(candidate)):
                        mutated = candidate

        return mutated

    def calculate_population_diversity(self, population: List[Solution]) -> float:
        """
        Calculate population diversity using Jaccard similarity.
        Returns a value between 0 (all identical) and 1 (completely diverse).
        """
        if not population or len(population) <= 1:
            return 0.0

        # Convert each solution to a set of rectangle types for comparison
        solution_sets = []
        for solution in population:
            # Create tuple of (rect_type, approx_x, approx_y) to identify "similar" placements
            # We round positions to create a grid effect for similarity comparison
            grid_size = 0.1
            rect_set = set()
            for rect_type_idx, x, y in solution:
                x_grid = round(x / grid_size)
                y_grid = round(y / grid_size)
                rect_set.add((rect_type_idx, x_grid, y_grid))
            solution_sets.append(rect_set)

        # Calculate average Jaccard similarity
        total_similarity = 0.0
        comparisons = 0

        # Sample pairs for efficiency in large populations
        max_comparisons = 100
        if len(population) > 20:
            # Sample random solution pairs
            pairs = [(i, j) for i in range(len(population)) for j in range(i+1, len(population))]
            random.shuffle(pairs)
            pairs = pairs[:max_comparisons]
        else:
            # Use all pairs for small populations
            pairs = [(i, j) for i in range(len(population)) for j in range(i+1, len(population))]

        for i, j in pairs:
            set_i = solution_sets[i]
            set_j = solution_sets[j]

            # Calculate Jaccard similarity
            if not set_i and not set_j:
                similarity = 1.0  # Both empty sets are identical
            elif not set_i or not set_j:
                similarity = 0.0  # One empty, one non-empty
            else:
                intersection = len(set_i.intersection(set_j))
                union = len(set_i.union(set_j))
                similarity = intersection / union if union > 0 else 0.0

            total_similarity += similarity
            comparisons += 1

        # Convert similarity to diversity (1 - similarity)
        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
        diversity = 1.0 - avg_similarity

        return diversity

    def inject_diversity(self, population: List[Solution], elite_solutions: List[Solution]) -> List[Solution]:
        """
        Inject diversity into the population while preserving elite solutions.
        """
        # Keep elite solutions
        new_population = elite_solutions.copy()

        # Generate new random solutions for the rest
        replacements_needed = self.population_size - len(elite_solutions)
        for _ in range(replacements_needed):
            # 70% brand new solutions, 30% mutations of elites
            if random.random() < 0.7:
                new_solution = self.generate_single_specimen()
            else:
                # Mutate a random elite with high mutation rate
                elite = random.choice(elite_solutions)
                temp_mutation_rate = self.mutation_rate
                self.mutation_rate = 0.5  # Temporarily increase mutation rate
                new_solution = self.mutate(elite)
                self.mutation_rate = temp_mutation_rate  # Restore original mutation rate

            new_population.append(new_solution)

        return new_population

    def update_adaptive_parameters(self, generation: int, best_fitness: float):
        """
        Update mutation rate, crossover rate, and tournament size based on
        algorithm performance and current state.
        """
        # Track best fitness history
        self.best_fitness_history.append(best_fitness)

        # Only adjust after we have some history
        if len(self.best_fitness_history) > 5:
            # Calculate improvement rate
            recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-6]

            # Check for improvement
            if recent_improvement > 0.01 * self.best_fitness_history[-6]:
                self.improvement_streak += 1
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                self.improvement_streak = 0

            # Adjust mutation rate based on improvement trend
            if self.stagnation_counter > 10:
                # Increase mutation rate when progress stalls
                self.mutation_rate = min(0.5, self.mutation_rate * 1.2)
            elif self.improvement_streak > 5:
                # Decrease mutation rate during improvement phases
                self.mutation_rate = max(0.05, self.mutation_rate * 0.9)
            else:
                # Gradually return to default
                self.mutation_rate = 0.8 * self.mutation_rate + 0.2 * self.initial_mutation_rate

            # Tournament size adaptation - increase pressure over time
            progress_ratio = min(1.0, generation / (self.max_generations * 0.7))
            max_tournament = min(15, int(self.population_size / 4))
            self.tournament_size = min(
                max_tournament,
                max(3, int(self.initial_tournament_size + progress_ratio * (max_tournament - self.initial_tournament_size)))
            )

    def evolve(self):
        """Run the genetic algorithm for the specified number of generations."""
        best_solution = None
        best_fitness = 0

        # Calculate initial fitness scores
        fitness_scores = [self.calculate_fitness(solution) for solution in self.population]

        # Sort population by fitness (highest first)
        combined = list(zip(self.population, fitness_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        self.population, fitness_scores = zip(*combined)
        self.population = list(self.population)
        fitness_scores = list(fitness_scores)

        for generation in range(self.max_generations):
            # Calculate population diversity
            diversity = self.calculate_population_diversity(self.population)
            self.diversity_history.append(diversity)

            # Identify elite solutions
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            elite_solutions = [self.population[i] for i in elite_indices]

            # Check if diversity injection is needed
            if diversity < self.diversity_threshold and generation > 10:
                print(f"Generation {generation + 1}: Low diversity detected ({diversity:.4f}), injecting new solutions")
                self.population = self.inject_diversity(self.population, elite_solutions)
                # Recalculate fitness scores after diversity injection
                fitness_scores = [self.calculate_fitness(solution) for solution in self.population]
            else:
                # Create new population
                new_population = elite_solutions.copy()

                # Fill the rest of the population with crossover and mutation
                while len(new_population) < self.population_size:
                    # Select parents
                    parent1 = self.select_parent(fitness_scores)
                    parent2 = self.select_parent(fitness_scores)

                    # Apply crossover
                    child1, child2 = self.crossover(parent1, parent2)

                    # Apply mutation
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)

                    # Add children to new population
                    if self.is_valid_solution(tuple(child1)):
                        new_population.append(child1)
                    if len(new_population) < self.population_size and self.is_valid_solution(tuple(child2)):
                        new_population.append(child2)

                    # If both children were invalid, try adding a new random solution
                    if len(new_population) < self.population_size and not (
                            self.is_valid_solution(tuple(child1)) or self.is_valid_solution(tuple(child2))
                    ):
                        random_solution = self.generate_single_specimen()
                        if self.is_valid_solution(tuple(random_solution)):
                            new_population.append(random_solution)

                # Update population
                self.population = new_population

            # Calculate new fitness scores
            fitness_scores = [self.calculate_fitness(solution) for solution in self.population]

            # Update best solution
            current_best_idx = fitness_scores.index(max(fitness_scores))
            current_best_solution = self.population[current_best_idx]
            current_best_fitness = fitness_scores[current_best_idx]

            if current_best_fitness > best_fitness:
                best_solution = current_best_solution
                best_fitness = current_best_fitness
                print(f"Generation {generation + 1}: New best solution found! Value = {best_fitness:.2f}")

            # Update adaptive parameters
            self.update_adaptive_parameters(generation, current_best_fitness)

            # Print progress
            if (generation + 1) % 10 == 0 or generation == 0:
                fill_ratio = self.calculate_filled_ratio(self.population[current_best_idx])
                print(f"Generation {generation + 1}/{self.max_generations}: " +
                      f"Best fitness = {current_best_fitness:.2f}, Fill ratio = {fill_ratio:.4f}, " +
                      f"Diversity = {diversity:.4f}, Mutation rate = {self.mutation_rate:.4f}")

        # Return best found solution and its fitness
        return best_solution, best_fitness
    def visualize_solution(self, solution: Solution):
        """Visualize the solution."""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw unit circle
        circle = plt.Circle((0, 0), self.circle_radius, fill=False, color='blue', linewidth=2)
        ax.add_patch(circle)

        # Draw each rectangle
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.rectangle_types)))

        # First create a legend mapping rectangle types to colors
        rect_type_counts = {}
        for rect_type_idx, _, _ in solution:
            rect_type_counts[rect_type_idx] = rect_type_counts.get(rect_type_idx, 0) + 1

        # Sort by count for the legend
        legend_elements = []
        for rect_type_idx, count in sorted(rect_type_counts.items(), key=lambda x: x[1], reverse=True):
            width, height, value = self.rectangle_types[rect_type_idx]
            color = colors[rect_type_idx % len(colors)]
            legend_elements.append(
                patches.Patch(
                    facecolor=color,
                    edgecolor='black',
                    alpha=0.7,
                    label=f"Type {rect_type_idx}: {width:.2f}×{height:.2f}, v={value} (×{count})"
                )
            )

        # Now draw the rectangles
        for rect_type_idx, x, y in solution:
            width, height, value = self.rectangle_types[rect_type_idx]
            color = colors[rect_type_idx % len(colors)]

            # Create rectangle
            rect = patches.Rectangle(
                (x - width/2, y - height/2),  # lower left corner
                width,  # width
                height,  # height
                linewidth=1,
                edgecolor='black',
                facecolor=color,
                alpha=0.7
            )
            ax.add_patch(rect)

            # Add text with rectangle value
            ax.text(x, y, f"{value:.1f}", ha='center', va='center', fontweight='bold',
                    fontsize=8, color='black')

        # Set limits and aspect ratio
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')

        # Add title with total value and filled ratio
        total_value = sum(self.rectangle_types[idx][2] for idx, _, _ in solution)
        filled_ratio = self.calculate_filled_ratio(solution) * 100  # Convert to percentage
        rectangle_count = len(solution)

        ax.set_title(f'Rectangle Packing Solution\n'
                     f'Total Value: {total_value:.2f} | Fill Ratio: {filled_ratio:.2f}% | Rectangles: {rectangle_count}',
                     fontsize=12, fontweight='bold')

        # Add legend
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)

        # Show grid and axes
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        # Add text with algorithm parameters
        param_text = (
            f"Parameters:\n"
            f"Population: {self.population_size}\n"
            f"Mutation rate: {self.mutation_rate}\n"
            f"Crossover rate: {self.crossover_rate}\n"
            f"Elite size: {self.elite_size}"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.5, param_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=props)

        plt.tight_layout()
        plt.show()

        # Return the figure for optional saving
        return fig, ax



