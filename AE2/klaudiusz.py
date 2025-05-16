import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from typing import List, Tuple, Dict
import copy

# Define rectangle type as (width, height, value)
RectangleType = Tuple[float, float, float]

# Define solution as list of (rectangle_type_index, center_x, center_y)
Solution = List[Tuple[int, float, float]]

class CircleRectanglePacker:
    def __init__(self, rectangle_types: List[RectangleType],
                 population_size: int = 100,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7,
                 elite_size: int = 5,
                 tournament_size: int = 5,
                 max_generations: int = 100):
        """
        Initialize the genetic algorithm for rectangle packing in a unit circle.

        Args:
            rectangle_types: List of available rectangle types as (width, height, value)
            population_size: Size of the population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of best solutions to carry to next generation
            tournament_size: Size of tournament for selection
            max_generations: Maximum number of generations
        """
        self.rectangle_types = rectangle_types
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.circle_radius = 1.0  # Unit circle

        # Initialize population
        self.population = self.generate_initial_population()

    def generate_initial_population(self) -> List[Solution]:
        """Generate initial random population."""
        population = []
        for _ in range(self.population_size):
            # Start with an empty solution
            solution = []

            # Try to add random rectangles until we fail multiple times
            consecutive_failures = 0
            max_failures = 10

            while consecutive_failures < max_failures:
                # Pick a random rectangle type
                rect_type_idx = random.randint(0, len(self.rectangle_types) - 1)
                width, height, _ = self.rectangle_types[rect_type_idx]

                # Generate a random position that keeps the rectangle mostly inside the circle
                # Note: we allow some overlap with the circle boundary for exploration
                max_r = self.circle_radius - max(width, height) / 2.0
                if max_r <= 0:
                    # This rectangle is too big for the circle
                    consecutive_failures += 1
                    continue

                # Use polar coordinates for more uniform distribution
                r = random.uniform(0, max_r + 0.1)  # Allow slight circle boundary overlap
                theta = random.uniform(0, 2 * np.pi)

                # Convert to Cartesian coordinates
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                # Create a candidate solution by adding this rectangle
                candidate = solution + [(rect_type_idx, x, y)]

                # Check if candidate is valid (no overlaps)
                if self.is_valid_solution(candidate):
                    solution = candidate
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

            population.append(solution)

        return population

    def is_valid_solution(self, solution: Solution) -> bool:
        """Check if a solution is valid (all rectangles within circle and no overlaps)."""
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
                rect_type_idx2, x2, y2 = solution[j]
                width2, height2, _ = self.rectangle_types[rect_type_idx2]

                # Calculate rectangle boundaries
                left1, right1 = x1 - width1/2, x1 + width1/2
                bottom1, top1 = y1 - height1/2, y1 + height1/2

                left2, right2 = x2 - width2/2, x2 + width2/2
                bottom2, top2 = y2 - height2/2, y2 + height2/2

                # Check for overlap
                if not (right1 <= left2 or right2 <= left1 or top1 <= bottom2 or top2 <= bottom1):
                    return False  # Overlap detected

        return True

    def calculate_fitness(self, solution: Solution) -> float:
        """Calculate fitness (total value) of a solution."""
        if not self.is_valid_solution(solution):
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
        """Perform crossover between two parents."""
        if random.random() > self.crossover_rate or len(parent1) == 0 or len(parent2) == 0:
            return parent1, parent2  # No crossover

        # Determine crossover points
        point1 = random.randint(0, len(parent1))
        point2 = random.randint(0, len(parent2))

        # Create offspring
        child1 = parent1[:point1] + parent2[point2:]
        child2 = parent2[:point2] + parent1[point1:]

        return child1, child2

    def mutate(self, solution: Solution) -> Solution:
        """Apply mutation to a solution."""
        if not solution:
            return solution

        mutated = copy.deepcopy(solution)

        # 1. Rectangle addition mutation
        if random.random() < self.mutation_rate:
            rect_type_idx = random.randint(0, len(self.rectangle_types) - 1)
            width, height, _ = self.rectangle_types[rect_type_idx]

            # Generate random position
            max_r = self.circle_radius - max(width, height) / 2.0
            if max_r > 0:
                r = random.uniform(0, max_r)
                theta = random.uniform(0, 2 * np.pi)
                x = r * np.cos(theta)
                y = r * np.sin(theta)

                candidate = mutated + [(rect_type_idx, x, y)]
                if self.is_valid_solution(candidate):
                    mutated = candidate

        # 2. Rectangle removal mutation
        if random.random() < self.mutation_rate and len(mutated) > 0:
            idx_to_remove = random.randint(0, len(mutated) - 1)
            mutated = mutated[:idx_to_remove] + mutated[idx_to_remove+1:]

        # 3. Position mutation
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                rect_type_idx, x, y = mutated[i]
                width, height, _ = self.rectangle_types[rect_type_idx]

                # Small displacement
                dx = random.uniform(-0.1, 0.1)
                dy = random.uniform(-0.1, 0.1)

                new_x = x + dx
                new_y = y + dy

                # Create candidate with new position
                candidate = mutated[:i] + [(rect_type_idx, new_x, new_y)] + mutated[i+1:]
                if self.is_valid_solution(candidate):
                    mutated = candidate

        # 4. Rectangle type mutation
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                rect_type_idx, x, y = mutated[i]
                new_rect_type_idx = random.randint(0, len(self.rectangle_types) - 1)

                if new_rect_type_idx != rect_type_idx:
                    candidate = mutated[:i] + [(new_rect_type_idx, x, y)] + mutated[i+1:]
                    if self.is_valid_solution(candidate):
                        mutated = candidate

        return mutated

    def evolve(self):
        """Run the genetic algorithm for the specified number of generations."""
        best_solution = None
        best_fitness = 0

        # Calculate initial fitness scores
        fitness_scores = [self.calculate_fitness(solution) for solution in self.population]

        for generation in range(self.max_generations):
            # Identify elite solutions
            elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:self.elite_size]
            elite_solutions = [self.population[i] for i in elite_indices]

            # Create new population
            new_population = elite_solutions

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

                # Add valid children to new population
                if self.is_valid_solution(child1) and len(new_population) < self.population_size:
                    new_population.append(child1)
                if self.is_valid_solution(child2) and len(new_population) < self.population_size:
                    new_population.append(child2)

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

            # Print progress
            if (generation + 1) % 10 == 0:
                print(f"Generation {generation + 1}/{self.max_generations}: Best fitness = {best_fitness}")

        return best_solution, best_fitness

    def visualize_solution(self, solution: Solution):
        """Visualize the solution."""
        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw unit circle
        circle = plt.Circle((0, 0), self.circle_radius, fill=False, color='blue')
        ax.add_patch(circle)

        # Draw each rectangle
        for rect_type_idx, x, y in solution:
            width, height, value = self.rectangle_types[rect_type_idx]

            # Create rectangle
            rect = patches.Rectangle(
                (x - width/2, y - height/2),  # lower left corner
                width,  # width
                height,  # height
                linewidth=1,
                edgecolor='r',
                facecolor='none',
                alpha=0.7
            )
            ax.add_patch(rect)

            # Add text with rectangle value
            ax.text(x, y, f"{value:.1f}", ha='center', va='center')

        # Set limits and aspect ratio
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')

        # Add title with total value
        total_value = sum(self.rectangle_types[idx][2] for idx, _, _ in solution)
        ax.set_title(f'Total Value: {total_value:.2f}')

        # Show grid and axes
        ax.grid(True)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

        plt.show()

def main():
    # Define rectangle types: (width, height, value)
    rectangle_types = [
        (0.3125, 0.15, 100),
        (0.4, 0.2, 500),
        (0.3125, 0.2, 600),
        (0.1875, 0.15, 40),
        (0.5, 0.0375, 400),
        (0.15, 0.3125, 100),
        (0.2, 0.4, 500),
        (0.2, 0.3125, 600),
        (0.15, 0.1875, 40),
        (0.0375, 0.5, 400)
    ]

    # Create and run genetic algorithm
    packer = CircleRectanglePacker(
        rectangle_types=rectangle_types,
        population_size=300,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elite_size=30,
        tournament_size=5,
        max_generations=500
    )

    # Run the algorithm
    best_solution, best_fitness = packer.evolve()

    # Print results
    print(f"Best solution has total value: {best_fitness}")
    print(f"Number of rectangles used: {len(best_solution)}")

    # Visualize best solution
    packer.visualize_solution(best_solution)

if __name__ == "__main__":
    main()