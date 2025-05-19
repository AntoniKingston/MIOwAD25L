import numpy as np
from sklearn.metrics import log_loss
import copy
from typing import List, Tuple, Callable, Dict
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class ArrayOfMatrices():
    def __init__(self, shapes, consecutive_list_arg=None, vector=False, normal_init = False, count_first=False):
        if consecutive_list_arg:
            shapes = []
            for i in range(not count_first, len(consecutive_list_arg)):
                if vector:
                    shapes.append((consecutive_list_arg[i],))
                else:
                    shapes.append((consecutive_list_arg[i], consecutive_list_arg[i - 1]))
        array = [np.random.normal(0,1,shape) if normal_init else np.zeros(shape) for shape in shapes]
        self.shapes = shapes
        self.array = array

    def randomise_normal(self, mu=0, sigma=1.0):
        for i in range(len(self.array)):
            self.array[i] = np.random.normal(mu, sigma, self.array[i].shape)

    def randomise_uniform(self, a=0, b=1):
        for i in range(len(self.array)):
            self.array[i] = np.random.uniform(a, b, self.array[i].shape)




    def __add__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            if isinstance(other, ArrayOfMatrices):
                result.array[i] = self.array[i] + other.array[i]
            else:
                result.array[i] = self.array[i] + other
        return result

    def __radd__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = self.array[i] + other
        return result

    def __sub__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            if isinstance(other, ArrayOfMatrices):
                result.array[i] = self.array[i] - other.array[i]
            else:
                result.array[i] = self.array[i] - other
        return result

    def __rsub__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = other - self.array[i]
        return result

    def __mul__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            if isinstance(other, ArrayOfMatrices):
                result.array[i] = self.array[i] * other.array[i]
            else:
                result.array[i] = self.array[i] * other
        return result
    def __rmul__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = self.array[i] * other
        return result

    def __truediv__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            if isinstance(other, ArrayOfMatrices):
                result.array[i] = self.array[i] / other.array[i]
            else:
                result.array[i] = self.array[i] / other
        return result

    def __rtruediv__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = other / self.array[i]
        return result

    def __pow__(self, power, modulo=None):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = self.array[i] ** power
        return result
    def __str__(self):
        string = ""
        printables = self.array
        idx=1
        for el in printables:
            string += f"matrix {idx}.\n"
            string += str(el)
            string += "\n"
            idx+=1
        return string

class SimpleMLP():
    def __init__(self, shapes, loss, activation, normal_init = False):
        self.weights = ArrayOfMatrices(None, shapes, False, normal_init)
        self.biases = ArrayOfMatrices(None, shapes, True, normal_init)
        self.loss = loss
        self.activation = activation
        self.setup_activation_function()

    def setup_activation_function(self):
        if self.activation == "relu":
            def relu(x):
                return np.maximum(x, 0)
            self.activation_function = np.vectorize(relu)
        elif self.activation == "sigmoid":
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            self.activation_function = np.vectorize(sigmoid)
        elif self.activation == "tanh":
            def tanh(x):
                return np.tanh(x)
            self.activation_function = tanh
        else:
            self.activation_function = None
    def feed_forward(self, input):
        for i in range(len(self.weights.array)):
            input = np.dot(self.weights.array[i], input)
            input += self.biases.array[i]
            if i != len(self.weights.array) - 1 and self.activation_function:
                input = self.activation_function(input)

        return input

    def evaluate_loss_single(self, input, target):
        pred = self.feed_forward(input)
        if self.loss == "MSE":
            return np.sum((target - pred) ** 2)
        else:
            pred = softmax(pred)
            return log_loss([target], [pred])

    def evaluate_accuracy(self, inputs, targets):
        if self.loss == "MSE":
            raise ValueError("No accuracy metric for regression")
        preds = np.array([self.feed_forward(input) for input in inputs])
        preds = np.argmax(preds, axis=1)
        targets = np.argmax(targets, axis=1)
        return np.mean(targets == preds)

    def evaluate_loss(self, inputs, targets):
        total_loss = 0
        for x, y in zip(inputs, targets):
            total_loss += self.evaluate_loss_single(x, y)
        return total_loss / len(inputs)

def train_mlp_evolutionary(
        mlp: 'SimpleMLP',
        inputs: np.ndarray,
        targets: np.ndarray,
        inputs_test: np.ndarray = None,
        targets_test: np.ndarray = None,
        population_size: int = 50,
        generations: int = 100,
        tournament_size: int = 5,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.1,
        crossover_prob: float = 0.7,
        verbose: bool = True,
        verbose_interval: int = 10,
        dynamic_hyperparams: bool = True,
        stagnation_threshold: int = 10,
        improvement_threshold: float = 0.01
) -> Tuple['SimpleMLP', List[float], List[float], Dict[str, float]]:
    """
    Train a SimpleMLP using an evolutionary algorithm with tournament selection,
    elitism, normal mutation, and single-point crossover for both weights and biases.

    Args:
        mlp: The initial SimpleMLP model
        inputs: Training inputs
        targets: Training targets
        inputs_test: Optional test inputs for validation
        targets_test: Optional test targets for validation
        population_size: Size of the population
        generations: Number of generations to evolve
        tournament_size: Number of individuals in each tournament
        elite_size: Number of top individuals to preserve unchanged
        mutation_rate: Probability of mutating each weight/bias matrix
        mutation_scale: Scale factor for normal distribution during mutation
        crossover_prob: Probability of crossover between two parents
        verbose: Whether to print progress
        verbose_interval: How often to print progress (every X generations)
        dynamic_hyperparams: Whether to dynamically adjust hyperparameters during training
        stagnation_threshold: Number of generations without improvement to consider stagnation
        improvement_threshold: Percentage improvement threshold to consider significant progress

    Returns:
        Tuple of (best_model, history_of_best_train_losses, history_of_best_test_losses)
    """
    # Initialize population with variations of the initial MLP
    population = [copy.deepcopy(mlp) for _ in range(population_size)]
    print(mlp.loss)
    # Randomize the initial population except for the first individual
    for i in range(1, population_size):
        population[i].weights.randomise_normal(0, 0.1)
        population[i].biases.randomise_normal(0, 0.1)

    best_train_losses = []
    best_test_losses = []


    # Variables for dynamic hyperparameter adjustment
    stagnation_counter = 0
    best_loss_so_far = float('inf')
    current_mutation_rate = mutation_rate
    current_mutation_scale = mutation_scale
    current_elite_size = elite_size

    # Track hyperparameter history
    mutation_rate_history = []
    mutation_scale_history = []
    elite_size_history = []

    for generation in range(generations):
        # Evaluate fitness for each individual
        fitness_scores = [
            -individual.evaluate_loss(inputs, targets) for individual in population
        ]

        # Higher fitness is better (negative loss)
        sorted_indices = np.argsort(fitness_scores)[::-1]

        # Store the best individual's performance
        best_loss = -fitness_scores[sorted_indices[0]]
        best_train_losses.append(best_loss)

        # Track current hyperparameters
        mutation_rate_history.append(current_mutation_rate)
        mutation_scale_history.append(current_mutation_scale)
        elite_size_history.append(current_elite_size)

        # If test data is provided, evaluate on test set
        if inputs_test is not None and targets_test is not None:
            best_test_loss = population[sorted_indices[0]].evaluate_loss(inputs_test, targets_test)
            best_test_losses.append(best_test_loss)

        # Dynamic hyperparameter adjustment
        if dynamic_hyperparams and generation > 0:
            # Calculate relative improvement
            relative_improvement = (best_loss_so_far - best_loss) / best_loss_so_far if best_loss_so_far > 0 else 0

            # Check for stagnation or significant improvement
            if best_loss < best_loss_so_far - improvement_threshold * best_loss_so_far:
                # Significant improvement - reduce mutation parameters and increase elitism
                stagnation_counter = 0
                best_loss_so_far = best_loss

                # Gradually decrease mutation as we converge to good solutions
                current_mutation_rate = max(0.01, current_mutation_rate * 0.95)
                current_mutation_scale = max(0.001, current_mutation_scale * 0.95)

                # Increase elitism to preserve good solutions
                current_elite_size = min(int(population_size * 0.15), current_elite_size + 1)

            elif best_loss >= best_loss_so_far:
                # No improvement - increase stagnation counter
                stagnation_counter += 1

                # If stagnation persists, increase mutation to explore more
                if stagnation_counter >= stagnation_threshold:
                    current_mutation_rate = min(0.9, current_mutation_rate * 1.5)
                    current_mutation_scale = min(0.5, current_mutation_scale * 1.5)

                    # Decrease elitism to allow more exploration
                    current_elite_size = max(1, current_elite_size - 1)

                    stagnation_counter = 0  # Reset counter after adjustment
            else:
                # Minor improvement - update best loss but don't change parameters
                best_loss_so_far = best_loss
        else:
            # Without dynamic parameters, just track the best loss
            if best_loss < best_loss_so_far:
                best_loss_so_far = best_loss

        if verbose and (generation % verbose_interval == 0 or generation == generations - 1):
            print(f"Generation {generation}: Best train loss = {best_loss:.6f}", end="")
            if inputs_test is not None and targets_test is not None:
                print(f", Best test loss = {best_test_losses[-1]:.6f}", end="")
            if dynamic_hyperparams:
                print(f" | mutation_rate={current_mutation_rate:.4f}, mutation_scale={current_mutation_scale:.4f}, elite_size={current_elite_size}")
            if mlp.loss != "MSE":
                print(f" | Best train accuracy = {population[sorted_indices[0]].evaluate_accuracy(inputs, targets):.6f}")
                print(f" | Best test accuracy = {population[sorted_indices[0]].evaluate_accuracy(inputs_test, targets_test):.6f}")
            else:
                print()

        # Early stopping if we've reached the last generation
        if generation == generations - 1:
            break

        # Create the next generation
        next_generation = []

        # Elitism: Keep the best individuals
        for i in range(int(current_elite_size)):
            if i < len(sorted_indices):  # Safety check
                next_generation.append(copy.deepcopy(population[sorted_indices[i]]))

        # Fill the rest of the population through selection, crossover, and mutation
        while len(next_generation) < population_size:
            # Tournament selection for parent 1
            parent1_idx = tournament_selection(fitness_scores, tournament_size)
            parent1 = population[parent1_idx]

            # Tournament selection for parent 2
            parent2_idx = tournament_selection(fitness_scores, tournament_size)
            parent2 = population[parent2_idx]

            # Perform crossover
            if np.random.random() < crossover_prob:
                child1, child2 = single_point_crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # Perform mutation
            mutate(child1, current_mutation_rate, current_mutation_scale)
            mutate(child2, current_mutation_rate, current_mutation_scale)

            # Add children to the next generation
            next_generation.append(child1)
            if len(next_generation) < population_size:
                next_generation.append(child2)

        # Replace the old population
        population = next_generation

    # Return the best individual from the final generation along with histories
    final_fitness_scores = [
        -individual.evaluate_loss(inputs, targets) for individual in population
    ]
    best_idx = np.argmax(final_fitness_scores)

    # For analysis, you might want the hyperparameter histories
    hyperparam_histories = {
        'mutation_rate_history': mutation_rate_history,
        'mutation_scale_history': mutation_scale_history,
        'elite_size_history': elite_size_history
    }

    # Add hyperparameter histories as the fourth return value
    if inputs_test is not None and targets_test is not None:
        return population[best_idx], best_train_losses, best_test_losses, hyperparam_histories
    else:
        return population[best_idx], best_train_losses, [], hyperparam_histories


def tournament_selection(fitness_scores: List[float], tournament_size: int) -> int:
    """
    Select an individual using tournament selection.

    Args:
        fitness_scores: List of fitness scores
        tournament_size: Number of individuals in the tournament

    Returns:
        Index of the selected individual
    """
    # Randomly select tournament_size individuals
    tournament_indices = np.random.choice(
        len(fitness_scores), tournament_size, replace=False
    )

    # Select the best individual from the tournament
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_relative_idx = np.argmax(tournament_fitness)

    return tournament_indices[winner_relative_idx]


def single_point_crossover(
        parent1: 'SimpleMLP', parent2: 'SimpleMLP'
) -> Tuple['SimpleMLP', 'SimpleMLP']:
    """
    Perform single-point crossover on weights and biases of two MLPs.

    Args:
        parent1: First parent MLP
        parent2: Second parent MLP

    Returns:
        Two child MLPs resulting from crossover
    """
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    # Crossover for weights
    for i in range(len(parent1.weights.array)):
        # Get the shape of the current weight matrix
        shape = parent1.weights.array[i].shape

        # Flatten the matrices
        flat1 = parent1.weights.array[i].flatten()
        flat2 = parent2.weights.array[i].flatten()

        # Choose a crossover point (ensure valid even for small arrays)
        if len(flat1) > 1:
            crossover_point = np.random.randint(1, len(flat1))

            # Create new flattened arrays with crossover
            new_flat1 = np.concatenate([flat1[:crossover_point], flat2[crossover_point:]])
            new_flat2 = np.concatenate([flat2[:crossover_point], flat1[crossover_point:]])

            # Reshape back
            child1.weights.array[i] = new_flat1.reshape(shape)
            child2.weights.array[i] = new_flat2.reshape(shape)

    # Crossover for biases
    for i in range(len(parent1.biases.array)):
        # Get the shape of the current bias vector
        shape = parent1.biases.array[i].shape

        # Flatten the vectors
        flat1 = parent1.biases.array[i].flatten()
        flat2 = parent2.biases.array[i].flatten()

        # Choose a crossover point (ensure valid even for small arrays)
        if len(flat1) > 1:
            crossover_point = np.random.randint(1, len(flat1))

            # Create new flattened arrays with crossover
            new_flat1 = np.concatenate([flat1[:crossover_point], flat2[crossover_point:]])
            new_flat2 = np.concatenate([flat2[:crossover_point], flat1[crossover_point:]])

            # Reshape back
            child1.biases.array[i] = new_flat1.reshape(shape)
            child2.biases.array[i] = new_flat2.reshape(shape)

    return child1, child2


def mutate(mlp: 'SimpleMLP', mutation_rate: float, mutation_scale: float) -> None:
    """
    Apply normal mutation to weights and biases of an MLP.

    Args:
        mlp: The MLP to mutate
        mutation_rate: Probability of mutating each weight/bias matrix
        mutation_scale: Scale of the normal distribution for mutation
    """
    # Mutate weights
    for i in range(len(mlp.weights.array)):
        if np.random.random() < mutation_rate:
            # Apply random normal noise scaled by mutation_scale
            noise = np.random.normal(0, mutation_scale, mlp.weights.array[i].shape)
            mlp.weights.array[i] += noise

    # Mutate biases
    for i in range(len(mlp.biases.array)):
        if np.random.random() < mutation_rate:
            # Apply random normal noise scaled by mutation_scale
            noise = np.random.normal(0, mutation_scale, mlp.biases.array[i].shape)
            mlp.biases.array[i] += noise

def plot_training_progress(train_losses, test_losses=None, hyperparam_histories=None):
    """
    Plot the evolution progress including loss curves and hyperparameter changes.

    Args:
        train_losses: List of training losses per generation
        test_losses: Optional list of test losses per generation
        hyperparam_histories: Optional dictionary with hyperparameter histories
    """
    # Create a figure with specific size
    plt.figure(figsize=(12, 18))

    # Plot loss curves
    plt.subplot(3, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    if test_losses and len(test_losses) > 0:
        plt.plot(test_losses, label='Test Loss')
    plt.title('Evolutionary Training Progress')
    plt.xlabel('Generation')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # If hyperparameter histories are provided, plot them
    if hyperparam_histories:
        plt.subplot(3, 1, 2)

        if 'mutation_rate_history' in hyperparam_histories:
            plt.plot(hyperparam_histories['mutation_rate_history'],
                     label='Mutation Rate', color='blue')

        if 'mutation_scale_history' in hyperparam_histories:
            plt.plot(hyperparam_histories['mutation_scale_history'],
                     label='Mutation Scale', color='green')
        plt.title('Hyperparameter Adaptation')
        plt.xlabel('Generation')
        plt.ylabel('Parameter Value')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3)
        # Elite size is discrete, so use a different plot style
        if 'elite_size_history' in hyperparam_histories:
            plt.plot(hyperparam_histories['elite_size_history'],
                     label='Elite Size', color='red', linestyle='-')
        plt.title('Hyperparameter Adaptation')
        plt.xlabel('Generation')
        plt.ylabel('Parameter Value')
        plt.grid(True)
        plt.legend()



    plt.tight_layout()
    plt.show()