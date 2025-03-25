import numpy as np


class ArrayOfMatrices():
    def __init__(self, shapes, consecutive_list_arg=None, vector=False, count_first=False):
        if consecutive_list_arg:
            shapes = []
            for i in range(not count_first, len(consecutive_list_arg)):
                if vector:
                    shapes.append((consecutive_list_arg[i],))
                else:
                    shapes.append((consecutive_list_arg[i], consecutive_list_arg[i - 1]))
        array = [np.zeros(shape, dtype=float) for shape in shapes]
        self.shapes = shapes
        self.array = array

    def randomise_normal(self, mu=0, sigma=1):
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




class MLP():

    # def identity(x):
    #     return x
    #
    # def derivative_sv(function, x, dx=10 ** (-9)):
    #     return (function(x + dx) - function(x)) / dx

    def __init__(self, shape, activations):  # len(activations)+1=len(shape)
        fed_values = ArrayOfMatrices(None, shape, True, True)
        activation_values = ArrayOfMatrices(None, shape, True, True)
        self.fed_values = fed_values
        self.activation_values = activation_values

        weights = ArrayOfMatrices(None, shape)
        weights.randomise_normal(0, 1)
        biases = ArrayOfMatrices(None, shape, True)
        biases.randomise_normal(0, 1)
        self.weights = weights
        self.biases = biases
        self.activations = []
        self.derivatives = []
        self.weight_momentum = 0
        self.bias_momentum = 0
        self.weight_sq_mean = 0
        self.bias_sq_mean = 0
        for activation in activations:
            self.activations.append(np.vectorize(activation.function))
            self.derivatives.append(np.vectorize(activation.derivative))

    def set_input(self, inputt):
        self.fed_values.array[0] = np.array(inputt)
        self.activation_values.array[0] = np.array(inputt)  # input is input
        return True

    def feed_forward(self):
        for i in range(1, len(self.fed_values.array)):
            self.fed_values.array[i] = np.dot(self.weights.array[i - 1], self.activation_values.array[i - 1]) + self.biases.array[i - 1]
            self.activation_values.array[i] = self.activations[i - 1](self.fed_values.array[i])

    def predict(self, x):
        self.set_input(x)
        self.feed_forward()
        return self.activation_values.array[-1]

    def predict_multiple(self, x):
        y = []
        for el in x:
            y.append(self.predict(el))
        return np.array(y)

    def set_weights(self, weights):
        self.weights.array = weights

    def set_biases(self, biases):
        self.biases.array = biases

    def squared_error(pred, expected):
        return (expected - pred) ** 2

    # returns a pair first element is for weights second for biases
    def backpropagate(self, inputt, expected):
        self.predict(inputt)
        weight_grad = ArrayOfMatrices(self.weights.shapes)
        bias_grad = ArrayOfMatrices(self.biases.shapes)
        neuron_activation_grad = ArrayOfMatrices(self.fed_values.shapes)
        neuron_fed_grad = ArrayOfMatrices(self.fed_values.shapes)
        last_layer = True
        for i in range(len(self.fed_values.array) - 1, 0, -1):
            # derivatives with respect to neuron activation values
            if last_layer:
                x_0 = self.fed_values.array[i]
                y_0 = expected
                neuron_activation_grad.array[i] = 2 * (x_0 - y_0)
            else:
                neuron_activation_grad.array[i] = np.dot(self.weights.array[i].T, neuron_fed_grad.array[i+1])
            activation_func_deriv = self.derivatives[i-1](self.fed_values.array[i])
            neuron_fed_grad.array[i] = activation_func_deriv * neuron_activation_grad.array[i]
            bias_grad.array[i-1] = neuron_fed_grad.array[i]
            weight_grad.array[i-1] = np.outer(neuron_fed_grad.array[i], self.activation_values.array[i-1])
            last_layer = False
        return (weight_grad, bias_grad)

    def learn_batch(self, batchx, batchy):
        batch_size = len(batchx)
        avg_weight_gradient = 0
        avg_bias_gradient = 0
        for i in range(batch_size):
            local_weight_gradient, local_bias_gradient = self.backpropagate(batchx[i], batchy[i])
            avg_weight_gradient += local_weight_gradient
            avg_bias_gradient += local_bias_gradient
        avg_weight_gradient /= batch_size
        avg_bias_gradient /= batch_size
        return (avg_weight_gradient, avg_bias_gradient)

    def epoch(self, x, y, lr=0.01, momentum_coef = 0, batch_size=None):
        permutation = np.random.permutation(len(x))
        x_sh = x[permutation]
        y_sh = y[permutation]
        if not batch_size:
            batch_size = min(32, len(x))
        idx = 0
        while (idx + batch_size < len(x)):
            batchx = x_sh[idx:idx + batch_size]
            batchy = y_sh[idx:idx + batch_size]
            weight_step, bias_step = self.learn_batch(batchx, batchy)
            self.weight_momentum = self.weight_momentum * momentum_coef + weight_step * (1 - momentum_coef)
            self.bias_momentum = self.bias_momentum * momentum_coef + bias_step * (1 - momentum_coef)
            self.weights -= self.weight_momentum * lr
            self.biases -= self.bias_momentum * lr

            idx += batch_size
        if idx != len(x) - 1:
            weight_step, bias_step = self.learn_batch(batchx, batchy)
            self.weight_momentum = self.weight_momentum * momentum_coef + weight_step * (1 - momentum_coef)
            self.bias_momentum = self.bias_momentum * momentum_coef + bias_step * (1 - momentum_coef)
            self.weights -= self.weight_momentum * lr
            self.biases -= self.bias_momentum * lr

    def rms_epoch(self, x, y, lr=0.01, rms_coef = 0.9, batch_size=None):
        eps = 10**(-2)
        permutation = np.random.permutation(len(x))
        x_sh = x[permutation]
        y_sh = y[permutation]
        if not batch_size:
            batch_size = min(32, len(x))
        idx = 0
        while (idx + batch_size < len(x)):
            batchx = x_sh[idx:idx + batch_size]
            batchy = y_sh[idx:idx + batch_size]
            weight_step, bias_step = self.learn_batch(batchx, batchy)
            self.weight_sq_mean = rms_coef * self.weight_sq_mean + weight_step**2 * (1 - rms_coef)
            self.bias_sq_mean = rms_coef * self.bias_sq_mean + bias_step**2 * (1 - rms_coef)
            self.weights -= lr * weight_step / (self.weight_sq_mean + eps) ** (0.5)
            self.biases -= lr * bias_step / (self.bias_sq_mean + eps) ** (0.5)
            idx += batch_size
        if idx != len(x) - 1:
            weight_step, bias_step = self.learn_batch(batchx, batchy)
            self.weight_sq_mean = rms_coef * self.weight_sq_mean + weight_step**2 * (1 - rms_coef)
            self.bias_sq_mean = rms_coef * self.bias_sq_mean + bias_step**2 * (1 - rms_coef)
            self.weights -= lr * weight_step / (self.weight_sq_mean + eps) ** (0.5)
            self.biases -= lr * bias_step / (self.bias_sq_mean + eps) ** (0.5)



def main():
    def identity_func(x):
        return x
    def one_function(x):
        return 1
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    def sigmoid_derivative(x):
        return np.exp(-x)/(1+np.exp(-x))**2
    def relu_func(x):
        return np.maximum(0,x)
    def relu_derivative(x):
        return 0 if x < 0 else 1
    def soft_relu_func(x):
        return x*10**(-3) if x < 0 else x
    def soft_relu_derivative(x):
        return 10**(-3) if x < 0 else 1
    identity = FunctionWithDerivative(identity_func, one_function)
    sigmoid = FunctionWithDerivative(sigmoid, sigmoid_derivative)
    relu = FunctionWithDerivative(relu_func, relu_derivative)
    soft_relu = FunctionWithDerivative(soft_relu_func, soft_relu_derivative)
    network1 = MLP([1,5,1], [sigmoid, identity])
    network2 = MLP([1,5,1], [sigmoid, identity])
    weights1 = network1.weights
    weights2 = network2.weights
    print(weights1)
    print(weights2)
    print((weights1**2)**(1/2))

class FunctionWithDerivative():
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

if __name__ == "__main__":
    main()












