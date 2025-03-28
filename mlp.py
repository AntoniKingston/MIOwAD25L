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
            self.activations.append(FunctionWithDerivative(activation))

    def set_input(self, inputt):
        self.fed_values.array[0] = np.array(inputt)
        self.activation_values.array[0] = np.array(inputt)  # input is input
        return True


    def feed_forward(self):
        for i in range(1, len(self.fed_values.array)):
            self.fed_values.array[i] = np.dot(self.weights.array[i - 1], self.activation_values.array[i - 1]) + self.biases.array[i - 1]
            self.activation_values.array[i] = self.activations[i - 1].function(self.fed_values.array[i])

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


    # returns a pair first element is for weights second for biases
    def backpropagate(self, inputt, expected, metric):
        self.predict(inputt)
        weight_grad = ArrayOfMatrices(self.weights.shapes)
        bias_grad = ArrayOfMatrices(self.biases.shapes)
        neuron_activation_grad = ArrayOfMatrices(self.fed_values.shapes)
        neuron_fed_grad = ArrayOfMatrices(self.fed_values.shapes)
        last_layer = True
        for i in range(len(self.fed_values.array) - 1, 0, -1):
            # derivatives with respect to neuron activation values
            if last_layer:
                y_0 = expected
                #metrics names are expected to come from set: {"MSE", "CE"}
                if metric == "MSE":
                    if self.activations[i-1].name == "softmax":
                        x_0 = self.activation_values.array[i]
                        jacobian = self.activations[i-1].derivative(x_0)
                        MSE_grad = 2 * (x_0 - y_0)
                        neuron_fed_grad.array[i] = np.dot(jacobian, MSE_grad)
                    else:
                        x_0 = self.fed_values.array[i]
                        neuron_activation_grad.array[i] = 2 * (x_0 - y_0)
                elif metric == "CE":
                    if self.activations[i-1].name == "softmax":
                        x_0 = self.activation_values.array[i]
                        neuron_fed_grad.array[i] = x_0 - y_0
                    else:
                        raise ValueError("Cross entropy works on probability distributions.")
                else:
                    raise ValueError(f"Unknown metric {metric.name}")


                # if metric == "MSE":
                #     x_0 = self.fed_values.array[i]
                #     neuron_activation_grad.array[i] = 2 * (x_0 - y_0)
                # elif metric == "CE":
                #     x_0 = self.activation_values.array[i]
                #     neuron_fed_grad.array[i] = x_0 - y_0
                # else:
                #     raise ValueError("Metric not supported")
            else:
                neuron_activation_grad.array[i] = np.dot(self.weights.array[i].T, neuron_fed_grad.array[i+1])
            if not last_layer or self.activations[i-1].name == "identity" :
                activation_func_deriv = self.activations[i-1].derivative(self.activation_values.array[i])
                neuron_fed_grad.array[i] = activation_func_deriv * neuron_activation_grad.array[i]
            bias_grad.array[i-1] = neuron_fed_grad.array[i]
            weight_grad.array[i-1] = np.outer(neuron_fed_grad.array[i], self.activation_values.array[i-1])
            last_layer = False
        return (weight_grad, bias_grad)

    def learn_batch(self, batchx, batchy, metric):
        batch_size = len(batchx)
        avg_weight_gradient = 0
        avg_bias_gradient = 0
        for i in range(batch_size):
            local_weight_gradient, local_bias_gradient = self.backpropagate(batchx[i], batchy[i], metric)
            avg_weight_gradient += local_weight_gradient
            avg_bias_gradient += local_bias_gradient
        avg_weight_gradient /= batch_size
        avg_bias_gradient /= batch_size
        return (avg_weight_gradient, avg_bias_gradient)

    def epoch(self, x, y, metric, lr=0.01, momentum_coef = 0, batch_size=None):
        permutation = np.random.permutation(len(x))
        x_sh = x[permutation]
        y_sh = y[permutation]
        if not batch_size:
            batch_size = min(32, len(x))
        idx = 0
        while (idx + batch_size < len(x)):
            batchx = x_sh[idx:idx + batch_size]
            batchy = y_sh[idx:idx + batch_size]
            weight_step, bias_step = self.learn_batch(batchx, batchy, metric)
            self.weight_momentum = self.weight_momentum * momentum_coef + weight_step * (1 - momentum_coef)
            self.bias_momentum = self.bias_momentum * momentum_coef + bias_step * (1 - momentum_coef)
            self.weights -= self.weight_momentum * lr
            self.biases -= self.bias_momentum * lr

            idx += batch_size
        if idx != len(x) - 1:
            weight_step, bias_step = self.learn_batch(batchx, batchy, metric)
            self.weight_momentum = self.weight_momentum * momentum_coef + weight_step * (1 - momentum_coef)
            self.bias_momentum = self.bias_momentum * momentum_coef + bias_step * (1 - momentum_coef)
            self.weights -= self.weight_momentum * lr
            self.biases -= self.bias_momentum * lr

    def rms_epoch(self, x, y, metric, lr=0.01, rms_coef = 0.9, batch_size=None):
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
            weight_step, bias_step = self.learn_batch(batchx, batchy, metric)
            self.weight_sq_mean = rms_coef * self.weight_sq_mean + weight_step**2 * (1 - rms_coef)
            self.bias_sq_mean = rms_coef * self.bias_sq_mean + bias_step**2 * (1 - rms_coef)
            self.weights -= lr * weight_step / (self.weight_sq_mean + eps) ** (0.5)
            self.biases -= lr * bias_step / (self.bias_sq_mean + eps) ** (0.5)
            idx += batch_size
        if idx != len(x) - 1:
            weight_step, bias_step = self.learn_batch(batchx, batchy, metric)
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
    #every function needs to be numpy_vectorised derivative might be a vector (if jacobian is diagonal, or may be asquare matrix if it's not)
    def __init__(self, name):
        self.name = name
        def identity_init(self):
            def indentity(x):
                return x
            self.function = np.vectorize(indentity)
            def one(x):
                return np.ones_like(x)
            self.derivative = one
        def relu_init(self):
            def relu(x):
                return np.maximum(0,x)
            self.function = np.vectorize(relu)
            def relu_derivative(x):
                return int(x>=0)
            self.derivative = np.vectorize(relu_derivative)
        def soft_relu_init(self):
            def soft_relu(x):
                return x if x>=0 else 0.01 * x
            self.function = np.vectorize(soft_relu)
            def soft_relu_derivative(x):
                return 1 if x >= 0 else 0.01
            self.derivative = np.vectorize(soft_relu_derivative)
        def sigmoid_init(self):
            def sigmoid(x):
                return 1/(1+np.exp(-x))
            self.function = np.vectorize(sigmoid)
            def sigmoid_derivative(x):
                return x*(1-x)
            self.derivative = np.vectorize(sigmoid_derivative)
        def softmax_init(self):
            def softmax(x):
                return np.exp(x) / np.sum(np.exp(x))
            self.function = softmax

            def softmax_jacobian(x):
                jacobian = np.outer(x, 1 - x) - np.outer(x, np.ones_like(x)) + np.diag(x)# The softmax Jacobian
                return jacobian
            self.derivative = softmax_jacobian
            def softmax_jacobian_input(x):
                s = softmax(x)
                jacobian = np.outer(s, 1 - s) - np.outer(s, np.ones_like(s)) + np.diag(s)# The softmax Jacobian
                return jacobian
            self.derivative_input = softmax_jacobian_input
        activations_dict = {"identity" : identity_init, "relu" : relu_init, "soft_relu" : soft_relu_init, "sigmoid" : sigmoid_init, "softmax" : softmax_init}
        activations_dict[name](self)


if __name__ == "__main__":
    main()












