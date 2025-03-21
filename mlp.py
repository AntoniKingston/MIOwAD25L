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
            result.array[i] = self.array[i] + other.array[i]
        return result

    def __radd__(self, other):
        if other == 0:
            return self  # Ensure 0 + ArrayOfMatrices returns ArrayOfMatrices
        raise TypeError("Addition only supports another ArrayOfMatrices or 0")

    def __sub__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = self.array[i] - other.array[i]
        return result

    def __mul__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = self.array[i] * other
        return result

    def __truediv__(self, other):
        result = ArrayOfMatrices(self.shapes)
        for i in range(len(self.array)):
            result.array[i] = self.array[i] / other
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




class MLP_with_backpropagation():

    def identity(x):
        return x

    def derivative_sv(function, x, dx=10 ** (-9)):
        return (function(x + dx) - function(x)) / dx

    def __init__(self, shape, activations=None):  # len(activations)+1=len(shape)
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

        if activations:
            self.activations = [np.vectorize(activation) for activation in activations]
            self.activations_sv = [activation for activation in activations]
        else:
            self.activations = [np.vectorize(MLP_with_backpropagation.identity)] * (len(shape) - 1)
            self.activations_sv = [MLP_with_backpropagation.identity] * (len(shape) - 1)

    def set_input(self, inputt):
        self.fed_values.array[0] = np.array(inputt)
        self.activation_values.array[0] = np.array(inputt)  # input is input
        return True

    def feed_forward(self):
        for i in range(1, len(self.fed_values.array)):
            self.fed_values.array[i] = np.dot(self.weights.array[i - 1], self.activation_values.array[i - 1]) + \
                                       self.biases.array[
                                           i - 1]  ##can do if statement if activation is function of layer not neuron
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
        dx = 10 ** (-6)
        weight_grad = ArrayOfMatrices(self.weights.shapes)
        bias_grad = ArrayOfMatrices(self.biases.shapes)
        neuron_activation_grad = ArrayOfMatrices(self.fed_values.shapes)
        neuron_fed_grad = ArrayOfMatrices(self.fed_values.shapes)
        last_layer = True
        for i in range(len(self.fed_values.array) - 1, 0, -1):
            # derivatives with respect to neuron activation values
            for j in range(len(self.activation_values.array[i])):
                if last_layer:
                    x_0 = self.activation_values.array[i][j]
                    y_0 = expected[j]
                    neuron_activation_grad.array[i][j] = 2 * (x_0 - y_0)
                else:
                    neuron_activation_grad.array[i][j] = sum(
                        [neuron_fed_grad.array[i + 1][k] * self.weights.array[i][k][j] for k in
                         range(len(neuron_fed_grad.array[i + 1]))])

                neuron_fed_grad.array[i][j] = neuron_activation_grad.array[i][
                                                  j] * MLP_with_backpropagation.derivative_sv(
                    self.activations_sv[i - 1], self.fed_values.array[i][j])
                bias_grad.array[i - 1][j] = neuron_fed_grad.array[i][j]

                for k in range(len(self.weights.array[i - 1][j])):
                    weight_grad.array[i - 1][j][k] = neuron_fed_grad.array[i][j] * self.activation_values.array[i - 1][
                        k]
            last_layer = False
        return (weight_grad, bias_grad)

    def learn_batch(self, batchx, batchy, lr):
        batch_size = len(batchx)
        for i in range(batch_size):
            local_weight_gradient, local_bias_gradient = self.backpropagate(batchx[i], batchy[i])
            if i == 0:
                avg_weight_gradient = ArrayOfMatrices(local_weight_gradient.shapes)
                avg_bias_gradient = ArrayOfMatrices(local_bias_gradient.shapes)
            avg_weight_gradient = avg_weight_gradient + local_weight_gradient
            avg_bias_gradient = avg_bias_gradient + local_bias_gradient
        avg_weight_gradient /= batch_size
        avg_bias_gradient /= batch_size
        self.weights = self.weights - avg_weight_gradient * lr
        self.biases = self.biases - avg_bias_gradient * lr

    def epoch(self, x, y, lr=0.01, batch_size=None):
        permutation = np.random.permutation(len(x))
        x_sh = x[permutation]
        y_sh = y[permutation]
        if not batch_size:
            batch_size = min(32, len(x))
        idx = 0
        while (idx + batch_size < len(x)):
            batchx = x_sh[idx:idx + batch_size]
            batchy = y_sh[idx:idx + batch_size]
            self.learn_batch(batchx, batchy, lr)
            idx += batch_size
        if idx != len(x) - 1:
            batchx = x_sh[idx:]
            batchy = y_sh[idx:]
            self.learn_batch(batchx, batchy, lr)

    def update_network(self, weight_gradients, bias_gradients):
        pass
def main():
    weightsss = []
    for i in range(2):
        network = MLP_with_backpropagation([1,5,1])
        print(network.weights)
        weightsss.append(network.weights)

    print(sum(weightsss))

class FunctionWithDerivative():
    def __init__(self, function):
        self.function = function
    def value(self, x):
        if function == "identity":
            return

if __name__ == "__main__":
    main()












