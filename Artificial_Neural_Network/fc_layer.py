from layer import Layer


class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_prop(self, input_vector):
        self.input = input_vector
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def back_prop(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        dw = np.dot(self.input.T, output_error)
        db = output_error

        self.weights = self.weights - learning_rate * dw
        self.bias = self.bias - learning_rate * db
        return input_error