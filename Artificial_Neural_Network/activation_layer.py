from layer import Layer

class ACLayer(Layer):
    
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_prop(self, input_vector):
        self.input = input_vector
        self.output = self.activation(self.input)
        return self.output
    
    def back_prop(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output error