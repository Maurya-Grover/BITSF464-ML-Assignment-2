class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        result = []
        size = len(input_data)

        for i in range(size):
            # forward prop
            output = input_data[i]
            for layer in layers:
                output = layer.forward_prop(output)
            result.append(output)
        return result

    def fit(self, x_train, y_train, epochs=1000, learning_rate=0.01):
        samples = x_train.shape[0]
        
        # training
        for epoch in range(epochs):
            err = 0
            for i in range(samples):
                # forward prop
                output = x_train[i]
                for layer in self.layers:
                    output = layer.forward_prop(output)
                # for use in graph
                err += self.loss(y_train[i], output)

                # back prop
                error = self.loss_prime(y_train[i], output)
                for layer in self.layers:
                    error = layer.backward_prop(error, learning_rate)

            if epoch % 50 == 0:
                err /= samples
                print(f"Epoch : {epoch}/{epochs}  Error = {err:.3f}")
