# Implement multi-layer feed-forward Networks

import math
import random

class MultiLayerFeedForward:
    def __init__(self, layer_sizes, learning_rate=0.5, seed=42):
        if seed is not None:
            random.seed(seed)
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))
            w = [[random.uniform(-limit, limit) for _ in range(fan_out)] for _ in range(fan_in)]
            b = [0.0 for _ in range(fan_out)]
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

    def sigmoid_derivative(self, val):
        return val * (1.0 - val)

    def forward(self, inputs):
        activations = [inputs]
        current = inputs
        for w, b in zip(self.weights, self.biases):
            next_act = []
            for j in range(len(b)):
                z = b[j] + sum(current[i] * w[i][j] for i in range(len(current)))
                next_act.append(self.sigmoid(z))
            current = next_act
            activations.append(current)
        return activations[-1]

    def forward_full(self, inputs):
        activations = [inputs]
        current = inputs
        for w, b in zip(self.weights, self.biases):
            next_act = []
            for j in range(len(b)):
                z = b[j] + sum(current[i] * w[i][j] for i in range(len(current)))
                next_act.append(self.sigmoid(z))
            current = next_act
            activations.append(current)
        return activations

    def train(self, training_inputs, training_targets, epochs=2000):
        history = []
        num_layers = len(self.weights)

        for epoch in range(epochs):
            total_error = 0.0

            for x, y in zip(training_inputs, training_targets):
                acts = self.forward_full(x)
                output = acts[-1]

                sample_error = sum((y[j] - output[j]) ** 2 for j in range(len(y))) / len(y)
                total_error += sample_error

                deltas = [None] * num_layers

                output_delta = []
                for j in range(len(output)):
                    err = output[j] - y[j]
                    output_delta.append(err * self.sigmoid_derivative(output[j]))
                deltas[-1] = output_delta

                for l in range(num_layers - 2, -1, -1):
                    current_act = acts[l + 1]
                    next_delta = deltas[l + 1]
                    current_w = self.weights[l + 1]

                    layer_delta = []
                    for i in range(len(current_act)):
                        backprop_err = sum(current_w[i][j] * next_delta[j] for j in range(len(next_delta)))
                        layer_delta.append(backprop_err * self.sigmoid_derivative(current_act[i]))
                    deltas[l] = layer_delta

                for l in range(num_layers):
                    prev_act = acts[l]
                    curr_delta = deltas[l]
                    for i in range(len(prev_act)):
                        for j in range(len(curr_delta)):
                            self.weights[l][i][j] -= self.lr * curr_delta[j] * prev_act[i]
                    for j in range(len(curr_delta)):
                        self.biases[l][j] -= self.lr * curr_delta[j]

            mse = total_error / len(training_inputs)
            if (epoch + 1) % 400 == 0 or epoch == 0:
                history.append((epoch + 1, mse))

        return history

MLFF = MultiLayerFeedForward

if __name__ == "__main__":
    print("=" * 60)
    print("       MULTI-LAYER FEED-FORWARD NETWORK (MLFF)")
    print("=" * 60)

    xor_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0])
    ]
    inputs = [d[0] for d in xor_data]
    targets = [d[1] for d in xor_data]

    architecture = [2, 4, 1]
    network = MultiLayerFeedForward(architecture, learning_rate=0.8, seed=42)
    print(f"Network Architecture: {' -> '.join(str(s) for s in architecture)}")
    print(f"Activation Function : Sigmoid")
    print(f"Learning Rate       : {network.lr}")
    print("-" * 60)

    print("Initial Untrained Predictions on XOR:")
    for x, y in xor_data:
        pred = network.forward(x)[0]
        print(f"  Input: {x} -> Predicted: {pred:.4f} (Target: {y[0]:.1f})")

    print("-" * 60)
    print("Training Network using Backpropagation Algorithm...")
    history = network.train(inputs, targets, epochs=2000)
    for epoch, mse in history:
        print(f"  Epoch {epoch:4d} | Mean Squared Error (MSE): {mse:.6f}")

    print("-" * 60)
    print("Final Trained Evaluation on XOR Problem:")
    print(f"{'Input':^15} | {'Target':^10} | {'Probability':^15} | {'Decision':^10}")
    print("-" * 60)
    for x, y in xor_data:
        pred_prob = network.forward(x)[0]
        decision = 1 if pred_prob >= 0.5 else 0
        print(f"{str(x):^15} | {y[0]:^10.1f} | {pred_prob:^15.4f} | {decision:^10d}")

    print("=" * 60)