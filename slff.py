# Implement single-layer feed-forward Networks

import math
import random

class SingleLayerFeedForward:
    def __init__(self, input_size, output_size, learning_rate=0.1, seed=42):
        if seed is not None:
            random.seed(seed)
        self.input_size = input_size
        self.output_size = output_size
        self.lr = learning_rate
        self.weights = [[random.uniform(-0.5, 0.5) for _ in range(output_size)] for _ in range(input_size)]
        self.bias = [random.uniform(-0.5, 0.5) for _ in range(output_size)]

    def sigmoid(self, x):
        return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))

    def sigmoid_derivative(self, output):
        return output * (1.0 - output)

    def forward(self, inputs):
        outputs = []
        for j in range(self.output_size):
            activation = self.bias[j]
            for i in range(self.input_size):
                activation += inputs[i] * self.weights[i][j]
            outputs.append(self.sigmoid(activation))
        return outputs

    def train(self, training_inputs, training_targets, epochs=500):
        history = []
        for epoch in range(epochs):
            total_error = 0.0
            for x, target in zip(training_inputs, training_targets):
                pred = self.forward(x)
                errors = [target[j] - pred[j] for j in range(self.output_size)]
                total_error += sum(e ** 2 for e in errors) / self.output_size

                for j in range(self.output_size):
                    delta = errors[j] * self.sigmoid_derivative(pred[j])
                    for i in range(self.input_size):
                        self.weights[i][j] += self.lr * delta * x[i]
                    self.bias[j] += self.lr * delta

            mse = total_error / len(training_inputs)
            if (epoch + 1) % 100 == 0 or epoch == 0:
                history.append((epoch + 1, mse))
        return history

SLFF = SingleLayerFeedForward

if __name__ == "__main__":
    print("=" * 60)
    print("      SINGLE-LAYER FEED-FORWARD NETWORK (SLFF)")
    print("=" * 60)

    dataset = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [1.0])
    ]
    inputs = [d[0] for d in dataset]
    targets = [d[1] for d in dataset]

    network = SingleLayerFeedForward(input_size=2, output_size=1, learning_rate=0.5, seed=42)
    print(f"Network Architecture: {network.input_size} Input Neuron(s) -> {network.output_size} Output Neuron(s)")
    print(f"Activation Function : Sigmoid")
    print(f"Learning Rate       : {network.lr}")
    print("-" * 60)

    print("Initial Untrained Predictions:")
    for x, y in dataset:
        out = network.forward(x)[0]
        print(f"  Input: {x} -> Predicted: {out:.4f} (Target: {y[0]:.1f})")

    print("-" * 60)
    print("Training Network using Delta Rule / Gradient Descent...")
    history = network.train(inputs, targets, epochs=500)
    for epoch, mse in history:
        print(f"  Epoch {epoch:3d} | Mean Squared Error (MSE): {mse:.6f}")

    print("-" * 60)
    print("Final Trained Evaluation:")
    print(f"{'Input':^15} | {'Target':^10} | {'Probability':^15} | {'Decision':^10}")
    print("-" * 60)
    for x, y in dataset:
        pred_prob = network.forward(x)[0]
        decision = 1 if pred_prob >= 0.5 else 0
        print(f"{str(x):^15} | {y[0]:^10.1f} | {pred_prob:^15.4f} | {decision:^10d}")

    print("-" * 60)
    print("Learned Weights and Biases:")
    for i in range(network.input_size):
        print(f"  Weight w_{i+1}->out: {network.weights[i][0]:.4f}")
    print(f"  Bias b->out       : {network.bias[0]:.4f}")
    print("=" * 60)