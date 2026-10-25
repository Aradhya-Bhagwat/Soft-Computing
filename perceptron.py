# Design and implement AND, OR, and XOR operations through perceptron

class Perceptron:
    def __init__(self, weights=None, bias=0.0, learning_rate=0.1):
        self.weights = weights if weights is not None else [0.0, 0.0]
        self.bias = bias
        self.lr = learning_rate

    def predict(self, inputs):
        activation = self.bias + sum(w * x for w, x in zip(self.weights, inputs))
        return 1 if activation > 0 else 0

    def train(self, training_inputs, targets, epochs=20):
        for _ in range(epochs):
            converged = True
            for x, y in zip(training_inputs, targets):
                pred = self.predict(x)
                error = y - pred
                if error != 0:
                    converged = False
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * x[i]
                    self.bias += self.lr * error
            if converged:
                break

def AND(x1, x2):
    p = Perceptron([1.0, 1.0], -1.5)
    return p.predict([x1, x2])

def OR(x1, x2):
    p = Perceptron([1.0, 1.0], -0.5)
    return p.predict([x1, x2])

def NAND(x1, x2):
    p = Perceptron([-1.0, -1.0], 1.5)
    return p.predict([x1, x2])

def XOR(x1, x2):
    or_out = OR(x1, x2)
    nand_out = NAND(x1, x2)
    return AND(or_out, nand_out)

def display_truth_table(name, gate_fn, notes=""):
    print(f"\n--- {name} Operation ---")
    if notes:
        print(f"Implementation: {notes}")
    print(f"{'X1':^6} | {'X2':^6} | {'Output':^8}")
    print("-" * 26)
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for x1, x2 in inputs:
        res = gate_fn(x1, x2)
        print(f"{x1:^6} | {x2:^6} | {res:^8}")

if __name__ == "__main__":
    print("=" * 60)
    print("   PERCEPTRON DESIGN FOR LOGIC GATES (AND, OR, XOR)")
    print("=" * 60)

    print("\n[Analysis of Linear Separability]")
    print("- AND and OR gates are linearly separable (single perceptron).")
    print("- XOR is non-linearly separable; implemented via multi-perceptron.")

    display_truth_table("AND Gate", AND, "Single Perceptron (Weights=[1.0, 1.0], Bias=-1.5)")
    display_truth_table("OR Gate", OR, "Single Perceptron (Weights=[1.0, 1.0], Bias=-0.5)")
    display_truth_table("XOR Gate", XOR, "Multi-Perceptron: XOR = AND(OR(x1, x2), NAND(x1, x2))")

    print("\n" + "=" * 60)
    print("Training Perceptron from Scratch using Perceptron Learning Rule:")
    inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    and_targets = [0, 0, 0, 1]
    or_targets = [0, 1, 1, 1]

    p_and = Perceptron(weights=[0.0, 0.0], bias=0.0, learning_rate=0.1)
    p_and.train(inputs, and_targets)
    print(f"  Trained AND Perceptron -> Weights: {[round(w, 2) for w in p_and.weights]}, Bias: {round(p_and.bias, 2)}")

    p_or = Perceptron(weights=[0.0, 0.0], bias=0.0, learning_rate=0.1)
    p_or.train(inputs, or_targets)
    print(f"  Trained OR Perceptron  -> Weights: {[round(w, 2) for w in p_or.weights]}, Bias: {round(p_or.bias, 2)}")
    print("=" * 60)