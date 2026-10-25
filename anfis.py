# Develop an Adaptive Neuro-Fuzzy Inference System (ANFIS) for a prediction problem

import math
import random

class ANFIS:
    def __init__(self, num_inputs=2, mfs_per_input=2, lr_premise=0.01, lr_consequent=0.05, seed=42):
        random.seed(seed)
        self.num_inputs = num_inputs
        self.mfs_per_input = mfs_per_input
        self.lr_prem = lr_premise
        self.lr_cons = lr_consequent

        self.c = [
            [-0.5 + i * 1.0 for i in range(mfs_per_input)]
            for _ in range(num_inputs)
        ]
        self.sigma = [
            [1.0 for _ in range(mfs_per_input)]
            for _ in range(num_inputs)
        ]

        self.num_rules = mfs_per_input ** num_inputs
        self.rule_combinations = []
        for i in range(mfs_per_input):
            for j in range(mfs_per_input):
                self.rule_combinations.append((i, j))

        self.consequents = [
            [random.uniform(-0.5, 0.5) for _ in range(num_inputs + 1)]
            for _ in range(self.num_rules)
        ]

    def gaussian(self, x, c, sigma):
        sigma = max(1e-4, sigma)
        diff = (x - c) / sigma
        return math.exp(-0.5 * diff * diff)

    def forward(self, x):
        memberships = []
        for i in range(self.num_inputs):
            m_i = [self.gaussian(x[i], self.c[i][j], self.sigma[i][j]) for j in range(self.mfs_per_input)]
            memberships.append(m_i)

        w = []
        for idx1, idx2 in self.rule_combinations:
            firing = memberships[0][idx1] * memberships[1][idx2]
            w.append(max(1e-9, firing))

        sum_w = sum(w)
        w_bar = [w[r] / sum_w for r in range(self.num_rules)]

        f = []
        for r in range(self.num_rules):
            poly = self.consequents[r][0] * x[0] + self.consequents[r][1] * x[1] + self.consequents[r][2]
            f.append(poly)

        y_pred = sum(w_bar[r] * f[r] for r in range(self.num_rules))
        return y_pred, memberships, w, w_bar, f, sum_w

    def predict(self, x):
        y_pred, _, _, _, _, _ = self.forward(x)
        return y_pred

    def train(self, X_train, y_train, epochs=300):
        history = []
        for epoch in range(epochs):
            total_sq_err = 0.0

            for x, y_target in zip(X_train, y_train):
                y_pred, memberships, w, w_bar, f, sum_w = self.forward(x)
                err = y_pred - y_target
                total_sq_err += err * err

                for r in range(self.num_rules):
                    self.consequents[r][0] -= self.lr_cons * err * w_bar[r] * x[0]
                    self.consequents[r][1] -= self.lr_cons * err * w_bar[r] * x[1]
                    self.consequents[r][2] -= self.lr_cons * err * w_bar[r]

                dE_dw = [err * (f[r] - y_pred) / sum_w for r in range(self.num_rules)]

                for r, (idx1, idx2) in enumerate(self.rule_combinations):
                    delta_w = dE_dw[r]

                    diff0 = (x[0] - self.c[0][idx1]) / (self.sigma[0][idx1] ** 2)
                    dE_dc0 = delta_w * memberships[1][idx2] * memberships[0][idx1] * diff0
                    dE_dsig0 = delta_w * memberships[1][idx2] * memberships[0][idx1] * (diff0 * (x[0] - self.c[0][idx1]) / self.sigma[0][idx1])
                    self.c[0][idx1] -= self.lr_prem * dE_dc0
                    self.sigma[0][idx1] = max(0.1, self.sigma[0][idx1] - self.lr_prem * dE_dsig0)

                    diff1 = (x[1] - self.c[1][idx2]) / (self.sigma[1][idx2] ** 2)
                    dE_dc1 = delta_w * memberships[0][idx1] * memberships[1][idx2] * diff1
                    dE_dsig1 = delta_w * memberships[0][idx1] * memberships[1][idx2] * (diff1 * (x[1] - self.c[1][idx2]) / self.sigma[1][idx2])
                    self.c[1][idx2] -= self.lr_prem * dE_dc1
                    self.sigma[1][idx2] = max(0.1, self.sigma[1][idx2] - self.lr_prem * dE_dsig1)

            rmse = math.sqrt(total_sq_err / len(X_train))
            if (epoch + 1) % 60 == 0 or epoch == 0:
                history.append((epoch + 1, rmse))

        return history

def target_system(x1, x2):
    return math.sin(x1) + 0.5 * math.cos(x2) + 0.3 * (x1 * x2)

if __name__ == "__main__":
    print("=" * 70)
    print("     ADAPTIVE NEURO-FUZZY INFERENCE SYSTEM (ANFIS) PREDICTION")
    print("=" * 70)
    print("Architecture Overview:")
    print("  Model Type : First-Order Sugeno Neuro-Fuzzy System (5 Layers)")
    print("  Inputs (2) : x1, x2 in range [-1.0, 1.0]")
    print("  Fuzzy Sets : 2 Gaussian MFs per input -> 4 Sugeno Rules")
    print("  Target     : Non-Linear Dynamic Function f(x1, x2) = sin(x1) + 0.5*cos(x2) + 0.3*x1*x2")
    print("-" * 70)

    random.seed(10)
    train_points = [
        [round(random.uniform(-1.0, 1.0), 3), round(random.uniform(-1.0, 1.0), 3)]
        for _ in range(60)
    ]
    y_train = [target_system(p[0], p[1]) for p in train_points]

    test_points = [
        [-0.8, -0.6], [-0.5, 0.4], [0.0, 0.0], [0.3, -0.7], [0.7, 0.8], [0.9, -0.2]
    ]
    y_test = [target_system(p[0], p[1]) for p in test_points]

    anfis = ANFIS(num_inputs=2, mfs_per_input=2, lr_premise=0.02, lr_consequent=0.08, seed=42)

    print("Training ANFIS Model...")
    history = anfis.train(train_points, y_train, epochs=300)
    for epoch, rmse in history:
        print(f"  Epoch {epoch:3d} | Root Mean Squared Error (RMSE): {rmse:.6f}")

    print("-" * 70)
    print("Testing ANFIS on Unseen Test Dataset:")
    print(f"{'Input (x1, x2)':^20} | {'Actual y':^12} | {'Predicted y':^14} | {'Abs Error':^12}")
    print("-" * 70)
    abs_errors = []
    for x, y_act in zip(test_points, y_test):
        y_p = anfis.predict(x)
        err = abs(y_p - y_act)
        abs_errors.append(err)
        print(f"  ({x[0]:>5.2f}, {x[1]:>5.2f})    | {y_act:^12.4f} | {y_p:^14.4f} | {err:^12.4f}")

    test_rmse = math.sqrt(sum(e ** 2 for e in abs_errors) / len(abs_errors))
    test_mae = sum(abs_errors) / len(abs_errors)
    print("-" * 70)
    print(f"Prediction Performance Summary:")
    print(f"  Test RMSE : {test_rmse:.6f}")
    print(f"  Test MAE  : {test_mae:.6f}")
    print("=" * 70)