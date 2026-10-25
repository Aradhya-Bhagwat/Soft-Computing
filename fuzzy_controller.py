# Implement a fuzzy controller system for a real world application

def trimf(x, a, b, c):
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    else:
        return (c - x) / (c - b) if c != b else 1.0

class WashingMachineFuzzyController:
    def __init__(self):
        self.dirt_mf = {
            "Low": lambda x: trimf(x, 0.0, 0.0, 50.0),
            "Medium": lambda x: trimf(x, 0.0, 50.0, 100.0),
            "High": lambda x: trimf(x, 50.0, 100.0, 100.0)
        }
        self.grease_mf = {
            "Low": lambda x: trimf(x, 0.0, 0.0, 50.0),
            "Medium": lambda x: trimf(x, 0.0, 50.0, 100.0),
            "High": lambda x: trimf(x, 50.0, 100.0, 100.0)
        }
        self.wash_time_mf = {
            "Very Short": lambda x: trimf(x, 0.0, 0.0, 15.0),
            "Short": lambda x: trimf(x, 0.0, 15.0, 30.0),
            "Medium": lambda x: trimf(x, 15.0, 30.0, 45.0),
            "Long": lambda x: trimf(x, 30.0, 45.0, 60.0),
            "Very Long": lambda x: trimf(x, 45.0, 60.0, 60.0)
        }
        self.rules = [
            ("Low", "Low", "Very Short"),
            ("Low", "Medium", "Short"),
            ("Low", "High", "Medium"),
            ("Medium", "Low", "Short"),
            ("Medium", "Medium", "Medium"),
            ("Medium", "High", "Long"),
            ("High", "Low", "Medium"),
            ("High", "Medium", "Long"),
            ("High", "High", "Very Long")
        ]

    def fuzzify(self, dirt, grease):
        dirt_deg = {term: func(dirt) for term, func in self.dirt_mf.items()}
        grease_deg = {term: func(grease) for term, func in self.grease_mf.items()}
        return dirt_deg, grease_deg

    def evaluate(self, dirt, grease):
        dirt_deg, grease_deg = self.fuzzify(dirt, grease)
        rule_firings = []

        for d_term, g_term, out_term in self.rules:
            strength = min(dirt_deg[d_term], grease_deg[g_term])
            if strength > 0.0:
                rule_firings.append((d_term, g_term, out_term, strength))

        time_points = [i * 0.5 for i in range(121)]
        aggregated = []
        for t in time_points:
            membership = 0.0
            for _, _, out_term, strength in rule_firings:
                term_val = self.wash_time_mf[out_term](t)
                clipped = min(strength, term_val)
                if clipped > membership:
                    membership = clipped
            aggregated.append((t, membership))

        num = sum(t * mu for t, mu in aggregated)
        den = sum(mu for t, mu in aggregated)
        crisp_time = (num / den) if den != 0.0 else 30.0

        return {
            "dirt": dirt,
            "grease": grease,
            "dirt_deg": dirt_deg,
            "grease_deg": grease_deg,
            "rule_firings": rule_firings,
            "crisp_time": crisp_time
        }

if __name__ == "__main__":
    WIDTH = 97
    print("=" * WIDTH)
    print("REAL-WORLD FUZZY CONTROLLER: WASHING MACHINE SYSTEM".center(WIDTH))
    print("=" * WIDTH)
    print("Application Overview:")
    print("  Inputs   : Dirtiness (0-100%), Grease (0-100%)")
    print("  Output   : Wash Time in minutes (0-60 min)")
    print("  Inference: Mamdani Min-Max Inference with Centroid Defuzzification")
    print("-" * WIDTH)

    controller = WashingMachineFuzzyController()
    test_cases = [
        (20.0, 15.0, "Light Wash"),
        (55.0, 50.0, "Medium Wash"),
        (85.0, 90.0, "Heavy Wash")
    ]

    for dirt, grease, label in test_cases:
        res = controller.evaluate(dirt, grease)
        print(f"\nScenario: {label} (Dirtiness = {dirt}%, Grease = {grease}%)")
        print("  Fuzzified Dirtiness: " + ", ".join(f"{k}={v:.2f}" for k, v in res["dirt_deg"].items() if v > 0))
        print("  Fuzzified Grease   : " + ", ".join(f"{k}={v:.2f}" for k, v in res["grease_deg"].items() if v > 0))
        print("  Active Fuzzy Rules :")
        for d_t, g_t, out_t, s in res["rule_firings"]:
            print(f"    - IF Dirt is {d_t:<6} AND Grease is {g_t:<6} THEN WashTime is {out_t:<10} [Firing Strength = {s:.2f}]")
        print(f"  Recommended Wash Time (Defuzzified): {res['crisp_time']:.2f} minutes")
        print("-" * WIDTH)
    print("=" * WIDTH)
