# Implement union, intersection, complement and difference operations on fuzzy sets

def get_universe(A, B=None):
    if B is None:
        return sorted(list(A.keys()))
    return sorted(list(set(A.keys()) | set(B.keys())))

def fuzzy_union(A, B, universe=None):
    u = universe if universe is not None else get_universe(A, B)
    return {k: round(max(A.get(k, 0.0), B.get(k, 0.0)), 4) for k in u}

def fuzzy_intersection(A, B, universe=None):
    u = universe if universe is not None else get_universe(A, B)
    return {k: round(min(A.get(k, 0.0), B.get(k, 0.0)), 4) for k in u}

def fuzzy_complement(A, universe=None):
    u = universe if universe is not None else get_universe(A)
    return {k: round(1.0 - A.get(k, 0.0), 4) for k in u}

def fuzzy_difference(A, B, universe=None):
    u = universe if universe is not None else get_universe(A, B)
    b_comp = fuzzy_complement(B, universe=u)
    return {k: round(min(A.get(k, 0.0), b_comp.get(k, 1.0)), 4) for k in u}

def format_fuzzy_set(fuzzy_dict):
    items = [f"{k}: {v:.2f}" for k, v in sorted(fuzzy_dict.items())]
    return "{" + ", ".join(items) + "}"

if __name__ == "__main__":
    print("=" * 60)
    print("                    FUZZY SET OPERATIONS")
    print("=" * 60)

    A = {"a": 0.2, "b": 0.5, "c": 0.8, "d": 0.0}
    B = {"a": 0.4, "b": 0.3, "c": 0.0, "d": 0.9}
    universe = get_universe(A, B)

    print(f"Universe of Discourse : {universe}")
    print(f"Fuzzy Set A           : {format_fuzzy_set(A)}")
    print(f"Fuzzy Set B           : {format_fuzzy_set(B)}")
    print("-" * 60)

    union_res = fuzzy_union(A, B, universe)
    inter_res = fuzzy_intersection(A, B, universe)
    comp_a = fuzzy_complement(A, universe)
    comp_b = fuzzy_complement(B, universe)
    diff_ab = fuzzy_difference(A, B, universe)
    diff_ba = fuzzy_difference(B, A, universe)

    print("1. Fuzzy Union (A U B):")
    print("   Formula: mu_AUB(x) = max(mu_A(x), mu_B(x))")
    print(f"   Result : {format_fuzzy_set(union_res)}\n")

    print("2. Fuzzy Intersection (A ∩ B):")
    print("   Formula: mu_A∩B(x) = min(mu_A(x), mu_B(x))")
    print(f"   Result : {format_fuzzy_set(inter_res)}\n")

    print("3. Fuzzy Complement of A (A'):")
    print("   Formula: mu_A'(x) = 1 - mu_A(x)")
    print(f"   Result : {format_fuzzy_set(comp_a)}\n")

    print("4. Fuzzy Complement of B (B'):")
    print("   Formula: mu_B'(x) = 1 - mu_B(x)")
    print(f"   Result : {format_fuzzy_set(comp_b)}\n")

    print("5. Fuzzy Difference (A - B = A ∩ B'):")
    print("   Formula: mu_A-B(x) = min(mu_A(x), 1 - mu_B(x))")
    print(f"   Result : {format_fuzzy_set(diff_ab)}\n")

    print("6. Fuzzy Difference (B - A = B ∩ A'):")
    print("   Formula: mu_B-A(x) = min(mu_B(x), 1 - mu_A(x))")
    print(f"   Result : {format_fuzzy_set(diff_ba)}")
    print("=" * 60)