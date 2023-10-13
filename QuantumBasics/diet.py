import dimod
from dwave.system import LeapHybridCQMSampler

# code starts here
foods = {
    'rice'      : {'Calories': 100, 'Protein': 3, 'Fat': 1, 'Carbs': 22, 'Fiber': 2, 'Taste': 7, 'Cost': 2.5, 'Units': 'continuous'},
    'tofu'      : {'Calories': 140, 'Protein': 17,'Fat': 9, 'Carbs': 3,  'Fiber': 2, 'Taste': 2, 'Cost': 4.0, 'Units': 'continuous'},
    'banana'    : {'Calories': 90,  'Protein': 1, 'Fat': 0, 'Carbs': 23, 'Fiber': 3, 'Taste': 10,'Cost': 1.0, 'Units': 'discrete'},
    'lentils'   : {'Calories': 150, 'Protein': 9, 'Fat': 0, 'Carbs': 25, 'Fiber': 4, 'Taste': 3, 'Cost': 1.3, 'Units': 'continuous'},
    'bread'     : {'Calories': 270, 'Protein': 9, 'Fat': 3, 'Carbs': 50, 'Fiber': 3, 'Taste': 5, 'Cost': 0.25,'Units': 'continuous'},
    'avocado'   : {'Calories': 300, 'Protein': 4, 'Fat': 30,'Carbs': 20, 'Fiber': 14,'Taste': 5, 'Cost': 2.0, 'Units': 'discrete'}}

min_nutrients = {"Protein": 50, "Fat": 30, "Carbs": 130, "Fiber": 30}

max_calories = 2000

quantities = [
    dimod.Real(f"{food}") if foods[food]["Units"] == "continuous" 
    else dimod.Integer(f"{food}")
    for food in foods.keys()
]

print(quantities[0])

for ind, food in enumerate(foods.keys()):
    ub = max_calories / foods[food]["Calories"]
    quantities[ind].set_upper_bound(food, ub)

print(quantities[0].upper_bound("rice"))


def total_mix(quantity, category):
    return sum(q * c for q, c in zip(quantity, (foods[food][category] for food in foods.keys())))


def print_diet(cqm, sample):
    diet = {food: round(quantity, 1) for food, quantity in sample.items()}
    print(f"Diet: {diet}")
    taste_total = sum(foods[food]["Taste"] * amount for food, amount in sample.items())
    cost_total = sum(foods[food]["Cost"] * amount for food, amount in sample.items())
    print(f"Total taste of {round(taste_total, 2)} at cost {round(cost_total, 2)}")
    for constraint in cqm.iter_constraint_data(sample):
        print(f"{constraint.label} (nominal: {constraint.rhs_energy}): {round(constraint.lhs_energy)}")


def main():
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective( -1*total_mix(quantities, "Taste") + 6*total_mix(quantities, "Cost"))
    cqm.add_constraint(total_mix(quantities, "Calories") <= max_calories, label="Calories")

    for nutrient, amount in min_nutrients.items():
        cqm.add_constraint(total_mix(quantities, nutrient) >= amount, label=nutrient)

    print(list(cqm.constraints.keys()))
    print(cqm.constraints["Calories"].to_polystring())
    print(cqm.constraints["Protein"].to_polystring())
    print(cqm)

    sampler = LeapHybridCQMSampler()
    sampleset = sampler.sample_cqm(cqm)
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
    print("{} feasible solutions of {}.".format(len(feasible_sampleset), len(sampleset)))
    best = feasible_sampleset.first.sample
    print(best)
    print_diet(cqm, best)


if __name__ == "__main__":
    main()


# cqm.set_objective(-total_mix(quantities, "Taste"))
# sampleset_taste = sampler.sample_cqm(cqm)
# feasible_sampleset_taste = sampleset_taste.filter(lambda row: row.is_feasible)
# best_taste = feasible_sampleset_taste.first
# print(round(best_taste.energy))
# print_diet(best_taste.sample)

# cqm.set_objective(total_mix(quantities, "Cost"))
# sampleset_cost = sampler.sample_cqm(cqm)
# feasible_sampleset_cost = sampleset_cost.filter(lambda row: row.is_feasible)
# best_cost = feasible_sampleset_cost.first
# print(round(best_cost.energy))
# print_diet(best_cost.sample)

# cqm.set_objective(-total_mix(quantities, "Taste") + 1 * total_mix(quantities, "Cost"))
# sampleset = sampler.sample_cqm(cqm)
# feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)
# best = feasible_sampleset.first.sample
# print_diet(best)                                           

