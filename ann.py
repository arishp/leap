import dimod
from numpy import random
import numpy as np
from dwave.system import LeapHybridCQMSampler

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # XOR input
Y = np.array([[0, 1, 1, 0]])  # XOR output
weights = {
    'level1': {'w11': 0.0, 'w12': 0.0, 'b1': 0.0, 'w21': 0.0, 'w22': 0.0, 'b2': 0.0},
    'level2': {'w31': 0.0, 'w41': 0.0, 'b3': 0.0}
}
for level in weights.keys():
    for key in weights[level].keys():
        weights[level][key] = random.uniform(-1.0, 1.0)
print('INITIAL WEIGHTS: ')
for key in weights:
    print(weights[key])
lr = 0.01
print('LEARNING RATE: ', lr, '\n')

num_iterations = 3
for iteration in range(num_iterations):
    print('BACK PROPAGATION ITERATION - ', iteration + 1, ':')
    ######### BACK PROPAGATION - PHASE 1 ##########
    print('PHASE 1:')
    constant_level = 'level1'
    variable_level = 'level2'
    quadratic_weights = [dimod.Real(key) for key in weights[variable_level].keys()]
    for i in range(len(quadratic_weights)):
        print(quadratic_weights[i])
    for i, key in enumerate(weights[variable_level].keys()):
        quadratic_weights[i].set_lower_bound(key, -1e+29)
    cqm = dimod.ConstrainedQuadraticModel()
    obj_list = [(2.0 * weights[constant_level]['w11'] + 2.0 * weights[constant_level]['w21'] + 4.0 *
                 weights[constant_level]['b1']) * quadratic_weights[0],
                (2.0 * weights[constant_level]['w12'] + 2.0 * weights[constant_level]['w22'] + 4.0 *
                 weights[constant_level]['b2']) * quadratic_weights[1],
                4.0 * quadratic_weights[2], -2.0]
    cqm.set_objective(sum(obj_list))
    cqm.add_constraint(weights[constant_level]['b1'] * quadratic_weights[0] +
                       weights[constant_level]['b2'] * quadratic_weights[1] +
                       quadratic_weights[2] >= 0.0, label="Constraint 1")
    cqm.add_constraint((weights[constant_level]['w21'] + weights[constant_level]['b1']) * quadratic_weights[0] +
                       (weights[constant_level]['w22'] + weights[constant_level]['b2']) * quadratic_weights[1] +
                       quadratic_weights[2] - 1.0 >= 0.0, label="Constraint 2")
    cqm.add_constraint((weights[constant_level]['w11'] + weights[constant_level]['b1']) * quadratic_weights[0] +
                       (weights[constant_level]['w12'] + weights[constant_level]['b2']) * quadratic_weights[1] +
                       quadratic_weights[2] - 1.0 >= 0.0, label="Constraint 3")
    cqm.add_constraint((weights[constant_level]['w11'] + weights[constant_level]['w21'] +
                        weights[constant_level]['b1']) * quadratic_weights[0] +
                       (weights[constant_level]['w12'] + weights[constant_level]['w22'] +
                        weights[constant_level]['b2']) * quadratic_weights[1] +
                       quadratic_weights[2] >= 0.0, label="Constraint 4")
    print('\n', cqm)
    sampler = LeapHybridCQMSampler()
    sampleSet = sampler.sample_cqm(cqm)
    feasible_sampleSet = sampleSet.filter(lambda row: row.is_feasible)
    if len(feasible_sampleSet) > 0:
        print("FIRST FEASIBLE SAMPLE: ")
        print(feasible_sampleSet.first.sample)
        for key in feasible_sampleSet.first.sample.keys():
            weights[variable_level][key] = feasible_sampleSet.first.sample[key] # weights[variable_level][key] + lr * (feasible_sampleSet.first.sample[key] - weights[variable_level][key])
        print("weights:")
        for key in weights:
            print(weights[key])
        print('')
    else:
        print('feasible sample set not found')
        break

    #############################################################################################################################
    # for key in weights[variable_level].keys():
    #     weights[variable_level][key] = weights[variable_level][key] + lr * (random.uniform(-1, 1) - weights[variable_level][key])
    # print("weights: ")
    # for key in weights:
    #     print(weights[key])
    # print('')
    #############################################################################################################################
    W1 = np.array([[weights['level1']['w11'], weights['level1']['w12']],
                   [weights['level1']['w21'], weights['level1']['w22']],
                   [weights['level1']['b1'], weights['level1']['b2']]])
    W2 = np.array([[weights['level2']['w31']],
                   [weights['level2']['w41']],
                   [weights['level2']['b3']]])
    sq_error = 0.0
    for i in range(X.shape[1]):
        inputs = np.array([[X[0][i], X[1][i], 1.0]])
        hidden_sum = np.matmul(inputs, W1)
        hidden_act = hidden_sum # 1 / (1 + np.exp(-hidden_sum))
        hidden_out = np.matmul(hidden_act, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) + [0.0, 0.0, 1.0]
        output_sum = np.matmul(hidden_out, W2)
        outputs = output_sum # 1 / (1 + np.exp(-output_sum))
        print('input: ', inputs[0][:2], 'output: ', Y[0][i], 'prediction: ', outputs[0][0])
        sq_error = sq_error + (Y[0][i] - outputs[0][0])*(Y[0][i] - outputs[0][0])
        # log_probs = log_probs + Y[0][i]*np.log(outputs[0][0]) + (1 - Y[0][i])*np.log(1 - outputs[0][0])
    print('RMSE: ', np.sqrt(sq_error/X.shape[1]))

    ######### BACK PROPAGATION - PHASE 2 ##########
    print('PHASE 2:')
    constant_level = 'level2'
    variable_level = 'level1'
    quadratic_weights = [dimod.Real(key) for key in weights[variable_level].keys()]
    for i in range(len(quadratic_weights)):
        print(quadratic_weights[i])
    for i, key in enumerate(weights[variable_level].keys()):
        quadratic_weights[i].set_lower_bound(key, -1e+29)
    cqm = dimod.ConstrainedQuadraticModel()
    obj_list = [2.0 * weights[constant_level]['w31'] * quadratic_weights[0],
                2.0 * weights[constant_level]['w41'] * quadratic_weights[1],
                2.0 * weights[constant_level]['w31'] * quadratic_weights[2],
                2.0 * weights[constant_level]['w41'] * quadratic_weights[3],
                4.0 * weights[constant_level]['w31'] * quadratic_weights[4],
                4.0 * weights[constant_level]['w41'] * quadratic_weights[5],
                4.0 * weights[constant_level]['b3'], -2.0]
    cqm.set_objective(sum(obj_list))
    cqm.add_constraint(weights[constant_level]['w31'] * quadratic_weights[4] +
                       weights[constant_level]['w41'] * quadratic_weights[5] +
                       weights[constant_level]['b3'] >= 0.0, label="Constraint 1")
    cqm.add_constraint(weights[constant_level]['w31'] * (quadratic_weights[2] + quadratic_weights[4]) +
                       weights[constant_level]['w41'] * (quadratic_weights[3] + quadratic_weights[5]) +
                       weights[constant_level]['b3'] - 1.0 >= 0.0, label="Constraint 2")
    cqm.add_constraint(weights[constant_level]['w31'] * (quadratic_weights[0] + quadratic_weights[4]) +
                       weights[constant_level]['w41'] * (quadratic_weights[1] + quadratic_weights[5]) +
                       weights[constant_level]['b3'] - 1.0 >= 0.0, label="Constraint 3")
    cqm.add_constraint(
        weights[constant_level]['w31'] * (quadratic_weights[0] + quadratic_weights[2] + quadratic_weights[4]) +
        weights[constant_level]['w41'] * (quadratic_weights[1] + quadratic_weights[3] + quadratic_weights[5]) +
        weights[constant_level]['b3'] >= 0.0, label="Constraint 4")
    print('\n', cqm)
    sampler = LeapHybridCQMSampler()
    sampleSet = sampler.sample_cqm(cqm)
    feasible_sampleSet = sampleSet.filter(lambda row: row.is_feasible)
    if len(feasible_sampleSet) > 0:
        print("FIRST FEASIBLE SAMPLE: ")
        print(feasible_sampleSet.first.sample)
        for key in feasible_sampleSet.first.sample.keys():
            weights[variable_level][key] = feasible_sampleSet.first.sample[key] # weights[variable_level][key] + lr * (feasible_sampleSet.first.sample[key] - weights[variable_level][key])
        print("weights:")
        for key in weights:
            print(weights[key])
        print('')
    else:
        print('feasible sample set not found')
        break
    #############################################################################################################################
    # for key in weights[variable_level].keys():
    #     weights[variable_level][key] = weights[variable_level][key] + lr * (
    #                 random.uniform(-1, 1) - weights[variable_level][key])
    # print("weights: ")
    # for key in weights:
    #     print(weights[key])
    # print('')
    #############################################################################################################################

    W1 = np.array([[weights['level1']['w11'], weights['level1']['w12']],
                   [weights['level1']['w21'], weights['level1']['w22']],
                   [weights['level1']['b1'], weights['level1']['b2']]])
    W2 = np.array([[weights['level2']['w31']],
                   [weights['level2']['w41']],
                   [weights['level2']['b3']]])
    sq_error = 0.0
    for i in range(X.shape[1]):
        inputs = np.array([[X[0][i], X[1][i], 1.0]])
        hidden_sum = np.matmul(inputs, W1)
        hidden_act = hidden_sum # 1 / (1 + np.exp(-hidden_sum))
        hidden_out = np.matmul(hidden_act, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) + [0.0, 0.0, 1.0]
        output_sum = np.matmul(hidden_out, W2)
        outputs = output_sum # 1 / (1 + np.exp(-output_sum))
        print('input: ', inputs[0][:2], 'output: ', Y[0][i], 'prediction: ', outputs[0][0])
        sq_error = sq_error + (Y[0][i] - outputs[0][0])*(Y[0][i] - outputs[0][0])
        # log_probs = log_probs + Y[0][i]*np.log(outputs[0][0]) + (1 - Y[0][i])*np.log(1 - outputs[0][0])
    print('RMSE: ', np.sqrt(sq_error/X.shape[1]))
    print('')
