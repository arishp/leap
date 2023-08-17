import dimod
from numpy import random
import numpy as np
from dwave.system import LeapHybridCQMSampler

X = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])  # XOR input
Y = np.array([[0.0, 1.0, 1.0, 0.0]])  # XOR output
weights = {'level1': {'w11': 0.0, 'w21': 0.0, 'b1': 0.0, 'w12': 0.0, 'w22': 0.0, 'b2': 0.0, 'w13': 0.0, 'w23': 0.0, 'b3': 0.0},
           'level2': {'w31': 0.0, 'w41': 0.0, 'w51': 0.0, 'b4': 0.0}}
for level in weights.keys():
    for key in weights[level].keys():
        weights[level][key] = random.uniform(-1.0, 1.0)
print('INITIAL WEIGHTS: ')
for key in weights:
    print(weights[key])

W1 = np.array([[weights['level1']['w11'], weights['level1']['w12'], weights['level1']['w13']],
    [weights['level1']['w21'], weights['level1']['w22'], weights['level1']['w23']],
    [weights['level1']['b1'], weights['level1']['b2'], weights['level1']['b3']]])
W2 = np.array([[weights['level2']['w31']],
    [weights['level2']['w41']],
    [weights['level2']['w51']],
    [weights['level2']['b4']]])
sq_error = 0.0
for i in range(X.shape[1]):
    inputs = np.array([[X[0][i], X[1][i], 1.0]])
    hidden_sum = np.matmul(inputs, W1)
    hidden_out = np.matmul(hidden_sum, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]) + [0.0, 0.0, 0.0, 1.0]
    outputs = np.matmul(hidden_out, W2)
    print('input: ', inputs[0][:2], 'output: ', Y[0][i], 'prediction: ', outputs[0][0])
    sq_error = sq_error + (Y[0][i] - outputs[0][0])*(Y[0][i] - outputs[0][0])
print('RMSE: ', np.sqrt(sq_error/X.shape[1]))

num_iterations = 1
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
    obj_fun = 0.0
    for data in range(X.shape[1]):
        input_1 = X[0][data]
        input_2 = X[1][data]
        output_y = Y[0][data]
        obj_fun = obj_fun + (input_1*weights[constant_level]['w11'] + input_2*weights[constant_level]['w21'] + weights[constant_level]['b1'])*quadratic_weights[0] + \
            (input_1 * weights[constant_level]['w12'] + input_2 * weights[constant_level]['w22'] + weights[constant_level]['b2']) * quadratic_weights[1] + \
            (input_1 * weights[constant_level]['w13'] + input_2 * weights[constant_level]['w23'] + weights[constant_level]['b3']) * quadratic_weights[2] + \
            quadratic_weights[3] - output_y
        cqm.add_constraint((input_1*weights[constant_level]['w11'] + input_2*weights[constant_level]['w21'] + weights[constant_level]['b1'])*quadratic_weights[0] + \
            (input_1 * weights[constant_level]['w12'] + input_2 * weights[constant_level]['w22'] + weights[constant_level]['b2']) * quadratic_weights[1] + \
            (input_1 * weights[constant_level]['w13'] + input_2 * weights[constant_level]['w23'] + weights[constant_level]['b3']) * quadratic_weights[2] + \
            quadratic_weights[3] - output_y >= 0.0, label="C"+str(data+1))
    cqm.set_objective(obj_fun)
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

    W1 = np.array([[weights['level1']['w11'], weights['level1']['w12'], weights['level1']['w13']],
                   [weights['level1']['w21'], weights['level1']['w22'], weights['level1']['w23']],
                   [weights['level1']['b1'], weights['level1']['b2'], weights['level1']['b3']]])
    W2 = np.array([[weights['level2']['w31']],
                   [weights['level2']['w41']],
                   [weights['level2']['w51']],
                   [weights['level2']['b4']]])
    sq_error = 0.0
    for i in range(X.shape[1]):
        inputs = np.array([[X[0][i], X[1][i], 1.0]])
        hidden_sum = np.matmul(inputs, W1)
        hidden_out = np.matmul(hidden_sum, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]) + [0.0, 0.0, 0.0, 1.0]
        outputs = np.matmul(hidden_out, W2)
        print('input: ', inputs[0][:2], 'output: ', Y[0][i], 'prediction: ', outputs[0][0])
        sq_error = sq_error + (Y[0][i] - outputs[0][0])*(Y[0][i] - outputs[0][0])
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
    obj_fun = 0.0
    for data in range(X.shape[1]):
        input_1 = X[0][data]
        input_2 = X[1][data]
        output_y = Y[0][data]
        obj_fun = obj_fun + (input_1*quadratic_weights[0] + input_2*quadratic_weights[1] + quadratic_weights[2])*weights[constant_level]['w31'] + \
            (input_1 * quadratic_weights[3] + input_2 * quadratic_weights[4] + quadratic_weights[5]) * weights[constant_level]['w41'] + \
            (input_1 * quadratic_weights[6] + input_2 * quadratic_weights[7] + quadratic_weights[8]) *weights[constant_level]['w51'] + \
            weights[constant_level]['b4'] - output_y
        cqm.add_constraint((input_1*quadratic_weights[0] + input_2*quadratic_weights[1] + quadratic_weights[2])*weights[constant_level]['w31'] + \
            (input_1 * quadratic_weights[3] + input_2 * quadratic_weights[4] + quadratic_weights[5]) * weights[constant_level]['w41'] + \
            (input_1 * quadratic_weights[6] + input_2 * quadratic_weights[7] + quadratic_weights[8]) *weights[constant_level]['w51'] + \
            weights[constant_level]['b4'] - output_y >= 0.0, label="C"+str(data+1))
    cqm.set_objective(obj_fun)
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
    W1 = np.array([[weights['level1']['w11'], weights['level1']['w12'], weights['level1']['w13']],
                   [weights['level1']['w21'], weights['level1']['w22'], weights['level1']['w23']],
                   [weights['level1']['b1'], weights['level1']['b2'], weights['level1']['b3']]])
    W2 = np.array([[weights['level2']['w31']],
                   [weights['level2']['w41']],
                   [weights['level2']['w51']],
                   [weights['level2']['b4']]])
    sq_error = 0.0
    for i in range(X.shape[1]):
        inputs = np.array([[X[0][i], X[1][i], 1.0]])
        hidden_sum = np.matmul(inputs, W1)
        hidden_out = np.matmul(hidden_sum, [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]) + [0.0, 0.0, 0.0, 1.0]
        outputs = np.matmul(hidden_out, W2)
        print('input: ', inputs[0][:2], 'output: ', Y[0][i], 'prediction: ', outputs[0][0])
        sq_error = sq_error + (Y[0][i] - outputs[0][0])*(Y[0][i] - outputs[0][0])
    print('RMSE: ', np.sqrt(sq_error/X.shape[1]))
    print('')
