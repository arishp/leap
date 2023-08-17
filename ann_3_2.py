import dimod
from numpy import random
import numpy as np
from dwave.system import LeapHybridCQMSampler


def run_ann(X, Y, w):
    W1 = np.array([ [w['level1']['w11'], w['level1']['w12'], 0.0],
                    [w['level1']['w21'], w['level1']['w22'], 0.0],
                    [w['level1']['b1'],  w['level1']['b2'],  1.0] ])
    W2 = np.array([ [w['level2']['w31'], w['level2']['w32'], 0.0],
                    [w['level2']['w41'], w['level2']['w42'], 0.0],
                    [w['level2']['b3'],  w['level2']['b4'],  1.0]])
    W3 = np.array([ [w['level3']['w51']],
                    [w['level3']['w61']],
                    [w['level3']['b5']] ])
    sq_error = 0.0
    print("input_1\tinput_2\toutput\tprediction")
    for i in range(X.shape[1]):
        inputs = np.array([[X[0][i], X[1][i], 1.0]])
        hidden_1 = np.matmul(inputs, W1)
        hidden_2 = np.matmul(hidden_1, W2)
        outputs = np.matmul(hidden_2, W3)
        print(inputs[0][0], '\t', inputs[0][1], '\t', Y[0][i], '\t', outputs[0][0])
        sq_error = sq_error + (Y[0][i] - outputs[0][0]) * (Y[0][i] - outputs[0][0])
    print('RMSE: ', np.sqrt(sq_error / X.shape[1]), '\n')

def run_sampling(cqm, w, variable_level):
    sampler = LeapHybridCQMSampler()
    sampleSet = sampler.sample_cqm(cqm)
    feasible_sampleSet = sampleSet.filter(lambda row: row.is_feasible)
    if len(feasible_sampleSet) > 0:
        print("FIRST FEASIBLE SAMPLE: ", variable_level)
        print(feasible_sampleSet.first.sample)
        for key in feasible_sampleSet.first.sample.keys():
            w[variable_level][key] = feasible_sampleSet.first.sample[key]
        print("weights:")
        for level in w.keys():
            print(level, ': ', w[level])
        print('')
    else:
        print('feasible sample set not found')
    return w


X = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])  # XOR input
Y = np.array([[0.0, 1.0, 1.0, 0.0]])  # XOR output
w = {'level1': {'w11': 0.0, 'w21': 0.0, 'b1': 0.0, 'w12': 0.0, 'w22': 0.0, 'b2': 0.0},
     'level2': {'w31': 0.0, 'w41': 0.0, 'b3': 0.0, 'w32': 0.0, 'w42': 0.0, 'b4': 0.0},
     'level3': {'w51': 0.0, 'w61': 0.0, 'b5': 0.0}}
for level in w.keys():
    for key in w[level].keys():
        w[level][key] = random.uniform(-100.0, 100.0)
print('INITIAL WEIGHTS: ')
for level in w.keys():
    print(level, ': ', w[level])
run_ann(X, Y, w)
num_iterations = 1
for iteration in range(num_iterations):
    print('BACK PROPAGATION ITERATION - ', iteration + 1, ':')
    ######### BACK PROPAGATION - PHASE 1 ##########
    print('PHASE 1:')
    variable_level = 'level3'
    qw = [dimod.Real(key) for key in w[variable_level].keys()]
    for i in range(len(qw)):
        print(qw[i])
    for i, key in enumerate(w[variable_level].keys()):
        #qw[i].set_upper_bound(key, 100.0)
        #qw[i].set_lower_bound(key, -100.0)
        qw[i].set_lower_bound(key, -1e+29)
    cqm = dimod.ConstrainedQuadraticModel()
    obj_fun = 0.0
    for data in range(X.shape[1]):
        input_1 = X[0][data]
        input_2 = X[1][data]
        output_y = Y[0][data]
        obj_fun = obj_fun + ((input_1*w['level1']['w11']+input_2*w['level1']['w21']+w['level1']['b1'])*w['level2']['w31'] + \
            (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2'])*w['level2']['w41'] + w['level2']['b3'])*qw[0] + \
            ((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * w['level2']['w32'] + \
                (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * w['level2']['w42'] + w['level2']['b4'])*qw[1] + \
            qw[2] - output_y
        cqm.add_constraint((((input_1*w['level1']['w11']+input_2*w['level1']['w21']+w['level1']['b1'])*w['level2']['w31'] + \
            (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2'])*w['level2']['w41'] + w['level2']['b3'])*qw[0] + \
            ((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * w['level2']['w32'] + \
                (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * w['level2']['w42'] + w['level2']['b4'])*qw[1] + \
            qw[2] - output_y) >= 0.0, label='C'+str(data + 1))
    cqm.set_objective(obj_fun)
    print('\n', cqm)
    w = run_sampling(cqm, w, variable_level)
    run_ann(X, Y, w)

    ######### BACK PROPAGATION - PHASE 2 ##########
    print('PHASE 2:')
    variable_level = 'level2'
    qw = [dimod.Real(key) for key in w[variable_level].keys()]
    for i in range(len(qw)):
        print(qw[i])
    for i, key in enumerate(w[variable_level].keys()):
        #qw[i].set_upper_bound(key, 100.0)
        #qw[i].set_lower_bound(key, -100.0)
        qw[i].set_lower_bound(key, -1e+29)
    cqm = dimod.ConstrainedQuadraticModel()
    obj_fun = 0.0
    for data in range(X.shape[1]):
        input_1 = X[0][data]
        input_2 = X[1][data]
        output_y = Y[0][data]
        obj_fun = obj_fun + ((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[0] + \
            (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[1] + qw[2]) * w['level3']['w51'] + \
            ((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[3] + \
                (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[4] + qw[5]) * w['level3']['w61'] + \
            w['level3']['b5'] - output_y
        cqm.add_constraint((((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[0] + \
            (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[1] + qw[2]) * w['level3']['w51'] + \
            ((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[3] + \
                (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[4] + qw[5]) * w['level3']['w61'] + \
            w['level3']['b5'] - output_y) >= 0.0, label='C' + str(data + 1))
    cqm.set_objective(obj_fun)
    print('\n', cqm)
    w = run_sampling(cqm, w, variable_level)
    run_ann(X, Y, w)

######### BACK PROPAGATION - PHASE 3 ##########
    print('PHASE 3:')
    variable_level = 'level1'
    qw = [dimod.Real(key) for key in w[variable_level].keys()]
    for i in range(len(qw)):
        print(qw[i])
    for i, key in enumerate(w[variable_level].keys()):
        #qw[i].set_upper_bound(key, 100.0)
        #qw[i].set_lower_bound(key, -100.0)
        qw[i].set_lower_bound(key, -1e+29)
    cqm = dimod.ConstrainedQuadraticModel()
    obj_fun = 0.0
    for data in range(X.shape[1]):
        input_1 = X[0][data]
        input_2 = X[1][data]
        output_y = Y[0][data]
        obj_fun = obj_fun + ((input_1 * qw[0] + input_2 * qw[1] + qw[2]) * w['level2']['w31'] + \
            (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * w['level2']['w41'] + w['level2']['b3']) * w['level3']['w51'] + \
            ((input_1 * qw[0] + input_2 * qw[1] + qw[2]) * w['level2']['w32'] + \
                (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * w['level2']['w42'] + w['level2']['b4']) * w['level3']['w61'] + \
            w['level3']['b5'] - output_y
        cqm.add_constraint((((input_1 * qw[0] + input_2 * qw[1] + qw[2]) * w['level2']['w31'] + \
            (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * w['level2']['w41'] + w['level2']['b3']) * w['level3']['w51'] + \
            ((input_1 * qw[0] + input_2 * qw[1] + qw[2]) * w['level2']['w32'] + \
                (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * w['level2']['w42'] + w['level2']['b4']) * w['level3']['w61'] + \
            w['level3']['b5'] - output_y) >= 0.0, label='C' + str(data + 1))
    cqm.set_objective(obj_fun)
    print('\n', cqm)
    w = run_sampling(cqm, w, variable_level)
    run_ann(X, Y, w)

