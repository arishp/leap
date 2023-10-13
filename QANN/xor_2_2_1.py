import dimod
import copy
from numpy import random
import numpy as np
from dwave.system import LeapHybridCQMSampler
import time

X = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])  # XOR input
Y = np.array([[0.0, 1.0, 1.0, 0.0]])  # XOR output
w = {'level1': {'w11': 0.0, 'w21': 0.0, 'b1': 0.0, 'w12': 0.0, 'w22': 0.0, 'b2': 0.0},
     'level2': {'w31': 0.0, 'w41': 0.0, 'b3': 0.0}}
n_data = X.shape[1]


def run_sampling(cqm, w, variable_level):
    opt_w = copy.deepcopy(w)
    sampler = LeapHybridCQMSampler()
    start = time.time()
    sample_set = sampler.sample_cqm(cqm)
    end = time.time()
    feasible_sample_set = sample_set.filter(lambda row: row.is_feasible)
    print('sampling time: ', (end - start)*1000.0)
    if len(feasible_sample_set) > 0:
        for key in feasible_sample_set.first.sample.keys():
            opt_w[variable_level][key] = feasible_sample_set.first.sample[key]
    return opt_w


def run_ann_mae(w):
    W1 = np.array([[w['level1']['w11'], w['level1']['w12'], 0.0],
                   [w['level1']['w21'], w['level1']['w22'], 0.0],
                   [w['level1']['b1'], w['level1']['b2'], 1.0]])
    W2 = np.array([[w['level2']['w31']],
                   [w['level2']['w41']],
                   [w['level2']['b3']]])
    abs_error = 0.0
    print("input_1\tinput_2\toutput\tprediction")
    for i in range(X.shape[1]):
        inputs = np.array([[X[0][i], X[1][i], 1.0]])
        hidden_1 = np.matmul(inputs, W1)
        outputs = np.matmul(hidden_1, W2)
        print(inputs[0][0], '\t', inputs[0][1], '\t', Y[0][i], '\t', outputs[0][0])
        abs_error += np.abs(Y[0][i] - outputs[0][0])
    print("mean absolute error: ", abs_error/n_data)


def print_weights(w):
    for level in w.keys():
        print(level, ': ', w[level])


def random_initialization():
    for level in w.keys():
        for key in w[level].keys():
            w[level][key] = random.randn()*0.01


def main():
    random_initialization()
    print_weights(w)
    run_ann_mae(w)

    print('PHASE 1:')
    variable_level = 'level2'
    qw = [dimod.Real(key) for key in w[variable_level].keys()]
    for i, key in enumerate(w[variable_level].keys()):
        qw[i].set_lower_bound(key, -1e+29)
    cqm = dimod.ConstrainedQuadraticModel()
    obj_fun = 0.0
    for data in range(X.shape[1]):
        input_1 = X[0][data]
        input_2 = X[1][data]
        output_y = Y[0][data]
        obj_fun += ( (input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[0] + \
            (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[1] + qw[2] - output_y )
        cqm.add_constraint( ((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[0] + \
            (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[1] + qw[2] - output_y) >= 0.0, label='C' + str(data + 1) )
    cqm.set_objective(obj_fun)
    opt_w = run_sampling(cqm, w, variable_level)
    print_weights(opt_w)
    run_ann_mae(opt_w)

    print('PHASE 2:')
    variable_level = 'level1'
    qw = [dimod.Real(key) for key in w[variable_level].keys()]
    for i, key in enumerate(w[variable_level].keys()):
        qw[i].set_lower_bound(key, -1e+29)
    cqm = dimod.ConstrainedQuadraticModel()
    obj_fun = 0.0
    for data in range(X.shape[1]):
        input_1 = X[0][data]
        input_2 = X[1][data]
        output_y = Y[0][data]
        obj_fun += ( (input_1 * qw[0] + input_2 * qw[1] + qw[2]) * opt_w['level2']['w31'] + \
            (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * opt_w['level2']['w41'] + opt_w['level2']['b3'] - output_y )
        cqm.add_constraint( ((input_1 * qw[0] + input_2 * qw[1] + qw[2]) * opt_w['level2']['w31'] + \
            (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * opt_w['level2']['w41'] + opt_w['level2']['b3'] - output_y) >= 0.0, label='C'+ str(data + 1) )
    cqm.set_objective(obj_fun)
    opt_w = run_sampling(cqm, opt_w, variable_level)
    print_weights(opt_w)
    run_ann_mae(opt_w)


if __name__ == "__main__":
    main()
