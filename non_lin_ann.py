import dimod
import copy
from numpy import random
import numpy as np
from dwave.system import LeapHybridCQMSampler


def print_weights(w):
    for level in w.keys():
        print(level, ': ', w[level])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def run_ann(X, Y, w):
    W1 = np.array([[w['level1']['w11'], w['level1']['w12'], 0.0],
                   [w['level1']['w21'], w['level1']['w22'], 0.0],
                   [w['level1']['b1'], w['level1']['b2'], 1.0]])
    W2 = np.array([[w['level2']['w31']],
                   [w['level2']['w41']],
                   [w['level2']['b3']]])
    print_weights(w)
    sq_error = 0.0
    print("input_1\tinput_2\toutput\tprediction")
    for i in range(X.shape[1]):
        inputs = np.array([[X[0][i], X[1][i], 1.0]])
        hidden_1 = np.matmul(inputs, W1)
        sig_hidden_1 = sigmoid(hidden_1)
        outputs = np.matmul(sig_hidden_1, W2)
        sig_output = sigmoid(outputs)
        print(inputs[0][0], '\t', inputs[0][1], '\t', Y[0][i], '\t', sig_output[0][0])
        sq_error = sq_error + (Y[0][i] - sig_output[0][0]) * (Y[0][i] - sig_output[0][0])
    return np.sqrt(sq_error / X.shape[1])


def run_sampling(cqm, w, variable_level):
    opt_w = copy.deepcopy(w)
    sampler = LeapHybridCQMSampler()
    sample_set = sampler.sample_cqm(cqm)
    feasible_sample_set = sample_set.filter(lambda row: row.is_feasible)
    if len(feasible_sample_set) > 0:
        for key in feasible_sample_set.first.sample.keys():
            opt_w[variable_level][key] = feasible_sample_set.first.sample[key]
    else:
        print('feasible sample set not found')
    return opt_w


def update_weights(X, Y, cqm, w, variable_level):
    lr = 0.001
    new_w = copy.deepcopy(w)
    opt_w = run_sampling(cqm, w, variable_level)
    error_old = run_ann(X, Y, w)
    print('old error: ', error_old)
    error_opt_w = run_ann(X, Y, opt_w)
    print('opt error: ', error_opt_w)
    error_diff = error_opt_w - error_old
    if error_diff != 0:
        for key in w[variable_level].keys():
            slope = (opt_w[variable_level][key] - w[variable_level][key])/error_diff
            new_w[variable_level][key] = w[variable_level][key] - slope*lr
        if error_opt_w < error_old:
            print('optimal is better')
        else:
            print('old weights better')
    else:
        print('no error difference')
    return new_w


def he_et_al_initialization(w):
    random_weights = []
    len_level_minus_1 = 0
    for level in w.keys():
        if len_level_minus_1 == 0:
            random_weights.extend(np.random.randn(len(w[level])))
        else:
            random_weights.extend(np.random.randn(len(w[level])) * np.sqrt(2.0 / len_level_minus_1))
        len_level_minus_1 = len(w[level])
    count = 0
    for level in w.keys():
        for key in w[level].keys():
            w[level][key] = random_weights[count]
            count += 1
    return w


def main():
    X = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])  # XOR input
    Y = np.array([[0.0, 1.0, 1.0, 0.0]])  # XOR output
    w = {'level1': {'w11': 0.0, 'w21': 0.0, 'b1': 0.0, 'w12': 0.0, 'w22': 0.0, 'b2': 0.0},
         'level2': {'w31': 0.0, 'w41': 0.0, 'b3': 0.0}}
    w = he_et_al_initialization(w)

    num_iterations = 2
    for iteration in range(num_iterations):
        print('BACK PROPAGATION ITERATION - ', iteration + 1, ':')
        ######### BACK PROPAGATION - PHASE 1 ##########
        print('PHASE 1:')
        variable_level = 'level2'
        qw = [dimod.Real(key) for key in w[variable_level].keys()]
        for i, key in enumerate(w[variable_level].keys()):
            qw[i].set_upper_bound(key, 100.0)
            qw[i].set_lower_bound(key, -100.0)
            # qw[i].set_lower_bound(key, -1e+29)
        cqm = dimod.ConstrainedQuadraticModel()
        obj_fun = 0.0
        for data in range(X.shape[1]):
            input_1 = X[0][data]
            input_2 = X[1][data]
            output_y = Y[0][data]
            obj_fun = obj_fun + (input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[
                0] + \
                      (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[1] + \
                      qw[2] - output_y
            cqm.add_constraint(
                ((input_1 * w['level1']['w11'] + input_2 * w['level1']['w21'] + w['level1']['b1']) * qw[0] + \
                 (input_1 * w['level1']['w12'] + input_2 * w['level1']['w22'] + w['level1']['b2']) * qw[1] + \
                 qw[2] - output_y) >= 0.0, label='C' + str(data + 1))
        cqm.set_objective(obj_fun)
        w = update_weights(X, Y, cqm, w, variable_level)
        print('new error: ', run_ann(X, Y, w), '\n')

        ######### BACK PROPAGATION - PHASE 2 ##########
        print('PHASE 2:')
        variable_level = 'level1'
        qw = [dimod.Real(key) for key in w[variable_level].keys()]
        for i, key in enumerate(w[variable_level].keys()):
            qw[i].set_upper_bound(key, 100.0)
            qw[i].set_lower_bound(key, -100.0)
            # qw[i].set_lower_bound(key, -1e+29)
        cqm = dimod.ConstrainedQuadraticModel()
        obj_fun = 0.0
        for data in range(X.shape[1]):
            input_1 = X[0][data]
            input_2 = X[1][data]
            output_y = Y[0][data]
            obj_fun = obj_fun + (input_1 * qw[0] + input_2 * qw[1] + qw[2]) * w['level2']['w31'] + \
                      (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * w['level2']['w41'] + w['level2']['b3'] - output_y
            cqm.add_constraint(((input_1 * qw[0] + input_2 * qw[1] + qw[2]) * w['level2']['w31'] + \
                                (input_1 * qw[3] + input_2 * qw[4] + qw[5]) * w['level2']['w41'] + \
                                w['level2']['b3'] - output_y) >= 0.0, label='C' + str(data + 1))
        cqm.set_objective(obj_fun)
        w = update_weights(X, Y, cqm, w, variable_level)
        print('new error: ', run_ann(X, Y, w), '\n')


if __name__ == "__main__":
    main()
