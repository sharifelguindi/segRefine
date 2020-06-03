import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from functions import store_arrays_hdf5, distance_map
from scipy.optimize import minimize, basinhopping
import random
from timeit import default_timer as timer
import SimpleITK as sitk
import cv2
import pickle


def getMaskFromDistmap(roiSize, redSize, spacing_mm, G):
    G_reshaped = np.reshape(G, (redSize))
    G_resized = np.zeros(roiSize)
    for i in range(0, redSize[0]):
        G_resized[i, :, :] = cv2.resize(G_reshaped[i, :, :], dsize=(roiSize[1], roiSize[2]),
                                        interpolation=cv2.INTER_CUBIC)

    G_resized[G_resized <= 0] = 255
    G_resized[G_resized != 255] = 0
    G_resized[G_resized == 255] = 1

    return G_resized


def get_affine_matrix(x):
    R_x = np.zeros((4, 4))
    R_y = np.zeros((4, 4))
    R_z = np.zeros((4, 4))

    R_x[0, 0:4] = [1, 0, 0, 0]
    R_x[1, 0:4] = [0, np.cos(x[0]), (-1) * np.sin(x[0]), 0]
    R_x[2, 0:4] = [0, np.sin(x[0]), np.cos(x[0]), 0]
    R_x[3, 0:4] = [0, 0, 0, 1]

    R_y[0, 0:4] = [np.cos(x[1]), 0, np.sin(x[1]), 0]
    R_y[1, 0:4] = [0, 1, 0, 0]
    R_y[2, 0:4] = [(-1) * np.sin(x[1]), 0, np.cos(x[1]), 0]
    R_y[3, 0:4] = [0, 0, 0, 1]

    R_z[0, 0:4] = [np.cos(x[2]), (-1) * np.sin(x[2]), 0, 0]
    R_z[1, 0:4] = [np.sin(x[2]), np.cos(x[2]), 0, 0]
    R_z[2, 0:4] = [0, 0, 1, 0]
    R_z[3, 0:4] = [0, 0, 0, 1]

    S = np.zeros((4, 4))
    np.fill_diagonal(S, x[3])
    S[3, 3] = 1

    T = np.zeros((4, 4))
    np.fill_diagonal(T, 1)
    T[0, 3] = x[4]
    T[1, 3] = x[5]
    T[2, 3] = x[6]

    R = np.dot(R_x, np.dot(R_y, R_z))

    return R, S, T


def get_inverse_affine(R, S, T):
    T_inv = np.copy(T)
    T_inv[0:3, 3] = T_inv[0:3, 3] * (-1)

    R_inv = np.copy(R)
    R_inv[0:3, 0:3] = R_inv[0:3, 0:3].T

    S_inv = np.copy(S)
    S_inv[0, 0] = 1 / S_inv[0, 0]
    S_inv[1, 1] = 1 / S_inv[1, 1]
    S_inv[2, 2] = 1 / S_inv[2, 2]

    return R_inv, S_inv, T_inv


def apply_invAffineTransform(R, S, T, g):
    coord = np.ones((4, 1))
    g_prime = np.copy(g)
    for n in np.arange(0, np.shape(g)[1], 3):
        coord[0:3, 0] = g[0, n:n + 3]
        coord_1 = np.dot(T, coord)
        coord_2 = np.dot(R, coord_1)
        coord_3 = np.dot(S, coord_2)
        g_prime[0, n:n + 3] = coord_3[0:3, 0]

    return g_prime


def apply_affineTransform(R, S, T, g):
    coord = np.ones((4, 1))
    g_prime = np.copy(g)
    for n in np.arange(0, np.shape(g)[1], 3):
        coord[0:3, 0] = g[0, n:n + 3]
        coord_1 = np.dot(S, coord)
        coord_2 = np.dot(R, coord_1)
        coord_3 = np.dot(T, coord_2)
        g_prime[0, n:n + 3] = coord_3[0:3, 0]

    return g_prime


def procrustes_distance(mtx1, mtx2):
    distance = np.sum(
        np.sqrt((mtx1[:, 0] - mtx2[:, 0]) ** 2 + (mtx1[:, 1] - mtx2[:, 1]) ** 2 + (mtx1[:, 2] - mtx2[:, 2]) ** 2))
    return distance


def generalized_procrustes_analysis(G_cen):
    # Initialize data arrays based on input training set
    G_cen_aligned = np.zeros(np.shape(G_cen))
    shapes = np.zeros((np.shape(G_cen)[0], int(np.shape(G_cen)[1] / 3), 3))
    shapes_aligned = np.zeros((np.shape(G_cen)[0], int(np.shape(G_cen)[1] / 3), 3))

    # Convert training set for procrustest analysis
    for i in range(0, np.shape(G_cen)[0]):
        shapes[i, :, :] = np.reshape(G_cen[i, :], (-1, 3))

    # initialize Procrustest distance
    current_distance = 0

    # Initialize a mean shape (first element), with zero mean and Frobenius norm
    mean_shape = shapes[0, :, :]
    mean_shape -= np.mean(mean_shape, 0)
    norm1 = np.linalg.norm(mean_shape)
    mean_shape /= norm1
    num_shapes = len(shapes)

    while True:

        shapes_aligned[0, :, :] = mean_shape

        for sh in range(1, num_shapes):
            shapes_aligned[0, :, :], shapes_aligned[sh, :, :], _ = procrustes(mean_shape, shapes[sh, :, :])

        new_mean = np.mean(shapes_aligned, axis=0)

        new_distance = procrustes_distance(new_mean, mean_shape)

        if new_distance == current_distance:
            break

        _, new_mean, _ = procrustes(mean_shape, new_mean)
        mean_shape = new_mean
        current_distance = new_distance

    for i in range(0, np.shape(G_cen)[0]):
        G_cen_aligned[i, :] = shapes_aligned[i, :, :].flatten()
    return shapes_aligned, G_cen_aligned


def plt_centroids(G_m, markerSize, colorMap):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0
    color = np.zeros((1, 3))
    for n in np.arange(0, np.shape(G_m)[1], 3):
        color[0, :] = colorMap[i]
        xs = G_m[:, n]
        ys = G_m[:, n + 1]
        zs = G_m[:, n + 2]
        ax.scatter(xs, ys, zs, marker='o', s=markerSize, c=color)
        i = i + 1

    return fig, ax


def plt_centroids_compare(G_m, G_n, markerSize, colorMap):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0
    color = np.zeros((1, 3))
    for n in np.arange(0, np.shape(G_m)[1], 3):
        color[0, :] = colorMap[i]
        xs = G_m[:, n]
        ys = G_m[:, n + 1]
        zs = G_m[:, n + 2]
        ax.scatter(xs, ys, zs, marker='o', s=markerSize, c=color)
        i = i + 1

    i = 0
    color = np.zeros((1, 3))
    for n in np.arange(0, np.shape(G_n)[1], 3):
        color[0, :] = colorMap[i]
        xs = G_n[:, n]
        ys = G_n[:, n + 1]
        zs = G_n[:, n + 2]
        ax.scatter(xs, ys, zs, marker='s', s=markerSize, c=color)
        i = i + 1

    return fig, ax


def get_int_list(start, stop, size):
    result = []
    unique_set = set()
    for i in range(size):
        x = int(round(random.uniform(start, stop), 0))
        while x in unique_set:
            x = int(round(random.uniform(start, stop), 0))
        unique_set.add(x)
        result.append(x)

    return result


def generate_random_shape(g_sample, stdDev, maxIncorrect, incorrect_range, correct_range):
    g_new = np.zeros(np.shape(g_sample))
    L_max = int(len(g_sample) / 256)
    L = int(random.uniform(0, maxIncorrect))
    correct = list(range(0, L_max))
    incorrect = get_int_list(0, L_max - 1, L)
    for sample in incorrect:
        correct.remove(sample)

    for n in incorrect:
        coord = g_sample[(256 * n):(256 * n + 256)]
        coord_new = np.zeros((np.shape(coord)))
        sd = stdDev[(256 * n):(256 * n + 256)]
        k = 0
        sign = random.uniform(-1, 1)
        sign = sign / np.absolute(sign)
        for s in sd:
            coord_new[k] = coord[k] + (sign * random.uniform(incorrect_range[0], incorrect_range[1]) * s)
            k = k + 1
        g_new[(256 * n):(256 * n + 256)] = coord_new

    for n in correct:
        coord = g_sample[(256 * n):(256 * n + 256)]
        coord_new = np.zeros((np.shape(coord)))
        sd = stdDev[(256 * n):(256 * n + 256)]
        k = 0
        sign = random.uniform(-1, 1)
        sign = sign / np.absolute(sign)
        for s in sd:
            coord_new[k] = coord[k] + (sign * random.uniform(correct_range[0], correct_range[1]) * s)
            k = k + 1
        g_new[(256 * n):(256 * n + 256)] = coord_new

    return g_new, correct, incorrect


def generate_random_sample(g_sample, stdDev, maxIncorrect, incorrect_range, correct_range):
    g_new = np.zeros(np.shape(g_sample))
    L_max = int(len(g_sample) / 3)
    L = int(random.uniform(0, maxIncorrect))
    correct = list(range(0, L_max))
    incorrect = get_int_list(0, L_max - 1, L)
    for sample in incorrect:
        correct.remove(sample)

    for n in incorrect:
        coord = g_sample[(3 * n):(3 * n + 3)]
        coord_new = np.zeros((np.shape(coord)))
        sd = stdDev[(3 * n):(3 * n + 3)]
        k = 0
        sign = random.uniform(-1, 1)
        sign = sign / np.absolute(sign)
        for s in sd:
            coord_new[k] = coord[k] + (sign * random.uniform(incorrect_range[0], incorrect_range[1]) * s)
            k = k + 1
        g_new[(3 * n):(3 * n + 3)] = coord_new

    for n in correct:
        coord = g_sample[(3 * n):(3 * n + 3)]
        coord_new = np.zeros((np.shape(coord)))
        sd = stdDev[(3 * n):(3 * n + 3)]
        k = 0
        sign = random.uniform(-1, 1)
        sign = sign / np.absolute(sign)
        for s in sd:
            coord_new[k] = coord[k] + (sign * random.uniform(correct_range[0], correct_range[1]) * s)
            k = k + 1
        g_new[(3 * n):(3 * n + 3)] = coord_new

    return g_new, correct, incorrect


def cost_function(W, G_ex, G_bar, R, S, T, E_vec, B_weights):
    G_com = G_bar + np.dot(B_weights, E_vec)
    G_com_RST = apply_affineTransform(R, S, T, G_com)
    delta_G = G_ex - G_com_RST
    func_val = np.dot(W, delta_G.T)
    cost_value = np.sum((np.sqrt(func_val ** 2)) / len(G_ex))

    return cost_value


def cost_minimiation_function_shape_noaffine(x, G_ex, G_bar, E_vec, E_var):
    for i in range(0, len(E_var)):
        if np.absolute(x[i]) > 2 * np.sqrt(E_var[i]):
            x[i] = (x[i] / np.absolute(x[i])) * 2 * np.sqrt(E_var[i])

    G_com = G_bar + np.dot(x, E_vec)
    delta_G = G_ex - G_com
    cost_value = np.sum((np.sqrt(delta_G ** 2)) / len(G_ex))

    return cost_value


def cost_minimization_function_shape(x, G_ex, G_bar, E_vec, E_var, redSize, roiSize, spacing_mm):
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((x[6], x[7], x[8]))

    affine_rot = sitk.Euler3DTransform()
    affine_rot.SetRotation(x[0], x[1], x[2])

    affine_scale = sitk.AffineTransform(3)
    affine_scale.Scale((x[3], x[4], x[5]))

    composite = sitk.Transform(3, sitk.sitkComposite)
    composite.AddTransform(affine_rot)
    composite.AddTransform(affine_scale)
    composite.AddTransform(translation)

    B_weights = np.copy(x[9:])
    print(B_weights)
    for i in range(0, len(E_var)):
        if np.absolute(B_weights[i]) > 2 * np.sqrt(E_var[i]):
            B_weights[i] = (B_weights[i] / np.absolute(B_weights[i])) * 2 * np.sqrt(E_var[i])

    G_com = G_bar + np.dot(B_weights, E_vec)

    G_com_reshaped = getMaskFromDistmap(roiSize, redSize, spacing_mm, G_com)
    print(np.max(G_com_reshaped))
    print(x)
    fixed = sitk.GetImageFromArray(G_com_reshaped.reshape(roiSize))
    fixed.SetSpacing(spacing_mm)
    fixed.SetOrigin([0, 0, 0])
    moved = sitk.Resample(fixed, composite)
    transformed_array = sitk.GetArrayFromImage(moved)
    print(np.max(transformed_array))
    G_translated, G_borders = distance_map(transformed_array, spacing_mm, roi_size=roiSize, crop_mask=False)
    G_com_translated = np.zeros(redSize)
    for i in range(0, redSize[0]):
        G_com_translated[i, :, :] = cv2.resize(G_translated[i, :, :], dsize=(16, 16), interpolation=cv2.INTER_CUBIC)

    delta_G = G_ex - G_com_translated.flatten()

    cost_value = np.sum((np.sqrt(delta_G ** 2)) / len(G_ex))

    return cost_value


def cost_minimization_function(x, G_ex, G_bar, E_vec, W, E_var):
    R_x = np.zeros((4, 4))
    R_y = np.zeros((4, 4))
    R_z = np.zeros((4, 4))

    R_x[0, 0:4] = [1, 0, 0, 0]
    R_x[1, 0:4] = [0, np.cos(x[0]), (-1) * np.sin(x[0]), 0]
    R_x[2, 0:4] = [0, np.sin(x[0]), np.cos(x[0]), 0]
    R_x[3, 0:4] = [0, 0, 0, 1]

    R_y[0, 0:4] = [np.cos(x[1]), 0, np.sin(x[1]), 0]
    R_y[1, 0:4] = [0, 1, 0, 0]
    R_y[2, 0:4] = [(-1) * np.sin(x[1]), 0, np.cos(x[1]), 0]
    R_y[3, 0:4] = [0, 0, 0, 1]

    R_z[0, 0:4] = [np.cos(x[2]), (-1) * np.sin(x[2]), 0, 0]
    R_z[1, 0:4] = [np.sin(x[2]), np.cos(x[2]), 0, 0]
    R_z[2, 0:4] = [0, 0, 1, 0]
    R_z[3, 0:4] = [0, 0, 0, 1]

    S = np.zeros((4, 4))
    np.fill_diagonal(S, x[3])
    S[3, 3] = 1

    T = np.zeros((4, 4))
    np.fill_diagonal(T, 1)
    T[0, 3] = x[4]
    T[1, 3] = x[5]
    T[2, 3] = x[6]

    R = np.dot(R_x, np.dot(R_y, R_z))

    B_weights = np.copy(np.reshape(x[7:], (len(E_var),)))

    for i in range(0, len(E_var)):
        if np.absolute(B_weights[i]) > 2 * np.sqrt(E_var[i]):
            B_weights[i] = (B_weights[i] / np.absolute(B_weights[i])) * 2 * np.sqrt(E_var[i])

    G_com = G_bar + np.dot(B_weights, E_vec)
    G_com_RST = apply_affineTransform(R, S, T, G_com)
    delta_G = G_ex - G_com_RST
    func_val = np.dot(W, delta_G.T)
    cost_value = np.sum((np.sqrt(func_val ** 2)) / len(G_ex))

    return cost_value


def iterate_theta(x, G_ex, G_bar, E_vec, B_weights, W):
    R_x = np.zeros((4, 4))
    R_y = np.zeros((4, 4))
    R_z = np.zeros((4, 4))

    R_x[0, 0:4] = [1, 0, 0, 0]
    R_x[1, 0:4] = [0, np.cos(x[0]), (-1) * np.sin(x[0]), 0]
    R_x[2, 0:4] = [0, np.sin(x[0]), np.cos(x[0]), 0]
    R_x[3, 0:4] = [0, 0, 0, 1]

    R_y[0, 0:4] = [np.cos(x[1]), 0, np.sin(x[1]), 0]
    R_y[1, 0:4] = [0, 1, 0, 0]
    R_y[2, 0:4] = [(-1) * np.sin(x[1]), 0, np.cos(x[1]), 0]
    R_y[3, 0:4] = [0, 0, 0, 1]

    R_z[0, 0:4] = [np.cos(x[2]), (-1) * np.sin(x[2]), 0, 0]
    R_z[1, 0:4] = [np.sin(x[2]), np.cos(x[2]), 0, 0]
    R_z[2, 0:4] = [0, 0, 1, 0]
    R_z[3, 0:4] = [0, 0, 0, 1]

    S = np.zeros((4, 4))
    np.fill_diagonal(S, x[3])
    S[3, 3] = 1

    T = np.zeros((4, 4))
    np.fill_diagonal(T, 1)
    T[0, 3] = x[4]
    T[1, 3] = x[5]
    T[2, 3] = x[6]

    R = np.dot(R_x, np.dot(R_y, R_z))

    T_inv = np.copy(T)
    T_inv[0:3, 3] = T_inv[0:3, 3] * (-1)

    R_inv = np.copy(R)
    R_inv[0:3, 0:3] = R_inv[0:3, 0:3].T

    S_inv = np.copy(S)
    S_inv[0, 0] = 1 / S_inv[0, 0]
    S_inv[1, 1] = 1 / S_inv[1, 1]
    S_inv[2, 2] = 1 / S_inv[2, 2]

    coord = np.ones((4, 1))
    g = np.copy(G_ex)
    g_prime = np.copy(g)
    for n in np.arange(0, np.shape(g)[1], 3):
        coord[0:3, 0] = g[0, n:n + 3]
        coord_1 = np.dot(T_inv, coord)
        coord_2 = np.dot(R_inv, coord_1)
        coord_3 = np.dot(S_inv, coord_2)
        g_prime[0, n:n + 3] = coord_3[0:3, 0]

    delta_G = g_prime - (G_bar + np.dot(B_weights, E_vec))
    arg = np.dot(W, delta_G.T)

    return np.sum((np.sqrt(arg ** 2)) / len(G_ex))


def iterate_weights(R_inv, S_inv, T_inv, G_ex, G_bar, E_vec, B_weights, E_var):
    G_ex_RST = apply_invAffineTransform(R_inv, S_inv, T_inv, G_ex)
    arg = (G_ex_RST - (G_bar + np.dot(B_weights, E_vec)))
    delta_B = np.dot(E_vec, arg.T)
    B_weights_updated = B_weights + delta_B.squeeze()
    for i in range(0, len(E_var)):
        if np.absolute(B_weights_updated[i]) > 2 * np.sqrt(E_var[i]):
            B_weights_updated[i] = (B_weights_updated[i] / np.absolute(B_weights_updated[i])) * 2 * np.sqrt(E_var[i])

    return B_weights_updated


def model_fit_process(W, G_ex, G_bar, x_init, E_vec, B_weights, E_var, method, max_steps, delta_stop):
    k = 0
    R, S, T = get_affine_matrix(x_init)
    cost_value = cost_function(W, G_ex, G_bar, R, S, T, E_vec, B_weights)
    x = np.copy(x_init)
    for i in range(0, max_steps):
        # results = basinhopping(iterate_theta, x, niter=10, minimizer_kwargs={"args": (G_ex, G_bar, E_vec, B_weights, W), "method": method})
        results = minimize(iterate_theta, x, args=(G_ex, G_bar, E_vec, B_weights, W), method=method)
        R_new, S_new, T_new = get_affine_matrix(results.x)
        R_inv_new, S_inv_new, T_inv_new = get_inverse_affine(R_new, S_new, T_new)
        B_weights_new = iterate_weights(R_inv_new, S_inv_new, T_inv_new, G_ex, G_bar, E_vec, B_weights, E_var)
        cost_value_new = cost_function(W, G_ex, G_bar, R_new, S_new, T_new, E_vec, B_weights_new)
        B_weights = np.copy(B_weights_new)
        x = np.copy(results.x)
        delta = np.absolute(cost_value - cost_value_new)
        cost_value = cost_value_new
        k = k + 1
    return cost_value, B_weights, x


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1

    return TP, FP, TN, FN


def generate_random_x(E_var):
    x = []
    for angle in range(0, 3):
        x.append(np.radians(random.uniform(0, 180)))

    x.append(random.uniform(0.1, 1.1))

    for shift in range(0, 3):
        x.append(random.uniform(-0.5, 0.5))

    for b in range(0, len(E_var)):
        x.append(random.uniform(-1 * np.sqrt(E_var[b]) * 2, 2 * np.sqrt(E_var[b])))

    return np.asarray(x)


def generate_random_shapeX(E_var):
    x = []
    for angle in range(0, 3):
        x.append(np.radians(random.uniform(0, 1)))

    for scale in range(0, 3):
        x.append(random.uniform(0.9, 1.1))

    for shift in range(0, 3):
        x.append(random.uniform(-0.1, 0.1))

    for b in range(0, len(E_var)):
        x.append(random.uniform(-1 * np.sqrt(E_var[b]) * 2, 2 * np.sqrt(E_var[b])))

    return x


def get_T1_T2(GAD_score, GAD_score_tot, incorrectList, n_f, plot_flag=False):
    # Optimize T2
    j = 0
    T2_init = 0.01
    T2_stop = 1.5
    stepSize = 0.01
    numSteps = int((T2_stop - T2_init) / stepSize)
    x = np.zeros(numSteps)
    y = np.zeros(numSteps)
    f_score = np.zeros(numSteps)
    T2_value = np.zeros(numSteps)
    k = 0
    for T2 in np.arange(T2_init, T2_stop, stepSize):
        k = 0
        TPR = np.zeros(len(incorrectList))  # Recall/Sensitivity
        PPV = np.zeros(len(incorrectList))  # Precision/positive predictive value
        FPR = np.zeros(len(incorrectList))  # Fallout
        for prediction in GAD_score:
            score = np.asarray(prediction)
            score[score > T2] = 1
            score[score <= T2] = 0

            ans = np.zeros(n_f)
            ans[incorrectList[k]] = 1
            TP, FP, TN, FN = perf_measure(ans, score)

            if (TN + FP) == 0:
                FPR[k] = 1
            else:
                FPR[k] = FP / (TN + FP)  # Fallout

            if (TP + FN) == 0:
                TPR[k] = 1
            else:
                TPR[k] = TP / (TP + FN)  # Recall/Sensitivity

            if (TP + FP) == 0:
                PPV[k] = 1
            else:
                PPV[k] = TP / (TP + FP)  # Precision

            k = k + 1
        y[j] = np.mean(TPR)
        x[j] = np.mean(FPR)
        f_score[j] = 2 * (np.mean(TPR) * np.mean(PPV)) / (np.mean(TPR) + np.mean(PPV))
        T2_value[j] = T2
        j = j + 1

    if plot_flag:
        plt.scatter(x, y)
        plt.show()

    # Optimize T1
    j = 0
    T1_init = 0.01
    T1_stop = 1.5
    stepSize = 0.01
    numSteps = int((T1_stop - T1_init) / stepSize)
    T2 = np.min(T2_value[np.where(f_score == np.max(f_score))])
    x = np.zeros(numSteps)
    y = np.zeros(numSteps)
    f_score = np.zeros(numSteps)
    T1_value = np.zeros(numSteps)
    for T1 in np.arange(T1_init, T1_stop, stepSize):
        k = 0
        TPR = np.zeros(len(incorrectList))
        PPV = np.zeros(len(incorrectList))  # Precision/positive predictive value
        FPR = np.zeros(len(incorrectList))
        for total_prediction in GAD_score_tot:
            if total_prediction < T1:
                score = np.zeros(n_f)
            else:
                score = np.asarray(GAD_score[k])
                score[score > T2] = 1
                score[score <= T2] = 0

            ans = np.zeros(n_f)
            ans[incorrectList[k]] = 1
            TP, FP, TN, FN = perf_measure(ans, score)

            if (TN + FP) == 0:
                FPR[k] = 1
            else:
                FPR[k] = FP / (TN + FP)  # False Positive Rate

            if (TP + FN) == 0:
                TPR[k] = 1
            else:
                TPR[k] = TP / (TP + FN)  # True Positive Rate

            if (TP + FP) == 0:
                PPV[k] = 1
            else:
                PPV[k] = TP / (TP + FP)  # Precision
            k = k + 1

        y[j] = np.mean(TPR)
        x[j] = np.mean(FPR)
        f_score[j] = 2 * (np.mean(TPR) * np.mean(PPV)) / (np.mean(TPR) + np.mean(PPV))
        T1_value[j] = T1
        j = j + 1

    if plot_flag:
        plt.scatter(x, y)
        plt.show()

    T1 = np.min(T1_value[np.min(np.where(f_score == np.max(f_score)))])
    TPR_mean = y[np.min(np.where(f_score == np.max(f_score)))]
    FPR_mean = x[np.min(np.where(f_score == np.max(f_score)))]

    return T1, T2, TPR_mean, FPR_mean, f_score[np.min(np.where(f_score == np.max(f_score)))]


def score_case(T1, T2, GAD_score, GAD_score_tot, incorrectList, n_f):
    k = 0
    TPR = np.zeros(len(incorrectList))
    PPV = np.zeros(len(incorrectList))  # Precision/positive predictive value
    FPR = np.zeros(len(incorrectList))
    for total_prediction in GAD_score_tot:
        if total_prediction < T1:
            score = np.zeros(n_f)
        else:
            score = np.asarray(GAD_score[k])
            score[score > T2] = 1
            score[score <= T2] = 0

        ans = np.zeros(n_f)
        ans[incorrectList[k]] = 1
        TP, FP, TN, FN = perf_measure(ans, score)

        if (TN + FP) == 0:
            FPR[k] = 1
        else:
            FPR[k] = FP / (TN + FP)  # False Positive Rate

        if (TP + FN) == 0:
            TPR[k] = 1
        else:
            TPR[k] = TP / (TP + FN)  # True Positive Rate

        if (TP + FP) == 0:
            PPV[k] = 1
        else:
            PPV[k] = TP / (TP + FP)  # Precision
        k = k + 1

    TPR_mean = np.mean(TPR)
    FPR_mean = np.mean(FPR)
    f_score = 2 * (np.mean(TPR) * np.mean(PPV)) / (np.mean(TPR) + np.mean(PPV))

    return TPR_mean, FPR_mean, f_score


def main():
    masterStructureList = "G:\\Projects\\mimTemplates\\StructureListMaster.xlsx"
    structureList = pd.read_excel(masterStructureList)
    contourDatabase = "H:\\Treatment Planning\\Elguindi\\contourDatabase\\contourDB.xlsx"
    db = pd.read_excel(contourDatabase, index=False)
    HDF5_DIR = 'H:\\Treatment Planning\\Elguindi\\GAD'
    filename = 'prostate_GAD_centroid'
    sl = [x.upper() for x in structureList['StructureName'].to_list()]
    sl.remove('EXTERNAL')
    # ['xFiducial_1', 'xFiducial_2', 'xFiducial_3']
    structures = ['Bladder_O', 'Rectum_O', 'CTV_PROST', 'Penile_Bulb',
                  'Rectal_Spacer', 'Femur_L', 'Femur_R']
    items = []
    for s in structures:
        items.append(s.upper() + '_' + 'Reference Centroid')

    db_centroid = db.filter(items=items).dropna(axis=0)
    G_cen = np.zeros((np.shape(db_centroid)[0], np.shape(db_centroid)[1] * 3))
    n = 0
    for i in db_centroid.index:
        gmn = db_centroid.loc[i, :]
        m = 0
        for item in items:
            coord = gmn[item].split('_')
            for c in coord:
                try:
                    G_cen[n, m] = float(c)
                    m = m + 1
                except:
                    m = m
        n = n + 1

    pctSplit = 0.8
    G_cen = G_cen[~np.isnan(G_cen).any(axis=1)]
    m_tot, n_tot = np.shape(G_cen)

    G_cen_test = G_cen[int(np.floor(m_tot * pctSplit)):, :]
    G_cen = G_cen[0:int(np.floor(m_tot * pctSplit)), :]
    G_m_all, G_cen_aligned = generalized_procrustes_analysis(G_cen)
    df = pd.DataFrame(G_cen)
    stdDev = np.asarray(df.describe().iloc[2, :])
    m, n = np.shape(G_cen_aligned)
    G_bar = np.zeros((1, np.shape(G_cen)[1]))
    G_bar[0, :] = np.mean(G_cen_aligned, 0)
    X = G_cen_aligned - G_bar
    U, s, Vh = linalg.svd(X, full_matrices=False)
    s = (s ** 2) / (m - 1)
    V = Vh.T

    E_vec = V[:, np.where(np.cumsum(s) / np.sum(s) <= 0.99)].squeeze()
    E_vec = E_vec.T
    t = len(np.where(np.cumsum(s) / np.sum(s) <= 0.99)[0])
    E_val = np.zeros((t, 1))
    E_val[:, 0] = s[np.where(np.cumsum(s) / np.sum(s) <= 0.99)].squeeze()
    E_var = np.copy(E_val).squeeze()

    data = {'trainingDataRaw': G_cen, 'trainingDataAligned': G_cen_aligned, 'eigenvectors': E_vec, 'eigenvalues': E_var,
            'G_bar': G_bar, 'standard_deviation': stdDev}
    store_arrays_hdf5(data, HDF5_DIR, filename)

    # m = 100
    mult = 1
    n_f = int(n / 3)
    num_iterations = 5

    # Create training set
    G_train = np.zeros((m * mult, n))
    g = 0
    correctList = []
    incorrectList = []
    for i in range(0, m):
        g_sample = G_cen[i, :]
        for k in range(0, mult):
            g_sample_mod, correct, incorrect = generate_random_sample(g_sample, stdDev, 4, [3, 6], [0, 0.5])
            mean_shape = np.reshape(g_sample_mod, (n_f, 3))
            mean_shape -= np.mean(mean_shape, 0)
            norm1 = np.linalg.norm(mean_shape)
            mean_shape /= norm1
            G_train[g, :] = mean_shape.flatten()
            correctList.append(correct)
            incorrectList.append(incorrect)
            g = g + 1

    # Initialization Parameters
    GAD_score = []
    GAD_score_tot = []
    for sample in range(0, m*mult):
        method = 'Powell'  # Powell, BFGS, Nelder-Mead, CG
        G_ex = np.zeros((1, n))
        G_ex[0, :] = G_train[sample, :]
        W = np.zeros((n, n))
        np.fill_diagonal(W, 1)

        print('Starting analysis on sample: ' + str(sample))
        start = timer()
        x_o = generate_random_x(E_var)
        results = basinhopping(cost_minimization_function, x_o, niter=num_iterations,
                               minimizer_kwargs={"args": (G_ex, G_bar, E_vec, W, E_var), "method": method})

        epsilon_a = np.copy(results.fun)
        GAD_score_tot.append(epsilon_a)
        epsilon_delta = list(range(0, n_f))
        for nn in epsilon_delta:
            print(nn)
            W = np.zeros((n, n))
            np.fill_diagonal(W, 1)
            zero_row = np.zeros(np.shape(W[0, :]))
            W[3 * nn, :] = zero_row
            W[3 * nn + 1, :] = zero_row
            W[3 * nn + 2, :] = zero_row
            results_nn = basinhopping(cost_minimization_function, x_o, niter=num_iterations,
                                      minimizer_kwargs={"args": (G_ex, G_bar, E_vec, W, E_var), "method": method})

            cost_value_nn = np.copy(results_nn.fun)
            epsilon_delta[nn] = np.absolute(cost_value_nn - epsilon_a)

        end = timer()
        print('Compared Train Sample:' + str(sample))
        print('Time taken: ' + str(end - start) + ' (s)')
        GAD_score.append(epsilon_delta)

    T1, T2, TPR_mean, FPR_mean, F_score = get_T1_T2(GAD_score, GAD_score_tot, incorrectList, n_f,
                                                    plot_flag=False)

    # Create testing set
    m, n = np.shape(G_cen_test)
    mult = 5
    G_test = np.zeros((m * mult, n))
    g = 0
    num_iterations = 5
    correctList = []
    incorrectList = []
    for i in range(0, m):
        g_sample = G_cen_test[i, :]
        for k in range(0, mult):
            g_sample_mod, correct, incorrect = generate_random_sample(g_sample, stdDev, 4, [3, 6], [0, 0.5])
            mean_shape = np.reshape(g_sample_mod, (n_f, 3))
            mean_shape -= np.mean(mean_shape, 0)
            norm1 = np.linalg.norm(mean_shape)
            mean_shape /= norm1
            G_test[g, :] = mean_shape.flatten()
            correctList.append(correct)
            incorrectList.append(incorrect)
            g = g + 1

    # Initialization Parameters
    GAD_score = []
    GAD_score_tot = []
    for sample in range(0, m*mult):
        method = 'Powell'  # Powell, BFGS, Nelder-Mead, CG
        G_ex = np.zeros((1, n))
        G_ex[0, :] = G_test[sample, :]
        W = np.zeros((n, n))
        np.fill_diagonal(W, 1)

        print('Starting analysis on sample: ' + str(sample))
        start = timer()
        x_o = generate_random_x(E_var)
        results = basinhopping(cost_minimization_function, x_o, niter=num_iterations,
                               minimizer_kwargs={"args": (G_ex, G_bar, E_vec, W, E_var), "method": method})
        epsilon_a = np.copy(results.fun)
        GAD_score_tot.append(epsilon_a)
        epsilon_delta = list(range(0, n_f))
        for nn in epsilon_delta:
            print(nn)
            W = np.zeros((n, n))
            np.fill_diagonal(W, 1)
            zero_row = np.zeros(np.shape(W[0, :]))
            W[3 * nn, :] = zero_row
            W[3 * nn + 1, :] = zero_row
            W[3 * nn + 2, :] = zero_row
            results_nn = basinhopping(cost_minimization_function, x_o, niter=num_iterations,
                                      minimizer_kwargs={"args": (G_ex, G_bar, E_vec, W, E_var), "method": method})
            cost_value_nn = np.copy(results_nn.fun)
            epsilon_delta[nn] = np.absolute(cost_value_nn - epsilon_a)
        end = timer()
        print('Compared Train Sample:' + str(sample))
        print('Time taken: ' + str(end - start) + ' (s)')
        GAD_score.append(epsilon_delta)

    print(T1)
    print(T2)


if __name__ == '__main__':
    main()
