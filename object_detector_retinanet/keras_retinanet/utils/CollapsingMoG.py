import math

import numpy as np
import scipy


def agglomerative_init(alpha, mu, covariance, n, k):

    mu_stack = np.zeros(shape=[n - k, mu.shape[1]], dtype=mu.dtype)
    mu_stack.fill(np.inf)
    mu_temp = np.vstack([mu.copy(), mu_stack])
    covariance_temp = np.vstack([
        covariance,
        np.zeros([n - k, covariance.shape[1], covariance.shape[2]],
                 dtype=covariance.dtype)
    ])
    alpha_temp = np.hstack([
        alpha,
        np.zeros(shape=(n - k), dtype=alpha.dtype)
    ])
    distances = scipy.spatial.distance.cdist(mu_temp, mu_temp)
    distances = np.triu(distances)
    distances = np.nan_to_num(distances)
    distances[distances == 0] = np.inf
    deleted = []
    for l in range(n, 2 * n - k):
        # get (row, col) of the min distance
        i, j = np.unravel_index(np.argmin(distances), distances.shape)

        alpha_ij = alpha_temp[i] + alpha_temp[j]
        mu_ij = (alpha_temp[i] * mu_temp[i] +
                 alpha_temp[j] * mu_temp[j]) / alpha_ij
        harmonic_mean = (alpha_temp[i] * alpha_temp[j]) / alpha_ij
        delta_mu = (mu_temp[i] - mu_temp[j])
        delta_mu = np.expand_dims(delta_mu, axis=1)
        covariance_ij = (alpha_temp[i] * covariance_temp[i] + alpha_temp[j] * covariance_temp[
            j] + harmonic_mean * np.dot(delta_mu, delta_mu.transpose())) / alpha_ij

        mu_temp[l] = mu_ij
        covariance_temp[l] = covariance_ij
        alpha_temp[l] = alpha_ij

        distances[:, i] = np.inf
        distances[:, j] = np.inf
        distances[i, :] = np.inf
        distances[j, :] = np.inf
        mu_temp[i] = np.inf
        mu_temp[j] = np.inf
        deleted.append(i)
        deleted.append(j)

        d = scipy.spatial.distance.cdist(
            mu_temp, np.expand_dims(mu_ij, axis=0))[:, 0]
        d[d == 0] = np.inf
        distances[:, l] = d
    deleted_indexes = np.array(deleted)
    mask = np.ones(alpha_temp.shape[0], dtype=bool)
    if deleted_indexes.shape[0] > 0:
        mask[deleted_indexes] = False
    return alpha_temp[mask], mu_temp[mask], covariance_temp[mask]


def gaussian_kl(mu1, cov1, mu2, cov2):
    cov2inv = np.linalg.inv(cov2)
    log_det_ratio = np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
    delta_mu = (mu1 - mu2)
    delta_mu = np.expand_dims(delta_mu, axis=1)
    return 0.5 * (log_det_ratio
                  + np.trace(np.dot(cov2inv, cov1))
                  + np.dot(np.dot(delta_mu.transpose(), cov2inv), delta_mu))[0][0]


def gaussian_kl_diag(mu1, cov1, mu2, cov2):
    cov2sqrt = np.sqrt(cov2)
    cov1sqrt = np.sqrt(cov1)
    log_ratio = math.log(cov2sqrt[0, 0] / cov1sqrt[0, 0]) + \
        math.log(cov2sqrt[1, 1] / cov1sqrt[1, 1])
    delta_mu = (mu1 - mu2)
    div = (cov1[0, 0] + delta_mu[0] * delta_mu[0]) / (2 * cov2[0, 0]) + (cov1[1, 1] + delta_mu[1] * delta_mu[1]) / (
        2 * cov2[1, 1])
    return div + log_ratio


def collapse(original_detection, k, offset, max_iter=100, epsilon=1e-100):

    n = original_detection.shape[0]

    # distance from (cx, cy) to the top-left corner of the contour
    mu_x = original_detection.x - offset[0]
    mu_y = original_detection.y - offset[1]

    sigma_xx = original_detection.sigma_x * original_detection.sigma_x
    sigma_yy = original_detection.sigma_y * original_detection.sigma_y

    alpha = np.array(original_detection.confidence /
                     original_detection.confidence.sum())
    mu = np.array([mu_x.values, mu_y.values]).transpose()
    covariance = np.array(
        [[sigma_xx.values, sigma_xx.values * 0],
         [0 * sigma_yy.values, sigma_yy.values]]).transpose()

    beta, mu_prime, covariance_prime = agglomerative_init(
        alpha.copy(), mu.copy(), covariance.copy(), n, k)

    beta_init = beta.copy()
    mu_prime_init = mu_prime.copy()
    covariance_prime_init = covariance_prime.copy()
    iteration = 0
    d_val = float('inf')
    delta = float('inf')
    min_kl_cache = {}
    while delta > epsilon and iteration < max_iter:
        iteration += 1
        clusters, clusters_inv = e_step(alpha, beta, covariance,
                                        covariance_prime, mu, mu_prime,
                                        min_kl_cache)
        m_step(alpha, beta, clusters, covariance,
               covariance_prime, mu, mu_prime)

        prev_d_val = d_val
        d_val = 0
        for t, (alpha_, mu_, cov_) in enumerate(zip(alpha, mu, covariance)):
            min_dist, selected_cluster = min_kl(
                beta, cov_, covariance_prime, mu_, mu_prime)
            min_kl_cache[t] = (min_dist, selected_cluster)
            d_val += alpha_ * min_dist
        delta = prev_d_val - d_val
        if delta < 0:
            print('EM bug - not monotonic- using fallback')
            return beta_init, mu_prime_init, covariance_prime_init
        #Log.debug('Iteration {}, d_val={}, delta={}, k={}, n={}'.format(iteration, d_val, delta, k, n))

    if delta > epsilon:
        print('EM did not converge- using fallback')
        return beta_init, mu_prime_init, covariance_prime_init
    return beta, mu_prime, covariance_prime


def e_step(alpha, beta, covariance, covariance_prime, mu, mu_prime, min_kl_cache):
    """ The E-step assigns each box to the nearest box cluster, 
    where box similarity is defined by a KL distance between 
    the corresponding Gaussians. 
    """
    clusters = {}
    clusters_inv = {}
    for t, (alpha_, mu_, cov_) in enumerate(zip(alpha, mu, covariance)):
        if t in min_kl_cache:
            min_dist, selected_cluster = min_kl_cache[t]
        else:
            min_dist, selected_cluster = min_kl(
                beta, cov_, covariance_prime, mu_, mu_prime)

        if selected_cluster not in clusters:
            clusters[selected_cluster] = []
        clusters[selected_cluster].append(t)
        clusters_inv[t] = selected_cluster
    return clusters, clusters_inv


def min_kl(beta, cov_, covariance_prime, mu_, mu_prime):
    cov_g = np.zeros_like(mu_prime)
    cov_g[:, 0] = covariance_prime[:, 0, 0]
    cov_g[:, 1] = covariance_prime[:, 1, 1]

    cov_f = np.zeros_like(mu_prime)
    cov_f[:, 0] = cov_[0, 0]
    cov_f[:, 1] = cov_[1, 1]

    mu_f = np.zeros_like(mu_prime)
    mu_f[:, 0] = mu_[0]
    mu_f[:, 1] = mu_[1]
    mu_g = mu_prime

    cov_g_sqrt = np.sqrt(cov_g)
    cov_f_sqrt = np.sqrt(cov_f)
    log_ratio = np.log(cov_g_sqrt[:, 0] / cov_f_sqrt[:, 0]) + \
        np.log(cov_g_sqrt[:, 1] / cov_f_sqrt[:, 1])
    delta_mu = mu_f - mu_g
    delta_mu_square = delta_mu * delta_mu
    div = (cov_f[:, 0] + delta_mu_square[:, 0]) / (2 * cov_g[:, 0]) + (cov_f[:, 1] + delta_mu_square[:, 1]) / (
        2 * cov_g[:, 1])
    kl = div + log_ratio
    return kl.min(), kl.argmin()


def m_step(alpha, beta, clusters, covariance, covariance_prime, mu, mu_prime):
    for j, t_vals in clusters.items():
        beta_update = 0
        for t in t_vals:
            beta_update += alpha[t]
        beta[j] = beta_update

        mu_update = np.array([0, 0])
        for t in t_vals:
            mu_update = np.add(mu_update, alpha[t] * mu[t])
        mu_update /= beta[j]
        mu_prime[j] = mu_update

        cov_update = np.array([[0, 0], [0, 0]])
        for t in t_vals:
            delta_mu = (mu[t] - mu_prime[j])
            delta_mu = np.expand_dims(delta_mu, axis=1)
            cov_update = np.add(
                cov_update, alpha[t] * (covariance[t] + np.dot(delta_mu, delta_mu.transpose())))
        cov_update /= beta[j]
        covariance_prime[j] = cov_update
