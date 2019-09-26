from numpy.random import random_sample
from scipy.stats import norm, invgamma
from numpy import sqrt, asarray, zeros, append


class Density(object):
    """
    输入一个list
    输出: 一个函数, 返回一个密度估计
    """
    def __init__(self, mu0=0., dev0=1., m=1., alpha=2.1, beta=2.):
        self.mu0 = mu0
        self.dev0 = dev0
        self.m = m
        self.alpha = alpha
        self.beta = beta

    def _sample(self, weights):
        """
        从多项分布中抽样
        返回系数
        """
        u = random_sample() * sum(weights)
        sample = 0
        weight_sum = weights[0]
        while sample < len(weights) - 1 and weight_sum <= u:
            sample += 1
            weight_sum += weights[sample]
        return sample

    def pdf(self, x):
        """
        返回
        """
        xs = asarray(x)
        density = zeros(xs.shape, 'd')
        denom = float(self.n + self.m)
        s = sqrt(self.s2)
        for i in range(len(self.means)):
            density += ((float(self.counts[i]) / denom)
                * norm.pdf(x, self.means[i], scale=s))
        density += ((self.m / denom)
            * norm.pdf(x, self.mu0, sqrt(self.dev0 + self.s2)))
        return density

    def __rv(self, weights):
        s = self._sample(weights)
        if s < len(self.means):  # Draw from a component
            return norm.rvs(self. means[s], scale=sqrt(self.s2))
        else:                    # Draw from the marginal
            return norm.rvs(self.mu0, scale=sqrt(self.dev0 + self.s2))

    def rvs(self, size=1):
        weights = list(self.counts)
        weights.append(self.m)
        if size == 1:
            return self.__rv(weights)
        else:
            r_vars = []
            for i in range(size):
                r_vars.append(self.__rv(weights))
            return r_vars

    def __create_weight_vector(self, x, means, counts, s2, mu0, dev0, m):
        weights = counts * norm.pdf(x, loc=means, scale=sqrt(s2))
        weights = append(weights,
                         m * norm.pdf(x, loc=mu0, scale=sqrt(dev0 + s2)))
        return weights

    def __fix_hole(self, assignments, sums, means, counts,
                   hole_location):
        last_idx = len(means) - 1
        for i in range(len(assignments)):
            if assignments[i] == last_idx:
                assignments[i] = hole_location
        sums[hole_location] = sums[-1]
        sums.pop()
        means[hole_location] = means[-1]
        means.pop()
        counts[hole_location] = counts[-1]
        counts.pop()

    def estimate(self, xs, max_iterations=10000):
        self.n = len(xs)
        self.s2 = invgamma.rvs(self.alpha, scale=self.beta)
        self.means = []
        response_sums = []
        self.counts = []
        assignments = []

        assignments.append(0)
        response_sums.append(xs[0])
        self.counts.append(1)
        self.means.append(self.sample_posterior_mean(response_sums[0], 1,
                                                     self.s2, self.mu0,
                                                     self.dev0))

        for i in range(1, len(xs)):
            weights = self.__create_weight_vector(xs[i], self.means, self.counts, self.s2,\
                self.mu0, self.dev0, elf.m)
            # Sample an assignment for each item and update statistics
            assignment = self._sample(weights)
            if assignment < len(self.means):
                response_sums[assignment] += xs[i]
                self.counts[assignment] += 1
            # Create a new component
            elif assignment == len(self.means):
                response_sums.append(xs[i])
                self.counts.append(1)
                self.means.append(self.sample_posterior_mean(xs[i], 1,
                                                             self.s2, self.mu0,
                                                             self.dev0))
            assignments.append(assignment)

        for i in range(max_iterations):
            # First sample an assignment for each data item

            for j in range(len(xs)):
                old_assignment = assignments[j]
                response_sums[old_assignment] -= xs[j]
                self.counts[old_assignment] -= 1

                if self.counts[old_assignment] == 0:
                    self.__fix_hole(assignments, response_sums, self.means,
                                    self.counts, old_assignment)
                weights = self.__create_weight_vector(xs[j], self.means,
                                                      self.counts,
                                                      self.s2, self.mu0,
                                                      self.dev0, self.m)
                new_assignment = self._sample(weights)
                # Create a new component
                if new_assignment < len(self.means):
                    response_sums[new_assignment] += xs[j]
                    self.counts[new_assignment] += 1
                elif new_assignment == len(self.means):
                    response_sums.append(xs[j])
                    self.counts.append(1)
                    self.means.append(self.sample_posterior_mean(xs[j], 1,
                                                                 self.s2,
                                                                 self.mu0,
                                                                 self.dev0))
                assignments[j] = new_assignment

            # Sample new values for the means
            for j in range(len(self.means)):
                self.means[j] = self.sample_posterior_mean(response_sums[j],
                                                           self.counts[j],
                                                           self.s2, self.mu0,
                                                           self.dev0)

            # Sample new value for the component variance
            res_sum_squares = self.calculate_res_sum_squares(xs, assignments,
                                                             self.means)
            self.s2 = self.sample_posterior_var(self.n, res_sum_squares,
                                                self.alpha, self.beta)
        return self

    def calculate_res_sum_squares(self, xs, assignments, means):
        res_sum_squares = 0.0
        for i in range(len(xs)):
            res = xs[i] - means[assignments[i]]
            res_sum_squares += (res * res)
        return res_sum_squares

    def sample_posterior_mean(self, data_sum, n, s2, mu0, dev0):
        var_star = 1. / ((1. / dev0) + (n / s2))
        m_star = var_star * ((mu0 / dev0) + (data_sum / s2))
        return norm.rvs(m_star, sqrt(var_star))

    def sample_posterior_var(self, n, res_sum_squares, alpha, beta):
        alpha_star = alpha + (n / 2.)
        beta_star = beta + (res_sum_squares / 2.)
        return invgamma.rvs(alpha_star, scale=beta_star)
