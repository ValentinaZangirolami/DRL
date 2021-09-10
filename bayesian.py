class Beta:

    def __init__(self, alpha0, beta0):
        self.alpha = alpha0
        self.beta = beta0

    def update(self, expert1, expert2):
        """
        Update alpha and beta (parameters of Beta) with Dirichlet moment-matching technique
        """
        alpha, beta = self.alpha, self.beta
        mean = expert1 * alpha + expert2 * beta
        if mean <= 0.0:
            return
        m = alpha / (alpha + beta + 1.) * (expert1 * (alpha + 1.) + expert2 * beta) / mean
        s = alpha / (alpha + beta + 1.) * (alpha + 1.) / (alpha + beta + 2.) * \
            (expert1 * (alpha + 2.) + expert2 * beta) / mean
        r = (m - s) / (s - m * m)
        self.alpha = m * r
        self.beta = (1. - m) * r
        return self.alpha, self.beta

class Average:

    def __init__(self):
        self.mean, self.m2, self.var, self.count = 0.0, 0.0, 0.0, 0

    def update(self, point):
        """
        Update mean and variance of returns
        """
        self.count += 1
        count = self.count
        delta = point - self.mean
        self.mean += delta / count
        self.m2 += delta * (point - self.mean)
        if count > 1:
            self.var = self.m2 / (count - 1.0)
        return self.var
