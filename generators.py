__author__ = 'spencerkadas'

import numpy as np
from scipy import stats
from pprint import pprint

class Experience(object):
    """
    Represents an experience of a split test
    """
    def __init__(self, mean, name=''):
        self.mean = mean
        self.name = name

    def draw_binomial(self, n=1):
        """
        Performs random binomail draws from the distribution
        :param n: number of draws to perform
        :return: returns a single draw for n=1, else returns a list of draws
        """
        draw = np.random.binomial(1, self.mean, n)

        if len(draw) == 1:
            return draw[0]
        else:
            return draw


class Test(object):
    """
    Represents a test
    """
    def __init__(self, *args):
        # list of all of the tests
        self.experiences = args

        # initialize dictionary for data
        # will take the form 'split': [data]
        self.data = {}

        for x in self.experiences:
            self.data[x.name] = []

    def get_random(self):
        return np.random.randint(1, 100)

    @property
    def nexperiences(self):
        return len(self.experiences)

    @property
    def nobs(self):
        inc = 0
        for d in self.data:
            inc += sum(self.data[d])
        return inc

    def increment_experience(self, experience, result):
        """
        adds the result of a single binomial draw to the test data
        """
        self.data[experience].append(result)

    def calc_stats(self):
        for d in self.data:
            split_name = d
            obs = len(self.data[d])
            suc = sum(self.data[d])
            try:
                conv = suc / obs
            except ZeroDivisionError:
                conv = 0
            print(split_name, obs, suc, conv)

    def calc_win_percent(self):
        stats = {}
        for d in self.data:
            obs = len(self.data[d])
            suc = sum(self.data[d])
            beta = BetaDist(k=suc, n=obs)
            stats[d] = []

            for i in range(0, 1000):
                stats[d].append(beta.draw_rvs())

        return stats


    def reset_test(self):
        for x in self.experiences:
            self.data[x.name] = []

    def perform_draw(self):
        """
        Increments the data of the drawn split with the result of that experience's binomial draw
        :return:
        """
        split = self.get_split()
        experience = self.experiences[split]
        draw = experience.draw_binomial()
        # return experience, draw
        self.increment_experience(experience.name, draw)


class NwayTest(Test):
    """
    nway split test
    """
    def get_split(self):
        return self.get_random() % self.nexperiences


class EpsilonGreedy(Test):

    @property
    def decay(self):
        """
        decay value for the algorithm. Higher decay = slower attribution to harvesting
        :return: decay
        """
        return 50

    @property
    def epsilon(self):
        """
        Current epsilon value
        :return: decay / (# obs + decay)
        """
        return self.decay / (self.nobs + self.decay)

    def get_split(self):
        if np.random.random() > self.epsilon:
            # return best performing arm
            values = []
            for d in self.data:
                values.append(sum(self.data[d]) / len(self.data[d]))

            return values.index(max(values))

        else:
            # return random arm
            return self.get_random() % self.nexperiences


class BetaDist(object):
    def __init__(self, k, n):
        self.k = k
        self.n = n
        self.mean = k / n
        self.a = k + 1
        self.b = n - k + 1

    def draw_rvs(self):
        return stats.beta.rvs(self.a, self.b)



s1 = Experience(0.05, 'control')
s2 = Experience(0.08, 'test')
s3 = Experience(0.06, 'other')
T = EpsilonGreedy(s1, s2, s3)

for i in range(0, 5000):
    T.perform_draw()

s = T.calc_win_percent()

pprint(s)