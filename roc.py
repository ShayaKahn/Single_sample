import numpy as np
from scipy.stats import rv_histogram

class Roc:
    def __init__(self, positive_data, negative_data, num_threshold_values):
        if np.mean(positive_data) > np.mean(negative_data):
            self.pos = positive_data
            self.neg = negative_data
        else:
            self.pos = negative_data
            self.neg = positive_data
        self.num = num_threshold_values
        self.threshold_vector = np.linspace(np.min(negative_data), np.max(positive_data), self.num)
        self.tpr_vector = np.zeros(np.size(self.threshold_vector))
        self.fpr_vector = np.zeros(np.size(self.threshold_vector))
        self.pos_hist_dist, self.neg_hist_dist = self.create_prob_dist()
        self.calc_fpr_tpr()
        self.Auc = self.calc_ara_under_curve()

    def create_prob_dist(self):
        pos_hist = np.histogram(self.pos)
        neg_hist = np.histogram(self.neg)
        pos_hist_dist = rv_histogram(pos_hist)
        neg_hist_dist = rv_histogram(neg_hist)
        return pos_hist_dist, neg_hist_dist

    def calc_fpr_tpr(self):
        for i, threshold in enumerate(self.threshold_vector):
            self.fpr_vector[i] = self.pos_hist_dist.cdf(threshold)
            self.tpr_vector[i] = self.neg_hist_dist.cdf(threshold)

    def calc_ara_under_curve(self):
        area = np.trapz(y=self.tpr_vector, x=self.fpr_vector)
        return area






