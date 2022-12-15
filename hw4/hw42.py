import matplotlib.pyplot as plt
import numpy as np
from random import random
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
import math
import warnings
from tqdm import tqdm
import altair as alt
from sklearn import mixture
warnings.filterwarnings("ignore")
def compute_log_likelihood(m, data):
    '''
    m is the model
    data is samples
    '''
    return m.score(data)
def alt_plot(dataset):

    chart = (alt.Chart(dataset).mark_bar().encode(x='GMM order based on likehood values:N',y='count()',color="GMM order based on likehood values:N",))
    return chart
def sample_distribution(n, w):
    output = list()
    N0 = 0
    N1 = 0
    N2 = 0
    N3 = 0
    for i in range(n):
        no = random()
        if no <= w[0]:
            N0 = N0 + 1
        elif no <= (w[0] + w[1]):
            N1 = N1 + 1
        elif no <= (w[0] + w[1] + w[2]):
            N2 = N2 + 1
        else:
            N3 = N3 + 1
    return (N0, N1, N2, N3)
def generate_gaussian_samples(num_sample, mu, cov):
    datapoints = np.random.multivariate_normal(mu, cov, num_sample)
    datapoints = pd.DataFrame(datapoints, columns=['x', 'y'])
    return datapoints
def data_gen(n):
    w0 = 0.2
    w1 = 0.3
    w2 = 0.15
    w3 = 0.35
    w = [w0, w1, w2, w3]
    mu01    = [2, 2]

    mu02    = [-2, 2]

    mu03    = [-2, -2]

    mu04    = [1, -1]

    mu =    [mu01, mu02, mu03, mu04]

    cov01 = [[0.1, 0], [0, 0.1]]
    cov02    = [[0.2, 0.1], [0.1, 0.3]]

    cov03    = [[0.3, 0], [0, 0.2]]

    cov04    = [[0.2, 0], [0, 0.2]]

    cov =    [cov01, cov02, cov03, cov04]

    num_samples = sample_distribution(n, w)
    datapoints = pd.DataFrame()
    for j in range(4):
        datapoints = datapoints.append(generate_gaussian_samples(num_samples[j], mu[j], cov[j]))
    datapoints = datapoints.reset_index(drop=True)
    return datapoints
def plot_datapoints_scatter(datapoints, title):
    plt.scatter(datapoints['x'], datapoints['y'], c='b', label='Data Generated')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.clf()
def get_xy_from_frame(dataset):
    x = list()
    y = list()
    for index, row in dataset.iterrows():
        x.append(row['x'])
        y.append(row['y'])
    xy = np.array([np.array(x), np.array(y)])
    return xy
def kfoldselection(dataset):
    N = dataset.shape[0]
    data = get_xy_from_frame(dataset)
    n_components = np.arange(1, 9)
    models = [mixture.GaussianMixture(n, covariance_type='full',random_state=0) for n in n_components]
    log_likelihood = []
    bic = []
    aic = []
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    for m in models:
        '''
        Lists scores, scores_bic, scores_aic to store 'k' scores on 'k'
        different validation set
        '''
        scores = []
        scores_bic = []
        scores_aic = []
        for train_index, test_index in cv.split(data.T):
            X_train, X_test = data.T[train_index], data.T[test_index]
            m.fit(X_train) # model.fit
            scores.append(compute_log_likelihood(m, X_test)) # find log_likelihood on test set and add it to the list
            scores_bic.append(m.bic(X_test)) # find bic on test set and add it to the list
            scores_aic.append(m.aic(X_test)) # find aic on test set and add it to the list
        log_likelihood.append(-1 * sum(scores) / len(scores)) # average the score over k validation sets
        bic.append(sum(scores_bic) / len(scores_bic)) # average the score over k validation sets
        aic.append(sum(scores_aic) / len(scores_aic)) # average the score over k validation sets
    return log_likelihood, bic, aic
def runs(no_points):
    selection = pd.DataFrame()
    index = 0
    for i in tqdm(range(30), total=30):
        dataset = data_gen(no_points)
        likelihood, bic, aic = kfoldselection(dataset=dataset)
        selection.loc[index, "GMM order based on likehood values"] =(likelihood.index(min(likelihood)) + 1)
        index += 1
        fig = plt.figure(figsize=(10, 10))
        plt.plot(np.arange(1, 9), likelihood, label='negative log_likelihood')
        plt.xlabel('n_components', size=13)
        plt.ylabel('score', size=13)
        plt.legend(loc='best')
        fig.suptitle('Negative log_likelihood score for different GMM models',fontsize=15)
        plt.subplots_adjust(top=0.95)
        plt.savefig(f"q2_likelihood/{str(len(dataset))}_{i}")
        plt.clf()
        fig = plt.figure(figsize=(10, 10))
        plt.xlabel('n_components', size=13)
        plt.ylabel('score', size=13)
        plt.plot(np.arange(1, 9), bic, label='BIC')
        plt.plot(np.arange(1, 9), aic, label='AIC')
        fig.suptitle('BIC and AIC scores for different GMM models',fontsize=15)
        plt.subplots_adjust(top=0.95)
        plt.savefig(f"q2_bic/{str(len(dataset))}_{i}")
        plt.clf()
    return selection
if __name__ == "__main__":
    # Generating samples to plot 
    data_10 = data_gen(10)
    data_100 = data_gen(100)
    data_1000 = data_gen(1000)
    data_10000 = data_gen(10000)
    plot_datapoints_scatter(data_10, "Dataset_with_10_points")
    plot_datapoints_scatter(data_100, "Dataset_with_100_points")
    plot_datapoints_scatter(data_1000, "Dataset_with_1000_points")
    plot_datapoints_scatter(data_10000, "Dataset_with_10000_points")
    selection_10000 = runs(10000)
    selection_1000 = runs(1000)
    selection_10 = runs(10)
    selection_100 = runs(100)
    selection_10.to_csv("10_data.csv", index=False)
    selection_100.to_csv("100_data.csv",index=False)
    selection_1000.to_csv("1000_data.csv",index=False)
    selection_10000.to_csv("10000_data.csv",index=False)
   # plot alt_charts 
    chart = alt_plot(selection_10)
    chart.save('chart10.html')
    chart = alt_plot(selection_100)
    chart.save('chart100.html')
    chart = alt_plot(selection_1000)
    chart.save('chart1000.html')
    chart = alt_plot(selection_10000)
    chart.save('chart10000.html')