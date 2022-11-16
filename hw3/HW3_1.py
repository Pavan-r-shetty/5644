from random import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import pandas as pd
from tqdm import tqdm
from sklearn.mixture import GaussianMixture as GMM
def sample_distribution(n, p0, p1):

    N0 = 0
    N1 = 0
    for i in range(n):
        if random() < p0:
            N0 = N0 + 1
        else:
            N1 = N1 + 1
    return (N0, N1)

def generate_gaussian_samples(num_sample, w, mu, cov):
    if len(w) > 1:
        no_gaussian_samples = sample_distribution(num_sample, w[0], w[1])
    else:
        no_gaussian_samples = tuple([num_sample])
    datapoints = list()
    for i in range(len(no_gaussian_samples)):
        if len(datapoints) == 0:
            datapoints = np.random.multivariate_normal(mu[i], cov[i], no_gaussian_samples[i])
        else:
            datapoints = np.append(datapoints,np.random.multivariate_normal(mu[i], cov[i],no_gaussian_samples[i]),axis=0,)
    return datapoints
def plot_datapoints_scatter(datapoints, labels, xlabel, ylabel, title):
    n0 = calculate_no_labels(labels, 0)
    print(n0)
    n1 = len(datapoints) - n0
    print(n1)
    plt.scatter(datapoints[:n0, 0], datapoints[:n0, 1], c='b', label='class 0')
    plt.scatter(datapoints[n0:, 0], datapoints[n0:, 1], c='r', label='class 1')
    plt.title(title)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
def plot_contours(data, means, covs, title):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')
    delta = 0.025
    k = means.shape[0]
    x = np.arange(-4.0, 10.0, delta)
    y = np.arange(-4.0, 10.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean,
        cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors=col[i])
    plt.title(title)
    plt.tight_layout()
def calculate_discriminant(datapoints, mu0, w0, cov0, mu1, cov1):
    data_discriminant = list()
    for i in tqdm(range(len(datapoints)), total=len(datapoints), desc="calcdata_discriminant"):
        val = np.log(multivariate_normal.pdf(datapoints[i], mu1[0], cov1[0])/ (w0[0] * multivariate_normal.pdf(datapoints[i], mu0[0], cov0[0])+ w0[1] * multivariate_normal.pdf(datapoints[i], mu0[1], cov0[1])))
        data_discriminant.append(val)
    return data_discriminant
def data_points_labels_preproc(data_points, data_labels):
    data_labels = np.append(np.array([data_labels]), np.array([np.zeros(len(data_labels))]), axis=0)
    data_points = np.append(np.array([data_points[:, 0]]), np.array([data_points[:, 1]]), axis=0)
    return (data_points, data_labels)
def hit_cls(data_discriminant, data_labels):
    false_positive = list()
    true_positive = list()
    gamma_values = list()
    prob_error = list()
    no_0_labels = calculate_no_labels(data_labels=data_labels, label_value=0)
    no_1_labels = calculate_no_labels(data_labels=data_labels, label_value=1)
    data_discriminant_label0 = data_discriminant[0:no_0_labels]
    data_discriminant_label1 = data_discriminant[no_0_labels:]
    for i in tqdm(sorted(data_discriminant), total=len(data_discriminant), desc="hitclassifier"):
        fp = len([j for j in data_discriminant_label0 if j >= i]) / no_0_labels
        tp = len([j for j in data_discriminant_label1 if j >= i]) / no_1_labels
        false_positive.append(fp)
        true_positive.append(tp)
        gamma_values.append(i)
        prob_error.append(fp * p0 + (1 - tp) * p1)
    return false_positive, true_positive, gamma_values, prob_error
def calculate_no_labels(data_labels, label_value):
    count = 0
    for i in data_labels:
        if i == label_value:
            count += 1
    return count
def find_estimated_params(train_data_points, train_data_labels):
    n0 = calculate_no_labels(data_labels=train_data_labels, label_value=0)
    GMMmodel = GMM(n_components=2, max_iter=100, tol=1e-3,covariance_type='full')
    GMMmodel.fit(train_data_points[0:n0])
    GMMmodel2 = GMM(n_components=1, max_iter=100, tol=1e-3,covariance_type='full')
    GMMmodel2.fit(train_data_points[n0:])
    return GMMmodel, GMMmodel2
def find_theta(train_data, alpha, iterations, train_labels, test_data, type_='l'):
    if type_ == 'l':
        z = np.c_[np.ones((train_data.shape[1])), train_data.T].T
        w = np.zeros((3, 1))
    else:
        z = np.c_[np.ones((train_data.shape[1])),train_data[0],train_data[1],train_data[0] * train_data[0],train_data[0] * train_data[1],train_data[1] * train_data[1],].T
        w = np.zeros((6, 1))
    for i in range(iterations):
        h = 1 / (1 + np.exp(-(np.dot(w.T, z))))
        cost_gradient = (1 / float(z.shape[1])) * np.dot(z, (h - train_labels[0]).T)
        w = w - alpha * cost_gradient
    print(w)
    if type_ == 'l':
        z = np.c_[np.ones((test_data.shape[1])), test_data.T].T
    else:
        z = np.c_[np.ones((test_data.shape[1])),test_data[0],test_data[1],test_data[0] * test_data[0],test_data[0] * test_data[1],test_data[1] * test_data[1],].T
    decisions = np.zeros((1, test_data.shape[1]))
    h = 1 / (1 + np.exp(-(np.dot(w.T, z))))
    decisions[0, :] = (h[0, :] >= 0.5).astype(int)
    return (w, decisions)
def plot_boundary(labels, W, flag, datapoints):
    X = datapoints
    x00 = [i for i in range(labels.shape[1]) if (labels[0, i] == 0 and labels[1, i] == 0)]
    x01 = [i for i in range(labels.shape[1]) if (labels[0, i] == 0 and labels[1, i] == 1)]
    x10 = [i for i in range(labels.shape[1]) if (labels[0, i] == 1 and labels[1, i] == 0)]
    x11 = [i for i in range(labels.shape[1]) if (labels[0, i] == 1 and labels[1, i] == 1)]
    plt.plot(X[0, x00], X[1, x00], '.', color='g', markersize=6)
    plt.plot(X[0, x01], X[1, x01], '.', color='r', markersize=6)
    plt.plot(X[0, x11], X[1, x11], '+', color='g', markersize=6)
    plt.plot(X[0, x10], X[1, x10], '+', color='r', markersize=6)
    plt.xlabel("Feature x1")
    plt.ylabel("Feature x2")
    plt.title('Distribution after classification overlapped by decision boundaries')
    plt.legend(["class 0 correctly classified",'class 0 wrongly classified','class 1 correctly classified','class 1 wrongly classified',])
    horizontalGrid = np.linspace(np.floor(min(X[0, :])), np.ceil(max(X[0, :])), 100)
    verticalGrid = np.linspace(np.floor(min(X[1, :])), np.ceil(max(X[1, :])), 100)
    dsg = np.zeros((100, 100))
    a = np.array(np.meshgrid(horizontalGrid, verticalGrid))
    for i in range(100):
        for j in range(100):
            x1 = a[0][i][j]
            x2 = a[1][i][j]
            if flag == 'l':
                z = np.c_[1, x1, x2].T
            else:
                z = np.c_[1, x1, x2, pow(x1, 2), x1 * x2, pow(x2, 2)].T
            dsg[i][j] = np.sum(np.dot(W, z))
    plt.contour(a[0], a[1], dsg, levels=[0])
    plt.show()
    return
if __name__ == '__main__':
    p0 = 0.6
    p1 = 0.4
    # mu and sigma's
    mu0 = list([np.transpose([5, 0]), np.transpose([0, 4])])
    cov0 = list([np.array([[4, 0], [0, 2]]), np.array([[1, 0], [0, 3]])])
    mu1 = list([np.transpose([3, 2])])
    cov1 = list([np.array([[2, 0], [0, 2]])])
    w0 = list([0.5, 0.5])
    w1 = list([1])
    # 100 dataset
    l0, l1 = sample_distribution(100, p0, p1)
    r0 = generate_gaussian_samples(num_sample=l0, w=w0, mu=mu0,cov=cov0)
    l0 = [0] * len(r0)
    r1 = generate_gaussian_samples(num_sample=l1, w=w1, mu=mu1,cov=cov1)
    l1 = [1] * len(r1)
    data_100_datapoints = np.append(r0, r1, axis=0)
    data_100_labels = l0 + l1
    # 1000 dataset
    l0, l1 = sample_distribution(1000, p0, p1)
    r0 = generate_gaussian_samples(num_sample=l0, w=w0, mu=mu0,cov=cov0)
    l0 = [0] * len(r0)
    r1 = generate_gaussian_samples(num_sample=l1, w=w1, mu=mu1,cov=cov1)
    l1 = [1] * len(r1)
    data_1000_datapoints = np.append(r0, r1, axis=0)
    data_1000_labels = l0 + l1
    # 10000 dataset
    l0, l1 = sample_distribution(10000, p0, p1)
    r0 = generate_gaussian_samples(num_sample=l0, w=w0, mu=mu0,cov=cov0)
    l0 = [0] * len(r0)
    r1 = generate_gaussian_samples(num_sample=l1, w=w1, mu=mu1,cov=cov1)
    l1 = [1] * len(r1)
    data_10000_datapoints = np.append(r0, r1, axis=0)
    data_10000_labels = l0 + l1
    # 20000 dataset
    n0, n1 = sample_distribution(20000, p0, p1)
    r0 = generate_gaussian_samples(num_sample=n0, w=w0, mu=mu0,cov=cov0)
    l0 = [0] * len(r0)
    r1 = generate_gaussian_samples(num_sample=n1, w=w1, mu=mu1,cov=cov1)
    l1 = [1] * len(r1)
    data_20000_datapoints = np.append(r0, r1, axis=0)
    data_20000_labels = l0 + l1
    fig0 = plt.figure(0)
    title = "20k datapoints from 2 classes"
    plot_datapoints_scatter(datapoints=data_20000_datapoints,labels=data_20000_labels,xlabel="X1",ylabel="X2",title=title,)
    plt.savefig('20k_ditribution.png', bbox_inches='tight')
    plt.show()
    title = "10k datapoints from 2 classes"
    plot_datapoints_scatter(datapoints=data_10000_datapoints,labels=data_10000_labels,xlabel="X1",ylabel="X2",title=title,)
    plt.savefig('10k_ditribution.png', bbox_inches='tight')
    plt.show()
    title = "1000 datapoints from 2 classes"
    plot_datapoints_scatter(datapoints=data_1000_datapoints,labels=data_1000_labels,xlabel="X1",ylabel="X2",title=title,)
    plt.savefig('1k_ditribution.png', bbox_inches='tight')
    plt.show()
    title = "100 datapoints from 2 classes"
    plot_datapoints_scatter(datapoints=data_100_datapoints,labels=data_100_labels,xlabel="X1",ylabel="X2",title=title,)
    plt.savefig('100_ditribution.png', bbox_inches='tight')
    plt.show()
    # PART A
    print("-" * 10 + "PART A" + "-" * 10)
    #theoritical gamma at 1.5 calculating discriminant scores for data_20000
    data_20000_discriminant = calculate_discriminant(datapoints=data_20000_datapoints,mu0=mu0,w0=(0.5, 0.5),cov0=cov0,mu1=mu1,cov1=cov1,)
    data_20000_discriminant = data_20000_discriminant + [0]
    false_positive, true_positive, gamma_values, prob_error = hit_cls(data_20000_discriminant, data_20000_labels)
    min_error, min_index = min(prob_error), prob_error.index(min(prob_error))
    print("experimental_gamma ", np.exp(gamma_values[min_index]))
    print("experimental_min_error ", min_error)
    no_0_labels = calculate_no_labels(data_labels=data_20000_labels,label_value=0)
    no_1_labels = calculate_no_labels(data_labels=data_20000_labels,label_value=1)
    data_20000_discriminant_label0 = data_20000_discriminant[0:no_0_labels]
    data_20000_discriminant_label1 = data_20000_discriminant[no_0_labels:]
    theoritical_fp = (len([j for j in data_20000_discriminant_label0 if j >= (p1 / p0)]) /no_0_labels)
    theoritical_tp = (len([j for j in data_20000_discriminant_label1 if j >= (p1 / p0)]) /no_1_labels)
    print("theoritical min_error ", theoritical_fp * p0 + (1 - theoritical_tp) * p1)
    fig1 = plt.figure(1)
    plt.plot(false_positive, true_positive, label="ROC_CURVE")
    plt.plot(false_positive[min_index],true_positive[min_index],"ro",label="Experimental min error",)
    plt.plot(theoritical_fp, theoritical_tp, "g+", label="Theorical min error")
    plt.title("Minimum Expected risk ROC_CURVE")
    plt.xlabel("P(False Positive)")
    plt.ylabel("P(Correct prediction)")
    plt.legend()
    plt.savefig('Minimum Expected risk ROC_CURVE.png', bbox_inches='tight')
    plt.show()
    # # PART B
    print("-" * 10 + "PART B" + "-" * 10)
    print("-" * 5 + "TRAINING WITH 10000 data points" + "-" * 5)
    GMMmodel, GMMmodel2 = find_estimated_params(train_data_points=data_10000_datapoints,train_data_labels=data_10000_labels)
    mle_mu0 = GMMmodel.means_
    mle_cov0 = GMMmodel.covariances_
    mle_mu1 = GMMmodel2.means_
    mle_cov1 = GMMmodel2.covariances_
    mle_alpha0 = GMMmodel.weights_
    mle_alpha1 = GMMmodel2.weights_
    print("estimated parameters")
    print(f"mu0 {mle_mu0}")
    print(f"cov0 {mle_cov0}")
    print(f"alpha0 {mle_alpha0}")
    print(f"mu1 {mle_mu1}")
    print(f"cov1 {mle_cov1}")
    print(f"alpha1 {mle_alpha1}")
    n0 = calculate_no_labels(data_labels=data_10000_labels, label_value=0)
    print(f"p(0) = {n0/10000} and p(1) = {1-(n0/10000)}")
    data_20000_discriminant = calculate_discriminant(datapoints=data_20000_datapoints,mu0=mle_mu0,w0=(0.5, 0.5),cov0=mle_cov0,mu1=mle_mu1,cov1=mle_cov1,)
    data_20000_discriminant = data_20000_discriminant + [0]
    false_positive, true_positive, gamma_values, prob_error = hit_cls(data_20000_discriminant, data_20000_labels)
    min_error, min_index = min(prob_error), prob_error.index(min(prob_error))
    print("experimental_gamma ", np.exp(gamma_values[min_index]))
    print("experimental_min_error ", min_error)
   
    plt.plot(false_positive, true_positive, label="ROC_CURVE 10000 dataset",c="r")
    plt.plot(false_positive[min_index],true_positive[min_index],"ro",
    label="Experimental min error for dataset 10000",)
    print("-" * 5 + "TRAINING WITH 1000 data points" + "-" * 5)
    GMMmodel, GMMmodel2 = find_estimated_params(train_data_points=data_1000_datapoints,train_data_labels=data_1000_labels)
    mle_mu0 = GMMmodel.means_
    mle_cov0 = GMMmodel.covariances_
    mle_mu1 = GMMmodel2.means_
    mle_cov1 = GMMmodel2.covariances_
    mle_alpha0 = GMMmodel.weights_
    mle_alpha1 = GMMmodel2.weights_
    print("estimated parameters")
    print(f"mu0 {mle_mu0}")
    print(f"cov0 {mle_cov0}")
    print(f"alpha0 {mle_alpha0}")
    print(f"mu1 {mle_mu1}")
    print(f"cov1 {mle_cov1}")
    print(f"alpha1 {mle_alpha1}")
    n0 = calculate_no_labels(data_labels=data_1000_labels, label_value=0)
    print(f"p(0) = {n0/10000} and p(1) = {1-(n0/10000)}")
    data_20000_discriminant = calculate_discriminant(datapoints=data_20000_datapoints,mu0=mle_mu0,w0=(0.5, 0.5),cov0=mle_cov0,mu1=mle_mu1,cov1=mle_cov1,)
    data_20000_discriminant = data_20000_discriminant + [0]
    false_positive, true_positive, gamma_values, prob_error = hit_cls(data_20000_discriminant, data_20000_labels)
    min_error, min_index = min(prob_error), prob_error.index(min(prob_error))
    print("experimental_gamma ", np.exp(gamma_values[min_index]))
    print("experimental_min_error ", min_error)
    plt.plot(false_positive, true_positive, label="ROC_CURVE 1000 dataset",c="g")
    plt.plot(false_positive[min_index],true_positive[min_index],"go",
    label="Experimental min error for dataset 1000 ",)
  
    print("-" * 5 + "TRAINING WITH 100 data points" + "-" * 5)
    GMMmodel, GMMmodel2 = find_estimated_params(train_data_points=data_100_datapoints, train_data_labels=data_100_labels)
    mle_mu0 = GMMmodel.means_
    mle_cov0 = GMMmodel.covariances_
    mle_mu1 = GMMmodel2.means_
    mle_cov1 = GMMmodel2.covariances_
    mle_alpha0 = GMMmodel.weights_
    mle_alpha1 = GMMmodel2.weights_
    print("estimated parameters")
    print(f"mu0 {mle_mu0}")
    print(f"cov0 {mle_cov0}")
    print(f"alpha0 {mle_alpha0}")
    print(f"mu1 {mle_mu1}")
    print(f"cov1 {mle_cov1}")
    print(f"alpha1 {mle_alpha1}")
    n0 = calculate_no_labels(data_labels=data_100_labels, label_value=0)
    print(f"p(0) = {n0/10000} and p(1) = {1-(n0/10000)}")
    data_20000_discriminant = calculate_discriminant(datapoints=data_20000_datapoints,mu0=mle_mu0,w0=(0.5, 0.5),cov0=mle_cov0,mu1=mle_mu1,cov1=mle_cov1,)
    data_20000_discriminant = data_20000_discriminant + [0]
    false_positive, true_positive, gamma_values, prob_error = hit_cls(data_20000_discriminant, data_20000_labels)
    min_error, min_index = min(prob_error), prob_error.index(min(prob_error))
    print("experimental_gamma ", np.exp(gamma_values[min_index]))
    print("experimental_min_error ", min_error)
    plt.plot(false_positive, true_positive, label="ROC_CURVE 100 dataset",c="b")
    plt.plot(false_positive[min_index],true_positive[min_index],"bo",label="Experimental min error for dataset 100",)
    
    plt.title("Minimum Expected risk roc for training data")
    plt.xlabel("P(False Positive)")
    plt.ylabel("P(Correct prediction)")
    plt.legend()
    plt.savefig('part2_all.png', bbox_inches='tight')
    plt.show()
    #exit()
    # PART C
    print("-" * 10 + "PART C" + "-" * 10)
    data_20000_datapoints, data_20000_labels = data_points_labels_preproc(data_points=data_20000_datapoints, data_labels=data_20000_labels)
    data_10000_datapoints, data_10000_labels = data_points_labels_preproc(data_points=data_10000_datapoints, data_labels=data_10000_labels)
    data_1000_datapoints, data_1000_labels = data_points_labels_preproc(data_points=data_1000_datapoints, data_labels=data_1000_labels)
    data_100_datapoints, data_100_labels = data_points_labels_preproc(data_points=data_100_datapoints, data_labels=data_100_labels)
    w_100, decisions_100 = find_theta(train_data=data_100_datapoints,alpha=0.01,iterations=2000,train_labels=data_100_labels,test_data=data_20000_datapoints,type_='l',)
    w_1000, decisions_1000 = find_theta(train_data=data_1000_datapoints,alpha=0.01,iterations=2000,train_labels=data_1000_labels,test_data=data_20000_datapoints,type_='l',)
    w_10000, decisions_10000 = find_theta(train_data=data_10000_datapoints,alpha=0.01,iterations=2000,train_labels=data_10000_labels,test_data=data_20000_datapoints,type_='l',)
    for decisions in [decisions_100, decisions_1000, decisions_10000]:
        x00 = [i for i in range(20000)
        if (data_20000_labels[0, i] == 0 and decisions[0, i] == 0)]
        x11 = [i for i in range(20000)
        if (data_20000_labels[0, i] == 1 and decisions[0, i] == 1)]
        print(1 - ((len(x00) + len(x11)) / 20000))
    plot_boundary(np.vstack((data_20000_labels[0, :], decisions_100)),w_100.T,'l',data_20000_datapoints,)
    plot_boundary(np.vstack((data_20000_labels[0, :], decisions_1000)),w_1000.T,'l',data_20000_datapoints,)
    plot_boundary(np.vstack((data_20000_labels[0, :], decisions_10000)),w_10000.T,'l',data_20000_datapoints,)
    w_100, decisions_100 = find_theta(train_data=data_100_datapoints,alpha=0.01,iterations=2000,train_labels=data_100_labels,test_data=data_20000_datapoints,type_='r',)
    w_1000, decisions_1000 = find_theta(train_data=data_1000_datapoints,alpha=0.01,iterations=2000,train_labels=data_1000_labels,test_data=data_20000_datapoints,type_='r',)
    w_10000, decisions_10000 = find_theta(train_data=data_10000_datapoints,alpha=0.01,iterations=2000,train_labels=data_10000_labels,test_data=data_20000_datapoints,type_='r',)
    for decisions in [decisions_100, decisions_1000, decisions_10000]:
        x00 = [i for i in range(20000)
        if (data_20000_labels[0, i] == 0 and decisions[0, i] == 0)]
        x11 = [i for i in range(20000)
        if (data_20000_labels[0, i] == 1 and decisions[0, i] == 1)]
        print(1 - ((len(x00) + len(x11)) / 20000))
    plot_boundary(
    np.vstack((data_20000_labels[0, :], decisions_100)),w_100.T,'r',data_20000_datapoints,)
    plot_boundary(np.vstack((data_20000_labels[0, :], decisions_1000)),w_1000.T,'r',data_20000_datapoints,)
    plot_boundary( np.vstack((data_20000_labels[0, :], decisions_10000)),w_10000.T,'r',data_20000_datapoints,)