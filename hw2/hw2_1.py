import numpy as np
from random import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

if __name__ == '__main__':
    N = 10000
    p0 = 0.65
    p1 = 0.35
    N0 = 0
    N1 = 0
    # number of samples for each distribution
    for i in range(N):
        if random() < p0:
            N0 = N0 + 1
        else:
            N1 = N1 + 1
    print(N0, N1)

    # create mean and covariance matrix 'm ' and 'c' 
    m01 = np.transpose([3, 0])
    cov01 = np.array([[2, 0], [0, 1]])
    m02 = np.transpose([0, 3])
    cov02 = np.array([[1, 0], [0, 2]])
    w1 = 0.5
    w2 = 0.5
    # class 0 is a mixture of 2 gaussians, we need to find the numbe of sample for each distribution
    no_first_distribution = 0
    no_second_distribution = 0
    for i in range(N0):
        if random() < 0.5:
            no_first_distribution += 1
        else:
            no_second_distribution += 1
    print(no_first_distribution, no_second_distribution)
    
    # creation of datapoints and plotting
    first_data_points = np.random.multivariate_normal(m01, cov01, no_first_distribution)
    first_data_points = np.append(first_data_points,np.random.multivariate_normal(m02, cov02, no_second_distribution),axis=0,)
    print(first_data_points)
    fig = plt.figure(0)
    plt.scatter(first_data_points[:, 0], first_data_points[:, 1], c='b',label='class 0')
    m1 = np.transpose([2, 2])
    cov1 = np.array([[1, 0], [0, 1]])
    second_data_points = np.random.multivariate_normal(m1, cov1, N1)
    plt.scatter(second_data_points[:, 0], second_data_points[:, 1], c='r',label='class 1')
    plt.legend()
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("True class labels")
    plt.savefig('q1_data_distibution_PartA_2.png')
    plt.show()
    N0_discriminant = list()
    N1_discriminant = list()
    
    # calculate the discriminant scores
    for i in range(N0):
        N0_discriminant.append(np.log(multivariate_normal.pdf(first_data_points[i], m1, cov1)/ ((w1 * multivariate_normal.pdf(first_data_points[i],m01, cov01)+ w2 *multivariate_normal.pdf(first_data_points[i], m02, cov02)))))
    for i in range(N1):
        N1_discriminant.append(np.log(multivariate_normal.pdf(second_data_points[i], m1, cov1)/ ((w1 * multivariate_normal.pdf(second_data_points[i],m01, cov01)+ w2* multivariate_normal.pdf(second_data_points[i],m02, cov02)))))
    false_positive = list()
    true_positive = list()
    gamma_values = list()
    prob_error = list()
    full_discriminant = N0_discriminant + N1_discriminant + [0]
    print(len(full_discriminant))
    full_discriminant = sorted(full_discriminant)


    # gamma values varied to plot the ROC curve
    for i in full_discriminant:
        fp = len([j for j in N0_discriminant if j >= i]) / N0
        tp = len([j for j in N1_discriminant if j >= i]) / N1
        false_positive.append(fp)
        true_positive.append(tp)
        gamma_values.append(i)
        prob_error.append(fp * p0 + (1 - tp) * p1)
    min_error, min_index = min(prob_error),prob_error.index(min(prob_error))
    print("PART A")
    print("experimental_gamma ", np.exp(gamma_values[min_index]))
    print("experimental_min_error ", min_error)


    # calculate the theoretical min_error
    theoritical_fp = len([j for j in N0_discriminant if j >= (p1 / p0)]) /N0
    theoritical_tp = len([j for j in N1_discriminant if j >= (p1 / p0)]) /N1
    print("theoretical min_error ", theoritical_fp * p0 + (1 -theoritical_tp) * p1)
    
    fig1 = plt.figure(1)
    plt.plot(false_positive, true_positive, label="ROC CURVE")
    plt.plot(false_positive[min_index],true_positive[min_index],"go",label="Experimental minimum error")
    plt.plot(theoritical_fp, theoritical_tp, "r+", label="Theoretical minimum error")
    plt.title("Minimum expected risk ROC Curve")
    plt.xlabel("Probability of False Positive")
    plt.ylabel("Probability of True Positive")
    plt.legend()
    plt.savefig('q1_ERM_ROC_PartA_2.png')
    plt.show()



    # PART B
    # calculating mean and covariance projections 
    m0proj = np.mean(first_data_points, axis=0)
    m1proj = np.mean(second_data_points, axis=0)
    sigma0proj = np.cov(first_data_points, rowvar=False)
    sigma1proj = np.cov(second_data_points, rowvar=False)


    # sb and sw
    sb = (m0proj - m1proj) * np.transpose(m0proj - m1proj)
    sw = sigma0proj + sigma1proj

    # eigenvectors and eigenvalues for sw^-1*b
    w, v = np.linalg.eig(np.linalg.inv(sw) * sb)
    max_eigen_index = list(w).index(max(w))


    # LDA for 2 distributions
    wlda = v[:, max_eigen_index]
    ylda0 = np.matmul(wlda, first_data_points.T)
    ylda1 = np.matmul(wlda, second_data_points.T)
    total_lda = list(ylda0) + list(ylda1)
    lda_false_positive = list()
    lda_true_positive = list()
    lda_gamma_values = list()
    lda_prob_error = list()


   

    for i in sorted(total_lda):
        fp = len([j for j in ylda0 if j >= i]) / N0
        tp = len([j for j in ylda1 if j >= i]) / N1
        lda_false_positive.append(fp)
        lda_true_positive.append(tp)
        lda_gamma_values.append(i)
        lda_prob_error.append(fp * p0 + (1 - tp) * p1)
    min_error, min_index = min(lda_prob_error), lda_prob_error.index(min(lda_prob_error))
    print("PART B")
    print("experimental_gamma ", lda_gamma_values[min_index])
    print("experimental_min_error ", min_error)

    theoritical_fp = len([j for j in ylda0 if j >= p1 / p0]) / N0
    theoritical_tp = len([j for j in ylda1 if j >= p1 / p0]) / N1
    print("theoretical min_error ", theoritical_fp * p0 + (1 -theoritical_tp) * p1)

    plt.figure(3)
    plt.plot(lda_false_positive, lda_true_positive, label="ROC CURVE")
    plt.plot(lda_false_positive[min_index],lda_true_positive[min_index],"go",label="Experimental minimum error")
    plt.plot(theoritical_fp, theoritical_tp, "r+", label="Theoretical minimum error")
    plt.title("Minimum expected risk ROC Curve")
    plt.xlabel("Probability of False Positive")
    plt.ylabel("Probability of True Positive")
    plt.legend()
    plt.savefig('q1_LDA_ROC_PART_B.png')
    plt.show()