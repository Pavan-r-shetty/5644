import pandas as pd
import numpy as np
from random import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sn

from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

def get_risk(i, x, p, mu, sigma, lambda_matrix):
    #function:To calculate Risk R(ai/x) for a particular choice
    
    total = 0
    for j in range(3):
        if j == 2:
            total += (lambda_matrix[i][j]* p[j]* (0.5 * multivariate_normal.pdf(x, mu[j], sigma[j])+ 0.5 * multivariate_normal.pdf(x, mu[j + 1], sigma[j +1])))
        else:
            total += (lambda_matrix[i][j] * p[j] * multivariate_normal.pdf(x,mu[j], sigma[j]))
    return total

def get_map(normal_dist_id, data_points, lambda_matrix, mu, p, sigma,loss_matrix):
    #function: makes the decision for datapoints for a given normal distribution
    
    correct = list()
    incorrect = list()
    for i in data_points[normal_dist_id]:
        choice = np.argmin(
        [get_risk(0, i, p, mu, sigma, lambda_matrix),get_risk(1, i, p, mu, sigma, lambda_matrix),get_risk(2, i, p, mu, sigma, lambda_matrix),])
        if choice == normal_dist_id:
            correct.append(i)
        else:
            incorrect.append(i)
        loss_matrix[normal_dist_id][choice] += 1
    return np.array(correct), np.array(incorrect)

def hit_mapclf(lambda_matrix, loss_matrix, mu_val, sigma_val, data_points, plot_title,image_name):
    #function: hits the map classifier for each of the class datapoints for a given lambda matrix
    
    fig1 = plt.figure(1)
    ax1 = fig1.gca(projection="3d")
    markers = dict({0: "o", 1: "+", 2: "^"})
    for i in range(3):
        correct, incorrect = get_map(normal_dist_id=i,lambda_matrix=lambda_matrix,mu=mu_val,p=p,sigma=sigma_val,data_points=data_points,loss_matrix=loss_matrix,)
        ax1.scatter(correct[:, 0],correct[:, 1],correct[:, 2],c="g",marker=markers[i],label="correct_class_1_preds",)
        ax1.scatter(incorrect[:, 0],incorrect[:, 1],incorrect[:, 2],c="r",marker=markers[i],label="incorrect_class_1_preds",)
    ax1.set_xlabel("X1")
    ax1.set_ylabel("X2")
    ax1.set_zlabel("X3")
    plt.legend()
    plt.title(plot_title)
    plt.savefig(image_name)
    plt.show()

def calc_total_loss(loss_matrix):
    #function:Calculates the total loss given the confusion matrix
    
    overall_loss = 0
    print(loss_matrix)
    for i in range(len(loss_matrix)):
        for j in range(len(loss_matrix[i])):
            if j == i:
                continue
            overall_loss += loss_matrix[i][j]
    return overall_loss

def get_conf_matrix(loss_matrix):
    #function: converts the conf_matrix to a dataframe
    
    df_loss_mat = pd.DataFrame(loss_matrix, index=[i for i in [1, 2, 3]], columns=[i for i in [1,2, 3]])
    df_loss_mat_percent = pd.DataFrame()
    for index, row in df_loss_mat.iterrows():
        sample_no = sum(row)
        for col_index, val in row.items():
            df_loss_mat_percent.loc[index, col_index] = val / sample_no
    return df_loss_mat, df_loss_mat_percent

if __name__ == "__main__":
    # assign samples to each distribution
    no_samples = 10000
    p = list([0.3, 0.3, 0.4])
    N = list([0, 0, 0, 0])
    for i in range(no_samples):
        val = random()
        if val <= p[0]:
            N[0] += 1
        elif val > p[0] and val <= p[0] + p[1]:
            N[1] += 1
        else:
            N[2] += 1
    N30 = 0
    N31 = 0
    for i in range(N[2]):
        if random() <= 0.5:
            N30 += 1
        else:
            N31 += 1
    N[2] = N30
    N[3] = N31
    print(N)
    
    
    # choosing mean and covariance values
    mu_val = [[7, 4, 4], [4, 7, 4], [4, 4, 7], [3, 3, 3]]
    sigma_val = [
    [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    [[4, 0, 0], [0, 4, 0], [0, 0, 4]],
    ]
    data_points = list()
    
    
    # creation of datapoints for each distribution
    for i in range(4):
        data_points.append(np.random.multivariate_normal(mu_val[i],sigma_val[i], N[i]))
    data_points[2] = np.append(data_points[2], data_points[3], axis=0)
    data_points.pop(3)
    print(len(data_points),data_points[0].shape,data_points[1].shape,data_points[2].shape)
    
    
    # plot the data points
    fig = plt.figure(0)
    ax = fig.gca(projection="3d")
    ax.scatter(data_points[0][:, 0],data_points[0][:, 1],data_points[0][:, 2],"g",label="class 1")
    ax.scatter(data_points[1][:, 0],data_points[1][:, 1],data_points[1][:, 2],"r",label="class 2")
    ax.scatter(data_points[2][:, 0],data_points[2][:, 1],data_points[2][:, 2],"b",label="class 3")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")
    plt.title("True class labels for datapoints")
    plt.savefig("q2_data_distribution.png")
    plt.legend()
    plt.show()
    
    # hitting the map classifier for lambda matrix 0-1-1
    print("*" * 50 + "MAP 0-1-1" + "*" * 50)
    lambda_matrix = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    loss_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hit_mapclf(lambda_matrix=lambda_matrix,loss_matrix=loss_matrix,mu_val=mu_val,sigma_val=sigma_val,data_points=data_points,plot_title="MAP classifier for 0-1 lambda matrix",image_name="q2_map01.png",)
    overall_loss = calc_total_loss(loss_matrix=loss_matrix)
    print(f"overall loss {overall_loss}, overall_loss percent = {overall_loss/ sum(N)}")
    df_loss_mat, df_loss_mat_percent = get_conf_matrix(loss_matrix)
    print("conf_matrix")
    print(df_loss_mat)
    print("conf_matrix_percent")
    print(df_loss_mat_percent)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_loss_mat_percent, annot=True)
    plt.savefig("heat_map_lambda_1.png")
    plt.show()
    
    
    
    # part B
    # hitting the map classifier for lambda matrix 0-1-10
    
    print("*" * 50 + "MAP 0-1-10" + "*" * 50)
    lambda_matrix_2 = [[0, 1, 10], [1, 0, 10], [1, 1, 0]]
    loss_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hit_mapclf(lambda_matrix=lambda_matrix_2,loss_matrix=loss_matrix,mu_val=mu_val,sigma_val=sigma_val,data_points=data_points,plot_title="MAP classifier for 0-1-10 lambda matrix",image_name="q2_map01.png",)
    overall_loss = calc_total_loss(loss_matrix=loss_matrix)
    print(f"overall loss {overall_loss}, overall_loss percent = {overall_loss/ sum(N)}")
    df_loss_mat, df_loss_mat_percent = get_conf_matrix(loss_matrix)
    print("conf_matrix")
    print(df_loss_mat)
    print("conf_matrix_percent")
    print(df_loss_mat_percent)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_loss_mat_percent, annot=True)
    plt.savefig("heat_map_lambda_1.png")
    plt.show()
    
    
    # hitting the map classifier for lambda matrix 0-1-100
    print("-" * 30 + "MAP 0-1-100" + "-" * 30)
    lambda_matrix_3 = [[0, 1, 100], [1, 0, 100], [1, 1, 0]]
    loss_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    hit_mapclf(lambda_matrix=lambda_matrix_3,loss_matrix=loss_matrix,mu_val=mu_val,sigma_val=sigma_val,data_points=data_points,plot_title="MAP classifier for 0-1-100 lambda matrix",image_name="q2_map01.png",)
    overall_loss = calc_total_loss(loss_matrix=loss_matrix)
    print(f"overall loss {overall_loss}, overall_loss percent = {overall_loss/ sum(N)}")
    df_loss_mat, df_loss_mat_percent = get_conf_matrix(loss_matrix)
    print("conf_matrix")
    print(df_loss_mat)
    print("conf_matrix_percent")
    print(df_loss_mat_percent)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_loss_mat_percent, annot=True)
    plt.savefig("heat_map_lambda_1.png")
    plt.show()