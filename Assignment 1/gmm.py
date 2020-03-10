import matplotlib.pyplot as plt
import numpy as np
import math


#r = int(input("Enter the number of rows in training mu matrix(k):  "))
r = 3
#c = int(input("Enter the number of collumns in training mu matrix(dimensions): "))
c = 2
print("Enter values for mu matrix separated by 'enter'")
t_mu = np.zeros((r, c))
for i in range(r):
   for j in range(c):
        t_mu[i][j] = float(input())

#t_mu = [[1, 2], [-1, -2], [3, -3]]
print(np.matrix(t_mu))

print("Enter values for a positive semi definite covariance matrix")
t_covar = np.zeros((r, c, c))
for s in range(r):
    for i in range(c):
        for j in range(c):
            t_covar[s][i][j] = float(input())
# t_covar=[[[3. 1.] [1. 2.]][ [2. 0.] [0. 1.]] [[1 0.3] [0.3 1]]
print(np.matrix(t_covar[0]))
print(np.matrix(t_covar[1]))
print(np.matrix(t_covar[2]))

scalingpi = np.zeros(r)
print("Enter values for scaling factorpi")
for i in range(r):
    scalingpi[i] = float(input())
# scalingpi = [1 1 1]
n = int(input("Enter number of values at which pdf has to be evaluvated"))

inp = np.zeros((n, c))
print("Enter values at which pdf has to be evaulvated")
for i in range(n):
    for j in range(c):
        inp[i][j] = float(input())

print(np.matrix(inp))
A = inp.tolist()
print(A)
X = []



def norm(x, mean, cov):
    x_mu = np.subtract(x, mean)
    x_mu = x_mu.T
    print("the shape of x_mu is:")
    print(x_mu.shape)
    print(x_mu.shape[0])
    k1 = x_mu.shape[0]
    print("The shape of sigma is:")

    sigma = np.linalg.inv(cov)
    print(sigma.shape)
    # print(x_mu[1,0])
    # print(x_mu(1,0).shape)
    c2 = np.einsum('...k,kl,...l->...', x - mean, sigma, x - mean)
    c1 = 1 / np.sqrt((2 * np.pi) ** k * np.linalg.det(cov)) * np.exp(-0.5 * c2)
    print(c1.shape)
    #	c1= np.reshape(x_mu.shape[0],1)
    #	np.reshape(c1, x_mu.shape[0])
    #	print(c1.shape)
    return c1


print(X)

#n_samples = 100

#X = []
#for mean, cov in zip(t_mu, t_covar):
#    x = np.random.multivariate_normal(mean, cov, n_samples)
#    X += list(x)

x = np.random.multivariate_normal(t_mu[0], t_covar[0], 100)
X += list(x)
x = np.random.multivariate_normal(t_mu[1], t_covar[1], 100)
X += list(x)
x = np.random.multivariate_normal(t_mu[2], t_covar[2], 200)
X += list(x)

# hardcode this

X = np.array(X)
np.random.shuffle(X)
print("Dataset shape:", X.shape)
# print(X)

x = np.linspace(np.min(X[..., 0]) - 1, np.max(X[..., 0]) + 1, 200)
y = np.linspace(np.min(X[..., 1]) - 1, np.max(X[..., 1]) + 1, 40)
X_, Y_ = np.meshgrid(x, y)
pos = np.array([X_.flatten(), Y_.flatten()]).T
print(pos.shape)
print(np.max(pos[..., 1]))
print(pos)

# define the number of clusters to be learned
k = 10

# create and initialize the cluster centers and the weight paramters
weights = np.ones((k)) / k
means = np.random.choice(X.flatten(), (k, X.shape[1]))
print("the means are")
print(means)
print("First means is")
print(means[1])
print("First means size is")
print(means[1].shape)
type(means)
print(weights)

cov = []
for i in range(k):
    A = np.random.rand(X.shape[1], X.shape[1])
    B = np.dot(A, A.transpose())
    cov.append(B)
cov = np.array(cov)
print(cov.shape)
print(cov)
plog = []
colors = ['tab:red', 'tab:blue', 'tab:green', 'magenta', 'yellow', 'orange', 'brown', 'grey', 'black', 'pink']
eps = 1e-8

# run GMM for 40 steps
for step in range(15):

    # visualize the learned clusters
    if step % 1 == 0:
        plt.figure(figsize=(12, int(8)))
        plt.title("Iteration {}".format(step))
        axes = plt.gca()

        likelihood = []
        for j in range(k):
            print(np.shape(pos))
            n = np.shape(pos)
            print(n)
            print(means[1])
            likelihood.append(norm(x=pos, mean=means[j], cov=cov[j]))

        likelihood = np.array(likelihood)
        #temp1[step] = temp1.append(np.sum(likelihood))
        print("The shape of the intial random likelihood is -")
        print(likelihood.shape)
        print("and its contents are -")
        print(likelihood)
        predictions = np.argmax(likelihood, axis=0)

        for c in range(k):
            pred_ids = np.where(predictions == c)
            plt.scatter(pos[pred_ids[0], 0], pos[pred_ids[0], 1], color=colors[c], alpha=0.2, edgecolors='none',
                        marker='s')

        plt.scatter(X[..., 0], X[..., 1], facecolors='none', edgecolors='grey')

        for j in range(k):
            plt.scatter(means[j][0], means[j][1], color=colors[j])

        plt.show()
    likelihood = []

    # Expectation step
    for j in range(k):
        likelihood.append(norm(x=X, mean=means[j], cov=cov[j]))
    likelihood = np.array(likelihood)
    print(likelihood.shape)
    print("the above is the likelihood shape, nd the below are the contents")
    print(likelihood)
    j=1
    temp1 = norm(x=X, mean=means[j], cov=cov[j])
    plog.append(np.sum(temp1))
    print("content of first plog")
    print(plog)


    print(len(X))
    print(k, len(X))
    assert likelihood.shape == (k, len(X))

    b = []
    # Maximization step
    for j in range(k):
        # use the current values for the parameters to evaluate the posterior
        # probabilities of the data to have been generanted by each gaussian
        b.append((likelihood[j] * weights[j]) / (np.sum([likelihood[i] * weights[i] for i in range(k)], axis=0) + eps))
        # updage mean and variance
        print(b[j].reshape(len(X), 1))
        print("The len of X")
        print(len(X))
        means[j] = np.sum(b[j].reshape(len(X), 1) * X, axis=0) / (np.sum(b[j] + eps))
        cov[j] = np.dot((b[j].reshape(len(X), 1) * (X - means[j])).T, (X - means[j])) / (np.sum(b[j]) + eps)

        # update the weights
        weights[j] = np.mean(b[j])

        assert cov.shape == (k, X.shape[1], X.shape[1])
        assert means.shape == (k, X.shape[1])
        plt.savefig("img10_{0:04d}".format(step), bbox_inches='tight')
        plt.show()
        plt.ion()

#np.array(temp1)
#print(temp1.shape)
print("plog values are")
print(plog)
#print(temp2)
plt.figure(figsize=(12, int(8)))
plt.title("Log Likelihood".format(step))
plt.plot(plog)
plt.savefig("img10loglikelihood".format(step), bbox_inches='tight')
plt.show()