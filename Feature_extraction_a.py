#%% imports
from MNIST_Dataloader import MnistDataloader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% load MNIST data
training_images_filepath =  'train-images.idx3-ubyte'
training_labels_filepath =  'train-labels.idx1-ubyte'
test_images_filepath = 't10k-images.idx3-ubyte'
test_labels_filepath =  't10k-labels.idx1-ubyte'


mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


# %% check size
print(len(x_train))
print(len(y_train))

# %% convert to np array, use only 4, 7, 8 images
y_train = np.array(y_train)
x_train = np.array(x_train)
subset = np.where((y_train == 4) | (y_train == 7) | (y_train == 8))
y_train_a = y_train[subset[0]]
x_train_a = x_train[subset[0]]


# %%
x_train_a = x_train_a.reshape(-1, 28**2)
x_train_mean = x_train_a.mean(axis=0)
print(x_train_mean.shape)

# %% compute covariance matrix
S = np.zeros((784,784))
for i in range(len(x_train_a)):
    S += np.matmul((x_train_a[i]-x_train_mean).reshape(784,1),(x_train_a[i]-x_train_mean).reshape(1,784))
S /= len(x_train_a)


# %% print summary statistics
print(S.min())
print(S.max())
print(S.mean())

# %% compute eigenvectors/eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(S)

# %% plot eigenvalues
plt.plot(eigenvalues)

# %% check length, unique eigenvalues 
print(len(eigenvalues))
print(len(set(eigenvalues)))
print(eigenvectors.shape)

# %% construct PCA matrix

def PCAMatrix(num_components, eigenvalues, eigenvectors):
    sorted = np.argsort(eigenvalues)
    A = np.zeros((num_components, 784))
    for j in range(num_components):
        A[j] = eigenvectors[sorted[j]]
                            
    return A


# %% map the train data to the 3d PCA space

def construct(dimensions, eigenvalues, eigenvectors):
    A = PCAMatrix(dimensions, eigenvalues, eigenvectors)
    y = np.zeros((len(x_train_a),dimensions))
    print(y.shape)
    for i in range(len(x_train_a)):
        y[i] = np.matmul(A,x_train_a[i].reshape(784,1)).reshape(1,dimensions)
    return A, y

def reconstruct(A,y, dims):
    x_prime = np.zeros((len(x_train_a),784))
    dist_err = 0
    for i in range(len(x_train_a)):
        x_prime[i] = np.matmul(np.transpose(A),y[i].reshape(dims,1)).reshape(1,784)
        dist_err += (x_prime[i] - x_train_a[i])**2
    dist_err /= len(x_train_a)
    return x_prime, dist_err


# %% deconstruct and reconstruct for a number of components
dims = [3, 5, 10, 50, 100, 200, 300, 500, 784]
errs = []
x_primes = []

for dim in dims:
    A, y = construct(dim, eigenvalues, eigenvectors)
    x_prime, err = reconstruct(A, y, dim)
    errs.append(err.mean())
    x_primes.append(x_prime)

plt.plot(dims, errs)

# %%
