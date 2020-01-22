from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
psd_kernels = tfp.math.psd_kernels

import matplotlib.pyplot as plt

from utils import data_import 
from utils.forward_op import model

images, labels = data_import.load_observations()
images = images.to_numpy().reshape(-1, 28, 28, 1)
result = model.predict(images[:10])

shape = (28, 28)
#f = lambda x, y: exp(- tf.reduce_sum((x - y)**2, axis=0))


kernel = psd_kernels.ExponentiatedQuadratic(length_scale=5, feature_ndims=2)
index_points = np.indices(shape, dtype=np.float32).T
prior = tfd.GaussianProcess(kernel, index_points=index_points)

# measurement = result[:1]
# def log_prob_fn(x, measurement=measurement, dx=np.product(shape)):
#     return tf.reduce_sum((model.predict(tf.reshape(x, x.shape + [1])) - measurement)**2) * dx

def log_prob_fn(x, data):
    pass

n_samples = 10**4
beta = 0.01
chain = []
acc = []
betas = []
xi = prior.sample(1)
for i in range(n_samples):
    xi_hat = tf.sqrt(1. - beta**2) * xi + beta * prior.sample(1)
    log_prob = tf.minimum(log_prob_fn(xi) - log_prob_fn(xi_hat), 0)
    if tfd.Uniform().sample(1) <= tf.exp(log_prob):
        xi = xi_hat
        acc.append(True)
    else:
        acc.append(False)
    chain.append(xi)
    beta += .005 * (tf.exp(log_prob) - .23)
    beta = tf.clip_by_value(beta, 1e-15, 1 - 1e-15)
    betas.append(beta)


result = tf.reduce_mean(chain, axis=0)
acc_prob = np.mean(acc)
plt.plot(betas)
plt.show()
plt.imshow(result.numpy()[0])
plt.show()

pass
