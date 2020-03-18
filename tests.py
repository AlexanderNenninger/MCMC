# %%
import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

GT_DIST = tfd.TruncatedNormal(.5, .5, 0, 1)
evidence = GT_DIST.sample(5)

def unnormalized_logprob(x):
    return -tf.reduce_sum((x - evidence)**2)

NUM_RESULTS = int(10e3)
NUM_BURNIN_STEPS = int(1e3)
adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
    tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn = unnormalized_logprob,
        num_leapfrog_steps=3,
        step_size=1.
    ),
    num_adaptation_steps = int(.8 *NUM_BURNIN_STEPS)
)

#%%
@tf.function
def run_chain():
    samples, is_accepted = tfp.mcmc.sample_chain(
        num_results=NUM_RESULTS,
        num_burnin_steps=NUM_BURNIN_STEPS,
        current_state=1.,
        kernel=adaptive_hmc,
        trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
    )

    sample_mean = tf.reduce_mean(samples),
    sample_standard_dev = tf.math.reduce_std(samples)
    is_accepted = tf.reduce_sum(
        tf.cast(is_accepted, dtype=tf.float32)
    )
    return samples, sample_mean, sample_standard_dev, is_accepted

samples, sample_mean, sample_stddev, is_accepted = run_chain()

# %%

plt.hist(samples)
print('mean:{:.4f}  stddev:{:.4f}  acceptance:{:.4f}'.format(
    sample_mean.numpy(), sample_stddev.numpy(), is_accepted.numpy()))


# %%
