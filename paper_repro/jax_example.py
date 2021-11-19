import jax

jax.profiler.start_trace("./tmp/tensorboard")

# Run the operations to be profiled
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5000, 5000))
y = x @ x
y.block_until_ready()

jax.profiler.stop_trace()