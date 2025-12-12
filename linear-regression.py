import jax
import jax.numpy as jnp
from jax import grad

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100, 1))

a = 2.0
b = -3.5

y = a * x[:, 0] + b + jax.random.normal(key, (100,))

def predict(params, x):
    w,b =params
    return w * x + b