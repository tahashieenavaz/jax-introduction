import jax
import jax.numpy as jnp
from jax import grad
from dataclasses import dataclass

@dataclass
class Configuration: 
    lr = 0.1
    epochs = 200

def predict(params, x):
    w,b = params
    return w * x + b

def mse(params, x, y):
    preds = predict(params, x)
    return jnp.mean((preds - y) ** 2)


a = 2.0
b = -3.5

key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100, 1))
y = a * x[:, 0] + b + jax.random.normal(key, (100,))

params = [jnp.array(0.0), jnp.array(0.0)]

for epoch in range(Configuration.epochs):
    gradients = grad(mse)(params, x[:, 0], y)
    params = [parameter - Configuration.lr * gradient for parameter, gradient in zip(params, gradients)]