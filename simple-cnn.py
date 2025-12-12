from flax import linen as nn
from jax.numpy import jnp

class SimpleConvolutionalLayer(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(3, 16, kernel_size=3) 
        x = nn.relu(x)
        x = nn.Conv(16, 32, kernel_size=3) 
        x = nn.relu(x)
        return x
