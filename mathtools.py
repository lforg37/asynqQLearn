from random import random

#Return value sampled from logUniform(10^low, 10^high) 
def logUniform(low, high):
    log_sample = low + random() * (high - low)
    return 10 ** log_sample
