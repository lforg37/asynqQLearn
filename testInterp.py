from improc import NearestNeighboorInterpolator2D
import numpy as np
from PIL import Image

sample = Image.open("sample.png")
sampleData = np.array(sample)
interpolator = NearestNeighboorInterpolator2D(sampleData.shape[0:2], [320, 180])
interpolated = interpolator.interpolate(sampleData)
Image.fromarray(interpolated).convert("L").save('big.png')

sample = Image.open('bulbasaur.png')
sampleData = np.array(sample)
interpolator = NearestNeighboorInterpolator2D(sampleData.shape[0:2], [320, 180])
interpolated = interpolator.interpolate(sampleData)
Image.fromarray(interpolated).convert("RGB").save('big_rgb.png')
