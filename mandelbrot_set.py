import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def simple_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iter):
    real_vals = np.linspace(real_low, real_high, width)
    imag_vals = np.linspace(imag_low, imag_high, height)
    mandelbrot_graph = np.ones((height, width), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            c= np.complex64(real_vals[x]+imag_vals[y]*1j)
            z=np.complex64(0)
            for i in range(max_iter):
                if abs(z) > 2:
                    mandelbrot_graph[y, x] = i
                    break
                z = z*z + c
    return mandelbrot_graph


if __name__ == "__main__":
    t1 = time()
    mandel = simple_mandelbrot(512,512,-2,2,-2,2,256)
    t2 = time()
    mandel_time = t2 - t1
    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel,extent=(-2, 2, -2, 2), cmap='hot', interpolation='bilinear')
    plt.savefig('mandelbrot.png', dpi=300)
    t2 = time()
    dump_time = t2 - t1
    print('it took %f seconds to compute the mandelbrot set' % mandel_time)
    print('it took %f seconds to dump the image' % dump_time)