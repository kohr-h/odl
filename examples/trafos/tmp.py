#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:13:01 2017

@author: lagerwer
"""
import numpy as np
import odl
from scipy import signal
import pylab
# %%
detecsize = 16
detecpix = 33

# %%
# fourier transform changes when changing the partition
filter_part = odl.uniform_partition(-detecsize,
                                    detecsize * (1 - 1 / detecpix),
                                    detecpix,
                                    nodes_on_bdry=(False, True))
#filter_part = odl.uniform_partition(0, detecpix, detecpix)
filter_space = odl.uniform_discr_frompartition(filter_part)

# %%
# Create a padded fourier operator for the detector space
resize_detu = int(2**(np.ceil(np.log2(detecpix+1))))
resize = odl.ResizingOperator(filter_space, ran_shp=(resize_detu,))
fourier = odl.trafos.FourierTransform(resize.range, impl='numpy',
                                      halfcomplex=False,
                                      interp='deltas')
padded_fourier = fourier * resize
# %%

# spatial definition of teh ramp filter
mid_det = detecpix - 1
filt = np.zeros(mid_det + 1)
n = np.arange(mid_det + 1)
# tau = pixelwidth
tau = 1
filt[0] = 1 / (4 * tau ** 2)
filt[1::2] = -1 / ( tau ** 2 * np.pi ** 2 * n[1::2] ** 2)
filt = np.append(np.flip(filt, 0), filt[1:-1])


#filt = np.zeros(detecpix)
#filt[16] = 1

f_filt = fourier(filt)

data = np.arange(detecpix)
f_data = padded_fourier(data)
odl_conv = np.sqrt(2 * np.pi)* padded_fourier.inverse(f_data * f_filt)

f_filt_np = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(filt)))
conv = signal.fftconvolve(filt, data, mode='full')
# %%
pylab.figure()
pylab.plot(conv)
pylab.plot(f_filt_np)
#filt_odl.show()
#f, axarr = pylab.subplots(1, 2)
#axarr[0].plot(np.real(f_filt_np))
#axarr[1].plot(np.imag(f_filt_np))
odl_conv.show()
