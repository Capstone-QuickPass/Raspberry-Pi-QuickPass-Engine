import time
import busio
import board
import adafruit_amg88xx
import matplotlib.pyplot as plt
import numpy as np
i2c = busio.I2C(board.SCL, board.SDA)
amg = adafruit_amg88xx.AMG88XX(i2c)
mlx_shape = (8,8)
plt.ion()
fig,ax = plt.subplots(figsize=(12,7))
therm1 = ax.imshow(np.zeros(mlx_shape),vmin=30,vmax=40)
cbar = fig.colorbar(therm1)
cbar.set_label('Temperature [$^{\circ}$C]',fontsize=14)


def detect_temp(t):
    frame = np.zeros((8*8,))
    t_array = []
    t_end = time.time() + t
    max_temp = 0
    while time.time() < t_end:
        t1 = time.monotonic()
        try:
            frame = amg.pixels
            print(amg.pixels)
            max_temp = max(map(max, amg.pixels))
            data_array = (np.reshape(frame,mlx_shape))
            therm1.set_data(np.fliplr(data_array))
            therm1.set_clim(vmin=np.min(data_array),vmax=np.max(data_array))
            cbar.on_mappable_changed(therm1)
            plt.pause(0.001)
            
            t_array.append(time.monotonic()-t1)
            print('Sample Rate: {0:2.1f}fps'.format(len(t_array)/np.sum(t_array)))
        except ValueError:
            continue
        
    return (max_temp)
