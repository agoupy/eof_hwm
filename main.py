import numpy as np
import matplotlib.pyplot as plt
from pyhwm2014 import HWM14, HWM14Plot

fig = plt.figure()

plt.axis([-60,60,0,200])
plt.title('Zonal wind during the 1/1/2003')
z=np.arange(0,201)
for i_hour in np.arange(0,23.5,0.5):
    hwm14Obj = HWM14( altlim=[0,200], altstp=1, ap=[-1, 35], day=1,
            option=1, ut=i_hour, verbose=False, year=2003 )
    plt.plot(hwm14Obj.Uwind,z,color=(0.8,0.5,0))    
    plt.pause(0.1)
