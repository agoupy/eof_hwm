import numpy as np
import numpy.linalg as lng
import matplotlib.pyplot as plt
from pyhwm2014 import HWM14, HWM14Plot



# Collect the data
data_year=2006
data_day=196  				#15 of July
data_lat=-0.180653			#latitude of Quito
data_lon=-78.467834			#longitude of Quito
data_N=1460				#number of profiles 
data_M=201				#number of points

data_U=np.zeros((data_M,data_N)) 	#Contain the N profils each one has M points
data_V=np.zeros((data_M,data_N))        #Contain the N profils each one has M points

i_count=1
for i_day in np.arange(0,365,1):
	for i_hour in np.arange(0,18,6):
	    hwm14Obj = HWM14( altlim=[0,200], altstp=1, ap=[-1, 35], glat=data_lat, glon=data_lon, day=i_day,
        	    option=1, ut=i_hour, verbose=False, year=data_year )
	    data_U[:,i_count]=hwm14Obj.Uwind
	    data_V[:,i_count]=hwm14Obj.Vwind
	    i_count+=1



# Compute the mean and substract from the data
data_U_mean=data_U.mean(1)
data_V_mean=data_V.mean(1)
i_count=1
for i_day in np.arange(0,365,1):
        for i_hour in np.arange(0,18,6):
            data_U[:,i_count]=data_U[:,i_count]-data_U_mean
	    data_V[:,i_count]=data_V[:,i_count]-data_V_mean
            i_count+=1


# Compute the svd
U, s, V = np.linalg.svd(data_U, full_matrices=False)
U2, s2, V2 = np.linalg.svd(data_V, full_matrices=True)

# Use the eigenfunctions as a basis on the 15 of July
ind_ref=784
ref_U=data_U[:,ind_ref]
U_4=data_U_mean
U_8=data_U_mean
U_16=data_U_mean
U_32=data_U_mean
U_128=data_U_mean
for order in np.arange(1,4):
	U_4=U_4+V[order,ind_ref]*s[order]*U[:,order]
for order in np.arange(1,8):
        U_8=U_8+V[order,ind_ref]*s[order]*U[:,order]
for order in np.arange(1,16):
        U_16=U_16+V[order,ind_ref]*s[order]*U[:,order]
for order in np.arange(1,32):
        U_32=U_32+V[order,ind_ref]*s[order]*U[:,order]
for order in np.arange(1,128):
        U_128=U_128+V[order,ind_ref]*s[order]*U[:,order]

ref_V=data_V[:,ind_ref]
V_4=data_V_mean
V_8=data_V_mean
V_16=data_V_mean
V_32=data_V_mean
for order in np.arange(1,4):
        V_4=V_4+V2[order,ind_ref]*s2[order]*U2[:,order]
for order in np.arange(1,8):
        V_8=V_8+V2[order,ind_ref]*s2[order]*U2[:,order]
for order in np.arange(1,16):
        V_16=V_16+V2[order,ind_ref]*s2[order]*U2[:,order]
for order in np.arange(1,32):
        V_32=V_32+V2[order,ind_ref]*s2[order]*U2[:,order]

# Plot the result
z=np.arange(0,201)
fig = plt.figure()
plt.subplot(221)
plt.plot(np.true_divide(np.abs(s),np.sum(s)),'b+-')
plt.title('Zonal EOF weights')
plt.subplot(223)
plt.plot(np.true_divide(np.abs(s2),np.sum(s2)),'b+-')
plt.title('Meridional EOF weights')
plt.subplot(222)
u1,=plt.plot(ref_U,z,'k',Linewidth=2,label='Measured profil')
u4,=plt.plot(U_4,z,'r',label='4 terms')
u8,=plt.plot(U_8,z,'y',label='8 terms')
u16,=plt.plot(U_16,z,'g',label='16 terms')
u32,=plt.plot(U_32,z,'b',label='32 terms')
u128,=plt.plot(U_128,z,'m',label='128 terms')
plt.legend(handles=[u1, u4, u8, u16, u32, u128])
plt.title('Reconstructed Zonal Wind')
plt.subplot(224)
v1,=plt.plot(ref_V,z,'k',Linewidth=2,label='Measured profil')
v4,=plt.plot(V_4,z,'r',label='4 terms')
v8,=plt.plot(V_8,z,'y',label='8 terms')
v16,=plt.plot(V_16,z,'g',label='16 terms')
v32,=plt.plot(V_32,z,'b',label='32 terms')
plt.legend(handles=[v1, v4, v8, v16, v32])
plt.title('Reconstructed Meridional Wind')
plt.show()

#plt.axis([-60,60,0,200])
#plt.title('Zonal wind during the 1/1/2003')
#z=np.arange(0,201)
#for i_hour in np.arange(0,23.5,0.5):
#    hwm14Obj = HWM14( altlim=[0,200], altstp=1, ap=[-1, 35], day=1,
#            option=1, ut=i_hour, verbose=False, year=2003 )
#    plt.plot(hwm14Obj.Uwind,z,color=(0.8,0.5,0))    
#    plt.pause(0.1)
