import numpy as np
import numpy.linalg as lng
import numpy.matlib as mtlb
import matplotlib.pyplot as plt
from pyhwm2014 import HWM14, HWM14Plot



# Collect the data
data_year=2006
data_day=196  				#15 of July
data_lat=-0.180653			#latitude of Quito
data_lon=-78.467834			#longitude of Quito
data_zmin=0
data_zmax=200
data_N=1460				#number of profiles 
data_M=data_zmax-data_zmin+1		#number of points

data_U=np.zeros((data_M,data_N)) 	#Contain the N profils each one has M points
data_V=np.zeros((data_M,data_N))        #Contain the N profils each one has M points

i_count=1
for i_day in np.arange(0,365,1):
	for i_hour in np.arange(0,18,6):
	    hwm14Obj = HWM14( altlim=[data_zmin,data_zmax], altstp=1, ap=[-1, 35], 
		glat=data_lat, glon=data_lon, day=i_day, option=1, ut=i_hour, 
		verbose=False, year=data_year )
	    data_U[:,i_count]=hwm14Obj.Uwind
	    data_V[:,i_count]=hwm14Obj.Vwind
	    i_count+=1

# Compute the mean and substract from the data
data_U_mean=data_U.mean(1)
data_V_mean=data_V.mean(1)
data_U=data_U-np.transpose(mtlb.repmat(data_U_mean,data_N,1))
data_V=data_V-np.transpose(mtlb.repmat(data_V_mean,data_N,1))

# Compute the svd
U, s, V = np.linalg.svd(data_U, full_matrices=True)
U2, s2, V2 = np.linalg.svd(data_V, full_matrices=True)

# Partial reconstruction of the matrix
U_4=reconst_matrix = np.dot(U[:,:4],np.dot(np.diag(s[:4]),V[:4,:]))
U_8=reconst_matrix = np.dot(U[:,:8],np.dot(np.diag(s[:8]),V[:8,:]))
U_16=reconst_matrix = np.dot(U[:,:16],np.dot(np.diag(s[:16]),V[:16,:]))

V_4=reconst_matrix = np.dot(U2[:,:4],np.dot(np.diag(s2[:4]),V2[:4,:]))
V_8=reconst_matrix = np.dot(U2[:,:8],np.dot(np.diag(s2[:8]),V2[:8,:]))
V_16=reconst_matrix = np.dot(U2[:,:16],np.dot(np.diag(s2[:16]),V2[:16,:]))


# Extract the original and reconstructed profile ind_ref
ind_ref=784
ref_U=data_U[:,ind_ref]+data_U_mean
ref_U_4=U_4[:,ind_ref]+data_U_mean
ref_U_8=U_8[:,ind_ref]+data_U_mean
ref_U_16=U_16[:,ind_ref]+data_U_mean

ref_V=data_V[:,ind_ref]+data_V_mean
ref_V_4=V_4[:,ind_ref]+data_V_mean
ref_V_8=V_8[:,ind_ref]+data_V_mean
ref_V_16=V_16[:,ind_ref]+data_V_mean

# Plot the result
z=np.arange(data_zmin,data_zmax+1)
fig = plt.figure()
fig.set_size_inches(12, 14)
plt.subplot(221)
plt.plot(np.true_divide(np.abs(s),np.sum(s)),'b+-')
plt.title('Zonal EOF weights')
plt.subplot(223)
plt.plot(np.true_divide(np.abs(s2),np.sum(s2)),'b+-')
plt.title('Meridional EOF weights')
plt.subplot(222)
u1,=plt.plot(ref_U,z,'k',Linewidth=5,label='Measured profil')
u4,=plt.plot(ref_U_4,z,'r',label='4 terms')
u8,=plt.plot(ref_U_8,z,'y',label='8 terms')
u16,=plt.plot(ref_U_16,z,'g',label='16 terms')
plt.legend(handles=[u1, u4, u8, u16],bbox_to_anchor=(0, 0), loc=3)
plt.title('Reconstructed Zonal Wind')
plt.subplot(224)
v1,=plt.plot(ref_V,z,'k',Linewidth=5,label='Measured profil')
v4,=plt.plot(ref_V_4,z,'r',label='4 terms')
v8,=plt.plot(ref_V_8,z,'y',label='8 terms')
v16,=plt.plot(ref_V_16,z,'g',label='16 terms')
plt.legend(handles=[v1, v4, v8, v16],bbox_to_anchor=(0, 0), loc=3)
plt.title('Reconstructed Meridional Wind')
plt.show()

