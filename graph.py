import matplotlib.pyplot as plt
import numpy as np
import csv

import numpy.polynomial.legendre as lg

eps1 = []
f1 = []
eps2 = []
f2 = []
with open(f'SterileDecayCPP2/aaa.csv', 'r') as mod:
        reader = csv.reader(mod)
        for line in reader:
            eps1.append(float(line[0]))
            f1.append(float(line[1]))
            eps2.append(float(line[2]))
            f2.append(float(line[3]))

plt.plot(eps1, f1, color="blue")
plt.plot(eps2, f2, color = "red")
plt.show()



scale = []
temp = []
time = []
ns = []
for i in range(1446):
    with open(f'SterileDecayCPP2/output{i}.csv', 'r') as mod:
        reader = csv.reader(mod)
        for line in reader:
            scale.append(float(line[0]))
            temp.append(float(line[-1]))
            time.append(float(line[-2]))
            ns.append(float(line[-3]))

        

sm_scale=[]
sm_temp=[]
sm_time =[]
for i in range(78):
    with open(f'SterileDecayCPP2/ou{i}.csv', 'r') as mod:
        reader = csv.reader(mod)
        for line in reader:
            sm_scale.append(float(line[0]))
            sm_temp.append(float(line[-1]))
            sm_time.append(float(line[-2]))
        

plt.semilogy(scale, ns, color='mediumpurple')
plt.ylabel("Sterile Number Density")
plt.xlabel("a")
plt.show()
Tcms = [1 / a for a in sm_scale]
Tcmmod = [1 / a for a in scale]
plt.loglog(Tcmmod,Tcmmod,color='black',linestyle='dotted')
plt.loglog(Tcms,sm_temp,color='mediumblue',linestyle='dashed',label='Standard Cosmology')
plt.loglog(Tcmmod, temp,color='mediumpurple',label='Our Model')
plt.xlim(10, 0.04)
plt.ylabel('T (MeV)')
plt.xlabel('$T_{cm} ~(MeV)$')
plt.legend()
plt.show()
#plt.loglog(Tcmmod,Tcmmod,color='black',linestyle='dotted')
time = [t / 1.52e21 for t in time]
print(scale)
print(time)
sm_time = [t / 1.52e21 for t in sm_time]
plt.loglog(sm_time,sm_temp,color='mediumblue',linestyle='dashed',label='Standard Cosmology')
plt.loglog(sm_time, Tcms,color='black',linestyle='dotted')
plt.loglog(time, temp,color='mediumpurple',label='Our Model')
plt.ylabel('T (MeV)')
plt.xlabel('t (sec)')
plt.legend()
plt.show()
plt.loglog(sm_scale,sm_time,color='mediumblue',linestyle='dashed',label='Standard Cosmology')
plt.loglog(scale, time,color='mediumpurple',label='Our Model')
plt.ylabel('t (sec)')
plt.xlabel('a')
plt.legend()
plt.show()
plt.loglog(sm_scale,sm_temp,color='mediumblue',linestyle='dashed',label='Standard Cosmology')
plt.loglog(scale, temp,color='mediumpurple',label='Our Model')
plt.ylabel('T (MeV)')
plt.xlabel('a')
plt.legend()
plt.show()

temp_cm = [1 / a for a in scale]
sm_temp_cm = [1 / a for a in sm_scale]
plt.loglog(temp_cm, temp)
plt.loglog(temp_cm, temp_cm)
plt.loglog(sm_temp_cm, sm_temp)
plt.title("")
plt.show()
print(temp_cm[-1] / temp[-1])
print(temp[-1])
print(time[-1] / 1.52e21)

energies = np.linspace(0, 301 / 2, 101)
energy = []
dp = []
with open("SterileDecayCPP2/spectra.csv", "r") as mod:
    reader = csv.reader(mod)
    for line in reader:
        energy.append(float(line[0]))
        dp.append(float(line[1]))

plt.semilogy(energy, dp)
plt.show()