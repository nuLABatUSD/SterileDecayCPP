import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import ODESolve as ODE
import decays
import csv


sterile_mass = 300
mixing_angle = 1.22e-5
electron_mass = 0.511
muon_mass = 105.658
neutral_pion_mass = 134.9768
charged_pion_mass = 139.57039
a_end = 1
temp_end = 1 / a_end


@nb.jit(nopython=True)
def get_bins(a_end:float):
    return int(np.floor(50 * a_end + 1))

@nb.jit(nopython=True)
def get_a_separations(energies_cm:np.ndarray):
    a_separations = []
    # kinetic terms
    gamma_muon3, muon_speed3, _ = decays.get_kinetic_terms(m0=sterile_mass, m1=muon_mass, m2=charged_pion_mass)
    gamma_pion_mu, pion_speed_mu, _ = decays.get_kinetic_terms(m0=sterile_mass,m1=charged_pion_mass,m2=muon_mass)
    gamma_pion_e, pion_speed_e, _ = decays.get_kinetic_terms(m0=sterile_mass,m1=charged_pion_mass,m2=electron_mass)
    gamma_muon4, muon_speed4, _ = decays.get_kinetic_terms(m0=charged_pion_mass, m1=muon_mass)
    # type I accounting
    nenergy1 = decays.get_decay_monoenergy(m0=sterile_mass)
    nenergy2 = decays.get_decay_monoenergy(m0=sterile_mass, m2=neutral_pion_mass)
    # type II accounting
    neutrino_energy_pion = decays.get_decay_monoenergy(m0=charged_pion_mass, m2=muon_mass)
    min_energy_mu = gamma_pion_mu * neutrino_energy_pion * (1 - pion_speed_mu)
    max_energy_mu = gamma_pion_mu * neutrino_energy_pion * (1 + pion_speed_mu)
    min_energy_e = gamma_pion_e * neutrino_energy_pion * (1 - pion_speed_e)
    max_energy_e = gamma_pion_e * neutrino_energy_pion * (1 + pion_speed_e)
    # type III accounting
    neutrino_max_energy_muon = decays.get_decay_monoenergy(m0=muon_mass, m2=electron_mass)
    nenergy3 = gamma_muon3 * (1 + muon_speed3) * neutrino_max_energy_muon
    nenergy32 = gamma_muon3 * (1 - muon_speed3) * neutrino_max_energy_muon
    #type IV accounting
    nenergy4 = neutrino_max_energy_muon * (gamma_pion_mu * (1 + pion_speed_mu) * gamma_muon4 * (1 + muon_speed4))
    nenergy42 = neutrino_max_energy_muon * (gamma_pion_e * (1 + pion_speed_e) * gamma_muon4 * (1 + muon_speed4))
    for energy in energies_cm:
        a_separations.append(energy / nenergy1)
        if sterile_mass > neutral_pion_mass:
            a_separations.append(energy / nenergy2)
        if sterile_mass >= (charged_pion_mass + electron_mass):
            a_separations.append(energy / min_energy_e)
            a_separations.append(energy / max_energy_e)
            #a_separations.append(energy / nenergy42)
        if sterile_mass >= (charged_pion_mass + muon_mass):
            a_separations.append(energy / nenergy3)
            a_separations.append(energy / nenergy32)
            a_separations.append(energy / nenergy4)
            a_separations.append(energy / min_energy_mu)
            a_separations.append(energy / max_energy_mu)
    removes = []
    a_separations.sort()
    for i in range(len(a_separations) - 1):
        if a_separations[i] > a_end:
            a_separations[i] = a_end
        if a_separations[i] <= 1/ 10 or np.abs(a_separations[i] - a_separations[i + 1]) <= 1e-6 or a_separations[i] == a_end:
            removes.append(a_separations[i])
    if a_separations[-1] > a_end:
        a_separations.remove(a_separations[-1])
    for rem in removes:
        a_separations.remove(rem)
    a_separations.sort()
    return a_separations


@nb.jit(nopython=True)
def compute_evolution(energies_cm:np.ndarray):
    a_separations = get_a_separations(energies_cm=energies_cm)
    a0 = 1 / 10
    da0 = 0.01 * a0
    t0 = 0
    temp0 = 1 / a0
    num_bins = len(energies_cm)

    # calculating initial sterile neutrino number density
    gwd = 10.75
    gsdec = 61.75
    zeta3 = 1.2020569031595942853
    multiplier = 3 * zeta3 / (2 * np.pi ** 2)
    ns0 =  gwd * multiplier * (temp0 ** 3) / gsdec

    fe = np.exp(-energies_cm) / (np.exp(-energies_cm) + 1)
    y0 = np.zeros(6 * num_bins + 3)
    for i in range(5):
        y0[i * num_bins:(i+1)* num_bins]=fe
    y0[-3] = ns0
    y0[-2] = t0
    y0[-1] = temp0

    p = np.zeros(len(energies_cm) + 2)
    p[0:-2] = energies_cm
    p[-2] = sterile_mass
    p[-1] = np.sin(mixing_angle) ** 2
    results = [ODE.ODEOneRun(a0, y0, da0, p, 100, 2, a_separations[0])]
    a_all = results[0][0]
    freqs = results[0][1][:,:-3]
    ns_all = results[0][1][:,-3]
    t_all = results[0][1][:,-2]
    T_all = results[0][1][:,-1]
    da_all = results[0][2]
    

    for i in range(len(a_separations) - 1):
        a0 = a_separations[i] + 1e-14
        da0 = results[-1][2][-1]
        y0 = np.zeros(6 * num_bins + 3)
        for j in range(5):
            y0[j * num_bins:(j+1)* num_bins]=results[i][1][-1,j * num_bins:(j+1)* num_bins]
        y0[-3] = results[i][1][-1,-3]
        y0[-2] = results[i][1][-1,-2]
        y0[-1] = results[i][1][-1,-1]
        result_mod = ODE.ODEOneRun(a0, y0, da0, p, 100, 2, a_separations[i + 1])
        a_all = np.concatenate((a_all, result_mod[0][1:]))
        freqs = np.concatenate((freqs, result_mod[1][:,:-3][1:]))
        ns_all = np.concatenate((ns_all , result_mod[1][:,-3][1:]))
        t_all = np.concatenate((t_all , result_mod[1][:,-2][1:]))
        T_all = np.concatenate((T_all , result_mod[1][:,-1][1:]))
        da_all = np.concatenate((da_all, result_mod[2][1:]))
        results.append(result_mod)

    a0 = a_separations[-1] + 1e-15
    da0 = results[-1][2][-1]
    y0 = np.zeros(6 * num_bins + 3)
    for j in range(5):
        y0[j * num_bins:(j+1)* num_bins]=results[j][1][-1,j * num_bins:(j+1)* num_bins]
    y0[-3] = results[-1][1][-1,-3]
    y0[-2] = results[-1][1][-1,-2]
    y0[-1] = results[-1][1][-1,-1]
    result_mod = ODE.ODEOneRun(a0, y0, da0, p, 100, 2, a_end)
    a_all = np.concatenate((a_all, result_mod[0][1:]))
    freqs = np.concatenate((freqs, result_mod[1][:,:-3][1:]))
    ns_all = np.concatenate((ns_all , result_mod[1][:,-3][1:]))
    t_all = np.concatenate((t_all , result_mod[1][:,-2][1:]))
    T_all = np.concatenate((T_all , result_mod[1][:,-1][1:]))
    da_all = np.concatenate((da_all, result_mod[2][1:]))
    results.append(result_mod)
    eps = [energies_cm] * len(a_all)
    return a_all, freqs, eps, ns_all, t_all, T_all, da_all

"""
** This can be used to check outputs

num = get_bins(a_end=a_end)
energies_cm = np.linspace(0, (sterile_mass + 1) / (2 * temp_end) , num)
a_all, freqs, eps, ns_all, t_all, T_all, da_all = compute_evolution(energies_cm=energies_cm)
comoving = [1 / a for a in a_all]
fe = freqs[:][:,:num]
print(len(fe[0]))
# Specify the file name
filename = "mass-300-life-1.004.npz"
np.savez(filename, a=a_all, fe=fe, e=eps, ns=ns_all, t=t_all, T=T_all)
plt.loglog(comoving, T_all)
plt.loglog(comoving,comoving)
plt.show()
ratio = T_all / comoving
plt.semilogy(t_all, ns_all)
plt.semilogy(t_all, ratio)
plt.show()
plt.plot(t_all,a_all)
plt.show()
print((comoving[-1] / T_all[-1]))
print(max(t_all) / (1.52e21))"""

