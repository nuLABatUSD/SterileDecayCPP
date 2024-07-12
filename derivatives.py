import numpy as np
import numba as nb
import numpy.polynomial.laguerre as lg
import decays

num_points = 100
sample_points, weights = lg.laggauss(num_points)

@nb.jit(nopython=True)
def f(a:float, y:np.ndarray, p:np.ndarray):
    der = np.zeros((len(p) - 2) * 6 + 3)
    planck_mass = 1.2 * 10 ** 22
    electron_mass = 0.511
    sterile_mass = p[-2]
    mixing_angle = np.arcsin(np.sqrt(p[-1]))
    temp = y[-1]
    ns = y[-3]
    length = len(p) - 2

    # dt/da calculations
    # sterile neutrino contribution
    sterile_energy_density = sterile_mass * ns
    # photon contribution
    photon_energy_density = (np.pi ** 2) * (y[-1] ** 4) / 15
    # neutrino contribution
    neutrino_energy_density = 0
    neutrino_multiplier = (1 / a ** 4) / (2 * np.pi ** 2)
    cube = np.zeros_like(p[:-2])
    for i, energy in enumerate(p[:-2]):
        cube[i] = energy ** 3 
    for i in range(6):
        integrand = np.multiply(cube, y[i * length: (i + 1) * length])
        add_term = np.trapz(integrand, p[:-2])
        neutrino_energy_density += neutrino_multiplier * add_term

    # electron-positron contribution
    e_multiplier = 2 * (temp ** 4) / (np.pi ** 2)
    x = electron_mass / temp
    e_density_integral = np.sum(weights * np.exp(sample_points) * (sample_points ** 2) *\
                        np.sqrt(sample_points ** 2 + x ** 2) / \
                            (np.exp(np.sqrt(sample_points ** 2 + x ** 2)) + 1))
    e_density = e_multiplier * e_density_integral
    density = photon_energy_density + e_density + neutrino_energy_density + sterile_energy_density
    der[-2] = np.sqrt(3 * (planck_mass ** 2) / (8 * np.pi)) / (a * np.sqrt(density))
    # dns/da calculations
    lifetime = decays.compute_lifetime(sterile_mass=sterile_mass, mixing_angle=mixing_angle)
    der[-3] = -(1 / lifetime) * ns * der[-2] - 3 * ns / a
    
    # df/da calculations
    e, ae, m, am, t, at = decays.compute_full_term(energies_cm=p[:-2],sterile_mass=sterile_mass, temp_cm= 1 / a, mixing_angle=mixing_angle)
    # electron
    dfda_e = e * ns * der[-2]
    der[0 : length] = dfda_e
    # anti-electron
    dfda_ae = ae * ns * der[-2]
    der[length : 2 * length] = dfda_ae
    # muon
    dfda_m = m * ns * der[-2]
    der[2 * length : 3 * length] = dfda_m
    # anti-muon
    dfda_am = am * ns * der[-2]
    der[3 * length : 4 * length] = dfda_am
    # tau
    dfda_t = t * ns * der[-2]
    der[4 * length: 5 * length] = dfda_t
    #anti-tau
    dfda_at = at * ns * der[-2]
    der[5 * length:-3] = dfda_at

    # dT/da calculation
    e_pressure_mutliplier = 2 * (temp ** 4) / (3 * np.pi ** 2)
    e_pressure_integral = np.sum(weights * np.exp(sample_points) * (sample_points ** 4) / \
                                 (np.sqrt(sample_points ** 2 + x ** 2) * \
                                  (np.exp(np.sqrt(sample_points ** 2 + x ** 2)) + 1)))
    e_pressure = e_pressure_mutliplier * e_pressure_integral

    e_dens_temp_multiplier = 2 * (temp ** 3) / (np.pi ** 2)
    e_dens_temp_integral = np.sum(weights * np.exp(sample_points) * (sample_points ** 2) * \
                                  (sample_points ** 2 + x ** 2) * \
                                    np.exp(-np.sqrt(sample_points ** 2 + x ** 2)) / \
                                        (np.exp(-np.sqrt(sample_points ** 2 + x ** 2)) + 1) ** 2)
    e_dens_temp = e_dens_temp_multiplier * e_dens_temp_integral

    e_pres_temp_multiplier = 2 * (temp ** 3) / (3 * (np.pi ** 2))
    e_pres_temp_integral = np.sum(weights * np.exp(sample_points) * (sample_points ** 4) *\
                                  np.exp(-np.sqrt(sample_points ** 2 + x ** 2)) / \
                                    (np.exp(-np.sqrt(sample_points ** 2 + x ** 2)) + 1) ** 2)
    e_pres_temp = e_pres_temp_multiplier * e_pres_temp_integral
    # dQ/da
    total_flow = sterile_mass * ns * (a ** 3) * der[-2] / (lifetime)
    dfda = np.zeros_like(dfda_e)
    for i in range(len(dfda_e)):
        dfda[i] = dfda_e[i] + dfda_ae[i] + dfda_m[i] + dfda_am[i] + dfda_t[i] + dfda_at[i]
    
    integrand = np.multiply(cube, dfda)
    neutrino_loss =  np.trapz(integrand, p[:-2]) / (2 * a * np.pi ** 2)
    dQda = total_flow - neutrino_loss
    # calculating full dT/da
    term_0 = (1 / temp) * dQda
    term_1 = 4 * (np.pi ** 2) * (temp ** 3) / 45
    term_2 = (e_density + e_pressure) / temp
    term_3 = (3 / temp) * term_1
    term_4 = -(1 / temp ** 2) * (e_density + e_pressure)
    term_5 = (1 / temp) * (e_dens_temp + e_pres_temp)
    
    num = term_0 - 3 * (a ** 2) * (term_1 + term_2)
    denom = (a ** 3) * (term_3 + term_4 + term_5)
    der[-1] = num / denom
    return der



    
