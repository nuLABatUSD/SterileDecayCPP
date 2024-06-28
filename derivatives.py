import numpy as np
import numba as nb
import numpy.polynomial.laguerre as lg
import decays

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
    neutrino_energy_density = 7 * (np.pi ** 2) * (1 / a ** 4) / 40
    # electron-positron contribution
    e_multiplier = 2 * (temp ** 4) / (np.pi ** 2)
    x = electron_mass / temp
    num_points = 100
    sample_points, weights = lg.laggauss(num_points)
    e_density_integral = np.sum(weights * np.exp(sample_points) * (sample_points ** 2) *\
                        np.sqrt(sample_points ** 2 + x ** 2) / \
                            (np.exp(np.sqrt(sample_points ** 2 + x ** 2)) + 1))
    e_density = e_multiplier * e_density_integral
    density = photon_energy_density + e_density + neutrino_energy_density + sterile_energy_density
    der[-2] = np.sqrt(3 * (planck_mass ** 2) / (8 * np.pi)) / (a * np.sqrt(density))

    # dT/da calculation (assuming dQ/da = 0)
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

    term_1 = 4 * (np.pi ** 2) * (temp ** 3) / 45
    term_2 = (e_density + e_pressure) / temp
    term_3 = (3 / temp) * term_1
    term_4 = -(1 / temp ** 2) * (e_density + e_pressure)
    term_5 = (1 / temp) * (e_dens_temp + e_pres_temp)
    
    der[-1] = (-3 / a) * (term_1 + term_2) / (term_3 + term_4 + term_5)

    # dns/da calculations
    r1, r2, r3, r4 = decays.compute_decay_rates(sterile_mass=sterile_mass, mixing_angle=mixing_angle)
    der[-3] = -(r1 + r2 + r3 + r4) * ns * der[-2]
    # df/da calculations
    e, ae, m, am, t, at = decays.compute_full_term(energies_cm=p[:-2],sterile_mass=sterile_mass, temp_cm= 1 / a, mixing_angle=mixing_angle)
    # electron

    # anti-electron

    # muon

    # anti-muon

    # tau
    dfda_t = t * ns * der[-2]
    #print(dfda_t)
    der[4 * length: 5 * length] = dfda_t
    #anti-tau
    dfda_at = at * ns * der[-2]
    der[5 * length:-3] = dfda_at
    return der



    
