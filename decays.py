import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import numpy.polynomial.legendre as lg

num_points = 100
sample_points, weights = lg.leggauss(num_points)

# energy binning for testing
num_bins = 100

# particle masses and constants in MeV
fermi = 1.16637e-11
fine_structure = 7.29735e-3
pion_decay = 131

electron_mass = 0.511
muon_mass = 105.658
neutral_pion_mass = 134.9768
charged_pion_mass = 139.57039
        

@nb.jit(nopython=True)
def compute_decay_rates(sterile_mass:float, mixing_angle:float):
    """
    Computes the decays rates of each decay channel.

    Parameters
    ----------
    sterile_mass : float
        Mass of sterile neutrino used.
    mixing_angle : float
        Mixing angle between the sterile and electron neutrinos.

    Returns
    -------
    rate_1 : float
        The decay rate corresponding to decay process 1.
    rate_2 : float
        The decay rate corresponding to decay process 2.
    rate_3 : float
        The decay rate corresponding to decay process 3.
    rate_4 : float
        The decay rate corresponding to decay process 4.

    """
    rate_1 = 3 * fine_structure * (fermi ** 2) * (sterile_mass ** 5) * \
            (np.sin(mixing_angle) ** 2) / (512 * np.pi ** 4)
    rate_2, rate_3, rate_4 = 0, 0, 0
    if sterile_mass >= neutral_pion_mass:
        rate_2 = (fermi ** 2) * (pion_decay ** 2) * sterile_mass * (sterile_mass ** 2 - neutral_pion_mass ** 2) * \
                (np.sin(mixing_angle) ** 2) / (48 * np.pi)
    if sterile_mass >= charged_pion_mass + electron_mass:
        rate_3 = (fermi ** 2) * (pion_decay ** 2) * sterile_mass * \
                np.sqrt((sterile_mass ** 2 - (charged_pion_mass + electron_mass) ** 2) *\
                        (sterile_mass ** 2 - (charged_pion_mass - electron_mass) ** 2)) *\
                        (np.sin(mixing_angle) ** 2) / (16 * np.pi)
    if sterile_mass >= charged_pion_mass + muon_mass:
        rate_4 = (fermi ** 2) * (pion_decay ** 2) * sterile_mass * \
                np.sqrt((sterile_mass ** 2 - (charged_pion_mass + muon_mass) ** 2) *\
                        (sterile_mass ** 2 - (charged_pion_mass - muon_mass) ** 2)) *\
                        (np.sin(mixing_angle) ** 2) / (16 * np.pi)
    return rate_1, rate_2, rate_3, rate_4


@nb.jit(nopython=True)
def compute_lifetime(sterile_mass:float, mixing_angle):
    """
    Computes the lifetime of the sterile neutrino.

    Parameters
    ----------
    sterile_mass : float
        Mass of sterile neutrino used.
    mixing_angle : float
        Mixing angle between the sterile and electron neutrinos.

    Returns
    -------
    lifetime : float
        The lifetime of the sterile neutrino.

    """
    r1, r2, r3, r4 = compute_decay_rates(sterile_mass=sterile_mass, mixing_angle=mixing_angle)
    total_rate = 3 * r1 + 3 * r2 + 2 * r3 + 2 * r4
    lifetime = 1 / total_rate
    return lifetime


@nb.jit(nopython=True)
def get_decay_monoenergy(m0:float, m1:float=0, m2:float=0):
    """
    Computes the energy of a particle as emerging from a two-body decay in 
    the frame of the decaying particle.

    Parameters
    ----------
    m0 : float
        The mass of the decaying particle.
    m1 : float, optional
        Mass of the particle in question. The default is 0.
    m2 : float, optional
        Mass of the other particle produce in decay. The default is 0.

    Returns
    -------
    particle_energy : float
        The energy of the outgoing particle in question.
    """
    if(m0 > 0):
        num = m0 ** 2 + m1 ** 2 - m2 ** 2
        particle_energy = num / (2 * m0)
        return particle_energy


@nb.jit(nopython=True)
def get_kinetic_terms(m0:float, m1:float=0, m2:float=0):
    """
    Computes the relativistic kinematic quantities of the particle in question 
    from the frame of the decaying particle.

    Parameters
    ----------
    m0 : float
        The mass of the decaying particle.
    m1 : float, optional
        Mass of the particle in question. The default is 0.
    m2 : float, optional
        Mass of the other particle produce in decay. The default is 0.

    Returns
    -------
    lorentz_factor : float
        The factor by which the particle's energy is greater than its mass.
    speed : float
        The unitless speed of the particle in question.
    momentum : float
        The momentum of the particle in question.
    """
    particle_energy = get_decay_monoenergy(m0,m1,m2)
    momentum = np.sqrt(particle_energy ** 2 - m1 ** 2)
    speed = momentum / particle_energy
    lorentz_factor = particle_energy / m1
    return lorentz_factor, speed, momentum


"""
**These functions have been retired

def diff_decay_rate_a(neutrino_energy:float):
    term_1 = neutrino_energy / (1 - (2 * neutrino_energy / muon_mass))
    term_2 = (1 - (2 * neutrino_energy / muon_mass) - (electron_mass / muon_mass) ** 2) ** 2
    return term_1 * term_2


def diff_decay_rate_b(neutrino_energy:float):
    return neutrino_energy * diff_decay_rate_a(neutrino_energy=neutrino_energy)


def integrate_func(integrand, start:float, end:float, width:float):
    integral = 0
    for i in range(num_bins):
        add_term = 0
        if (start + (i + 1) * width) <= end:
            add_term = 0.5 * width * (integrand(start + i * width) + integrand(start + (i + 1) * width))
        integral += add_term
    return integral

@nb.jit(nopython=True)
def integrate_array(values:list[float], start:float, end:float, bin_width:float):
    Integrates an array of values for the purpose of checking decay rates.
    
    Parameters
    ----------
    values : list[float]
        The array we wish to integrate.
    start : float
        The initial value of the variable you are integrating with respect to.
    end : float
        The final value of the variable you are integrating with respect to.
    bin_width : float
        The size of the integrating step.

    Returns
    -------
    integral : TYPE
        The integral of the array.
    
    integral = 0
    for i in range(num_bins):
        add_term = 0
        if i < num_bins - 1:
            add_term = 0.5 * bin_width * (values[i] + values[i + 1])
        integral += add_term
    return integral


@nb.jit(nopython=True)
def get_type_one(decay_rate:float, energy:float, width:float, sterile_mass:float, product_mass:float=0):
    neutrino_energy = get_decay_monoenergy(m0=sterile_mass, m2=product_mass)
    var = 5 * width
    ddecay_rate = decay_rate * np.sqrt( 1/ (2 * np.pi * var)) * np.exp(- ((energy - neutrino_energy) ** 2) / (2 * var))
    return ddecay_rate
"""


@nb.jit(nopython=True)
def get_gamma_a(evaluate_at:float):
    """
    Computes the analytically generated integral of the differential 
    decay rate of the muon with neutrino energy (in the muon frame), reduced by a factor 
    of the energy.

    Parameters
    ----------
    evaluate_at : float
        The neutrino energy (sterile frame) at which we would like to evluate 
        the integral.

    Returns
    -------
    float
        The indefinite integral at a particular neutrino energy 
        (constant is presumed zero out of irrelevance).
    """
    term_1 = 3 * (electron_mass ** 4) 
    term_2 = 6 * (electron_mass ** 2) * muon_mass * evaluate_at
    term_3 = (muon_mass ** 2) * evaluate_at * (4 * evaluate_at - 3 * muon_mass)
    term_4 = (electron_mass ** 4) * muon_mass * np.log(np.abs(2 * evaluate_at - muon_mass))
    numerator = -(evaluate_at / 6) * (term_1 + term_2 + term_3) - term_4 / 4
    return numerator / (muon_mass ** 3)


@nb.jit(nopython=True)
def get_gamma_b(evaluate_at:float):
    """
    Computes the analytically generated integral of the differential 
    decay rate of the muon with neutrino energy (in the muon frame).

    Parameters
    ----------
    evaluate_at : float
        The neutrino energy (sterile frame) at which we would like to evluate 
        the integral.

    Returns
    -------
    float
        The indefinite integral at a particular neutrino energy 
        (constant is presumed zero out of irrelevance).
    """
    term_1 = -16 * (electron_mass ** 2) * muon_mass * (evaluate_at ** 3)
    term_2 = -6 * (electron_mass ** 4) * evaluate_at * (muon_mass + evaluate_at)
    term_3 = -4 * (muon_mass ** 2) * (evaluate_at ** 3) * (3 * evaluate_at - 2 * muon_mass)
    term_4 = -3 * (electron_mass ** 4) * (muon_mass ** 2) * np.log(np.abs(2 * evaluate_at - muon_mass))
    return (term_1 + term_2 + term_3 + term_4) / (24 * muon_mass ** 3)


@nb.jit(nopython=True)
def get_decay_type_one(decay_rate:float, energy:float, width:float, sterile_mass:float, product_mass:float=0):
    """
    Computes the dP/dtdE contribution from type I decays.

    Parameters
    ----------
    decay_rate : float
        The rate of the decay of the sterile neutrino via the process that is
        generating the type I decay neutrino.
    energy : float
        The neutrino energy (sterile frame) at which we would like to 
        compute the contribution.
    width : float
        The enegry bin spacing.
    sterile_mass : float
        Mass of sterile neutrino used.
    product_mass : float, optional
        The mass of the particle accompanying the produced neutrino. 
        The default is 0.

    Returns
    -------
    ddecay_rate : float
        The dP/dtdE contribution form this neutrino production.

    """
    neutrino_energy = get_decay_monoenergy(m0=sterile_mass, m2=product_mass)
    ddecay_rate = 0
    if energy <= neutrino_energy <= energy + width:
        ddecay_rate = decay_rate / width
    return ddecay_rate


@nb.jit(nopython=True)
def get_decay_type_two(decay_rate:float, energy:float, sterile_mass:float, mass_spectator:float):
    """
    Computes the dP/dtdE contribution from type II decays.

    Parameters
    ----------
    decay_rate : float
        The rate of the decay of the sterile neutrino via the process that is
        generating the type I decay neutrino.
    energy : float
        The neutrino energy (sterile frame) at which we would like to 
        compute the contribution.
    sterile_mass : float
        Mass of sterile neutrino used.
    mass_spectator : float, optional
        The mass of the particle produced alongside the charged pion.

    Returns
    -------
    ddecay_rate : float
        The dP/dtdE contribution form this neutrino production.

    """
    ddecay_rate = 0
    gamma_pion, pion_speed, _ = get_kinetic_terms(m0=sterile_mass, m1=charged_pion_mass, m2=mass_spectator)
    neutrino_energy_pion = get_decay_monoenergy(m0=charged_pion_mass, m2=muon_mass)
    min_energy = gamma_pion * neutrino_energy_pion * (1 - pion_speed)
    max_energy = gamma_pion * neutrino_energy_pion * (1 + pion_speed)
    if min_energy <= energy <= max_energy:
        ddecay_rate = decay_rate / (2 * gamma_pion * pion_speed * neutrino_energy_pion)
    return ddecay_rate


@nb.jit(nopython=True)
def get_decay_type_three(decay_rate:float, energy:float, sterile_mass:float):
    """
    Computes the dP/dtdE contribution from type III decays.

    Parameters
    ----------
    decay_rate : float
        The rate of the decay of the sterile neutrino via the process that is
        generating the type I decay neutrino.
    energy : float
        The neutrino energy (sterile frame) at which we would like to 
        compute the contribution.
    sterile_mass : float
        Mass of sterile neutrino used.

    Returns
    -------
    ddecay_rate : float
        The dP/dtdE contribution form this neutrino production.

    """
    gamma_muon, muon_speed, _ = get_kinetic_terms(m0=sterile_mass, m1=muon_mass, m2=charged_pion_mass)
    neutrino_max_energy_muon = get_decay_monoenergy(m0=muon_mass, m2=electron_mass)
    a = energy / (gamma_muon * (1 + muon_speed))
    b = min(neutrino_max_energy_muon, energy / (gamma_muon * (1 - muon_speed)))
    if a < b:
        gamma_a = get_gamma_a(b) - get_gamma_a(a)
        gamma_b = get_gamma_b(neutrino_max_energy_muon) - get_gamma_b(0)
        return decay_rate * gamma_a / (2 * gamma_muon * muon_speed * gamma_b)
    else:
        return 0


@nb.jit(nopython=True)
def type_four_integrand(energy:float, muon_energy:float):
    """
    Computes the top-hat adjust distribution from decay type three to 
    incorporate an additional frame shift.

    Parameters
    ----------
    energy : float
        The neutrino energy (sterile frame) at which we would like to 
        compute the contribution.
    muon_energy : float
        The muon energy at which we would like to compute the contribution.

    Returns
    -------
    float
        The integrand for computation of type IV decays.

    """
    gamma_muon = muon_energy / muon_mass
    muon_speed = np.sqrt(muon_energy ** 2 - muon_mass ** 2) / muon_energy
    neutrino_max_energy_muon = get_decay_monoenergy(m0=muon_mass, m2=electron_mass)
    a = energy / (gamma_muon * (1 + muon_speed))
    b = min(energy / (gamma_muon * (1 - muon_speed)), neutrino_max_energy_muon)
    if b - a > 1e-4:
        gamma_a = get_gamma_a(b) - get_gamma_a(a)
        gamma_b = get_gamma_b(neutrino_max_energy_muon) - get_gamma_b(0)
        return (gamma_a / (2 * gamma_muon * muon_speed * gamma_b))
    return 0


@nb.jit(nopython=True)
def integrate_func_special(energy:float, start:float, end:float):
    """
    Works to integrate the type four integrand with its energy dependent terms.

    Parameters
    ----------
    energy : float
        The neutrino energy (sterile frame) at which we would like to 
        compute the contribution.
    start : float
        The energy at which we would like to start the integration.
    end : float
        The energy at which we would like to stop the integration.

    Returns
    -------
    integral : float
        The integrated term.

    """
    integral = 0
    for i,point in enumerate(sample_points):
        mu_energy = ((end - start) * point + (end + start))/2
        integral += weights[i] * type_four_integrand(energy, mu_energy) * (end - start) / 2
    return integral


@nb.jit(nopython=True)
def get_decay_type_four(decay_rate:float, energy:float, sterile_mass:float, pion_spectator_mass:float):
    """
    Computes the dP/dtdE contribution from type IV decays.

    Parameters
    ----------
    decay_rate : float
        The rate of the decay of the sterile neutrino via the process that is
        generating the type I decay neutrino.
    energy : float
        The neutrino energy (sterile frame) at which we would like to 
        compute the contribution.
    sterile_mass : float
        Mass of sterile neutrino used.
    pion_spectator_mass : float
        The mass of the particle produced alongside the charged pion.

    Returns
    -------
    float
        The dP/dtdE contribution form this neutrino production.

    """
    gamma_pion, pion_speed, _ = get_kinetic_terms(m0=sterile_mass, m1=charged_pion_mass, m2=pion_spectator_mass)
    gamma_muon, muon_speed, muon_momentum = get_kinetic_terms(m0=charged_pion_mass, m1=muon_mass)
    min_energy = gamma_pion * (gamma_muon * muon_mass - pion_speed * muon_momentum)
    max_energy = gamma_pion * (gamma_muon * muon_mass + pion_speed * muon_momentum)
    type_three_modifier = integrate_func_special(energy=energy, start=min_energy, end=max_energy)
    return decay_rate * type_three_modifier / (2 * gamma_pion * pion_speed * muon_momentum)


@nb.jit(nopython=True)
def compute_dPdtdE(energies_cm:np.ndarray, sterile_mass:float, temp_cm:float, mixing_angle:float):
    """
    Computes the arrays of dP/dtdE values for each neutrino species as created 
    from the sterile neutrino.

    Parameters
    ----------
    energies_cm : list[float]
        The unitless energy terms.
    sterile_mass : float
        Mass of the sterile neutrino.
    temp_cm : float
        The comoving tmeperature.
    mixing_angle : float
        Mixing angle between the sterile and electron neutrinos.

    Returns
    -------
    pe : np.ndarray
        The cummulative dP/dtdE contributions for electron neutrino production.
    pae : np.ndarray
        The cummulative dP/dtdE contributions for electron anti-neutrino production.
    pm : np.ndarray
        The cummulative dP/dtdE contributions for muon neutrino production.
    pam : np.ndarray
        The cummulative dP/dtdE contributions for muon anti-neutrino production.
    pt : np.ndarray
        The cummulative dP/dtdE contributions for tau neutrino production.
    pat : np.ndarray
        The cummulative dP/dtdE contributions for tau anti-neutrino production.

    """
    # decay rates
    rate_1, rate_2, rate_3, rate_4 = compute_decay_rates(sterile_mass=sterile_mass, mixing_angle=mixing_angle)
    energy_bins = energies_cm * temp_cm
    bin_width = (energy_bins[1] - energy_bins[0])
    pe = np.zeros_like(energy_bins)
    pae = np.zeros_like(energy_bins)
    pm = np.zeros_like(energy_bins)
    pam = np.zeros_like(energy_bins)
    pt = np.zeros_like(energy_bins)
    pat = np.zeros_like(energy_bins)
    d1s = np.zeros_like(energy_bins)
    d2s = np.zeros_like(energy_bins)
    d3_2s = np.zeros_like(energy_bins)
    d3_4s = np.zeros_like(energy_bins)
    d4_2s = np.zeros_like(energy_bins)
    d4_3s = np.zeros_like(energy_bins)
    d4_4s = np.zeros_like(energy_bins)
    d3s = np.zeros_like(energy_bins)
    d4s = np.zeros_like(energy_bins)
    #total = np.zeros_like(energy_bins)
    for i, energy in enumerate(energy_bins):
        # decay 1 contributions (all identical)
        d1 = get_decay_type_one(decay_rate=rate_1, energy=energy, sterile_mass=sterile_mass, width=bin_width)
        d2, d3_2, d3_4, d4_2, d4_3, d4_4 = 0, 0, 0, 0, 0, 0

        # decay 2 contributions (all identical)
        if sterile_mass >= neutral_pion_mass:
            d2 = get_decay_type_one(decay_rate=rate_2, energy=energy, width=bin_width, sterile_mass=sterile_mass, product_mass=neutral_pion_mass) 
        
        # decay 3 contributions
        if sterile_mass >= (electron_mass + charged_pion_mass):
            d3_2 = get_decay_type_two(decay_rate=rate_3, energy=energy, sterile_mass=sterile_mass, mass_spectator=electron_mass)
            d3_4 = get_decay_type_four(decay_rate=rate_3, energy=energy, sterile_mass=sterile_mass, pion_spectator_mass=electron_mass)
        
        # decay 4 contributions
        if sterile_mass >= (muon_mass + charged_pion_mass):
            d4_2 = get_decay_type_two(decay_rate=rate_4, energy=energy, sterile_mass=sterile_mass, mass_spectator=muon_mass)
            d4_3 = get_decay_type_three(decay_rate=rate_4, energy=energy, sterile_mass=sterile_mass)
            d4_4 = get_decay_type_four(decay_rate=rate_4, energy=energy, sterile_mass=sterile_mass, pion_spectator_mass=muon_mass)


        d1s[i] = d1
        d2s[i] = d2
        d3_2s[i] = d3_2
        d3_4s[i] = d3_4
        d4_2s[i] = d4_2
        d4_3s[i] = d4_3
        d4_4s[i] = d4_4

        # for decay rate checks
        d3s[i] = (d3_2 + 2 * d3_4) / 3
        d4s[i] = (2 * d4_3 + d4_2 + 2 * d4_4) / 5

        # electron neutrino contributions
        pe[i] = d1 + d2 + d3_4 + d4_3 + d4_4

        # anti-alectron neutrino contributions
        pae[i] = d3_4 + d4_3 + d4_4

        # muon-neutrino contributions
        pm[i] = d1 + d2 + d3_2 + d3_4 + d4_2 + d4_3 + d4_4

        # anti-muon neutrino contributions
        pam[i] = d3_2 + d3_4 + d4_2 + d4_3 + d4_4

        # tau-neutrino contributions
        pt[i] = d1 + d2

        # anti-tau neutrino contributions
        # none
    """
    This can be used to check individual plots.
    
    print(f'1:{integrate_array(d1s,0,sterile_mass / 2, bin_width) / rate_1}')
    if rate_2 != 0:
        print(integrate_array(d2s,0,sterile_mass / 2, bin_width) / rate_2)
        if rate_3 != 0:
            print(integrate_array(d3s,0,sterile_mass / 2, bin_width) / rate_3)
            if rate_4 != 0:
                print(integrate_array(d4s,0,sterile_mass / 2, bin_width) / rate_4)
    
    fig, axs = plt.subplots(2,3)
    axs[0,0].set_xlim(0, sterile_mass / 2 + 5)
    axs[0,1].set_xlim(0, sterile_mass / 2 + 5)
    axs[0,2].set_xlim(0, sterile_mass / 2 + 5)
    axs[1,0].set_xlim(0, sterile_mass / 2 + 5)
    axs[1,1].set_xlim(0, sterile_mass / 2 + 5)
    axs[1,2].set_xlim(0, sterile_mass / 2 + 5)
    axs[0,0].set_title('Electron Neutrinos')
    axs[0,0].semilogy(energy_bins, pe, label='electron neutrino')
    axs[1,0].set_title('Electron Anti-Neutrinos')
    axs[1,0].plot(energy_bins, pae, label='anti-electron neutrino')
    axs[0,1].set_title('Muon Neutrinos')
    axs[0,1].semilogy(energy_bins, pm, label='muon neutrino')
    axs[1,1].set_title('Muon Anti-Neutrinos')
    axs[1,1].semilogy(energy_bins, pam, label='anti-muon nuetrino')
    axs[0,2].set_title('Tau Neutrinos')
    axs[0,2].semilogy(energy_bins, pt, label='tau neutrino')
    axs[1,2].set_title('Tau Anti-Neutrinos')
    axs[1,2].plot(energy_bins, pat, label='anti-tau neutrino')
    plt.show()
    """
    return pe, pae, pm, pam, pt, pat


@nb.jit(nopython=True)
def compute_full_term(energies_cm:np.ndarray, sterile_mass:float, temp_cm:float, mixing_angle:float):
    """
    Computes the (2pi^2 / E^2) (dP/dtdE) terms.

    Parameters
    ----------
    energies_cm : np.ndarray
        The unitless energy terms.
    sterile_mass : np.ndarray
        Mass of the sterile neutrino.
    temp_cm : np.ndarray
        The comoving tmeperature.
    mixing_angle : np.ndarray
        Mixing angle between the sterile and electron neutrinos.

    Returns
    -------
    electron : np.ndarray
        The cummulative contributions for electron neutrino production.
    anti_electron : np.ndarray
        The cummulative contributions for electron anti-neutrino production.
    muon : np.ndarray
        The cummulative contributions for muon neutrino production.
    anti_muon : np.ndarray
        The cummulative contributions for muon anti-neutrino production.
    tau : np.ndarray
        The cummulative contributions for tau neutrino production.
    anti_tau : np.ndarray
        The cummulative contributions for tau anti-neutrino production.
    """
    pe, pae, pm, pam, pt, pat = compute_dPdtdE(energies_cm=energies_cm, sterile_mass=sterile_mass, temp_cm=temp_cm, mixing_angle=mixing_angle)
    electron = np.zeros_like(energies_cm)
    anti_electron = np.zeros_like(energies_cm)
    muon = np.zeros_like(energies_cm)
    anti_muon = np.zeros_like(energies_cm)
    tau = np.zeros_like(energies_cm)
    anti_tau = np.zeros_like(energies_cm)

    for i,energy in enumerate(energies_cm):
        if energy != 0:
            term = (2 * np.pi ** 2) / (energy * temp_cm) ** 2
            electron[i] = term * pe[i]
            anti_electron[i] = term * pae[i]
            muon[i] = term * pm[i]
            anti_muon[i] = term * pam[i]
            tau[i] = term * pt[i]
            # no anti-tau contributions
        else:
            electron[i] = 0
            anti_electron[i] = 0
            muon[i] = 0
            anti_muon[i] = 0
            tau[i] = 0

    return electron, anti_electron, muon, anti_muon, tau, anti_tau
    


"""
# sterile neutrino test
sterile_mass = 300
mixing_angle = 1.22e-5
energies_cm = np.linspace(0, (sterile_mass + 1) / (2 * 0.1) , num_bins)
print(compute_dPdtdE(energies_cm, sterile_mass, 0.1, mixing_angle))
#print(compute_dPdtdE(energies_cm, sterile_mass, 10, mixing_angle))
"""
