'''
This module contains a way to create an object of the class 'CPW.create', which allows the user to create a CPW object which can for example be subsequently plotted. Additionally, basic parameters can be estimated using the theory from Mattis-Bardeen and Holloway. Material properties can be added to the materials in the CPW groundplane and central line by providing 'CPW.material' objects.

### Example b-Ta/NbTiN hybrid CPW ###
NbTiN_properties = dict(Tc  = 15.0,   # K
                        rho = 2e-6,   # Ohm m
                        d   = 150e-9, # m
                        )
NbTiN = sd.CPW.material("NbTiN", **NbTiN_properties)

bTa_properties   = dict(Tc  = 0.6,    # K
                        rho = 210e-8, # Ohm m
                        d   = 40e-9,  # m
                        )
bTa = sd.CPW.material("\u03B2-Ta", **bTa_properties)

hybrid_CPW = sd.CPW.create(S=2e-6, W=2e-6, material=NbTiN)
hybrid_CPW.make_hybrid(bTa)
'''

import numpy as np
import plotly.graph_objects as go
from scipy.special import i0, kn, ellipk, expit
from scipy.integrate import quad
from scipy.optimize import root_scalar
import warnings

h_bar = 6.582119569e-16 # eV*s
kB    = 8.617333262e-5  # eV/K
eps_0 = 8.854187817e-12 # F/m
mu_0  = 4e-7*np.pi      # H/m
eV    = 1.60217653e-19  # J
c     = 2.99792458e8    # m/s

class material():
    def __init__(self, name, Tc, rho, d):
        '''
        name: string 
            Material name
        Tc: float or int
            Critical temperature in K
        rho: float or int
            Resistivity in Ohm m
        d: float or int
            Thickness superconductor in m
        '''
        self.name = name
        self.Tc  = Tc  # K
        self.rho = rho # Ohm m
        self.d   = d   # m
        
        self.Delta0 = np.pi*np.exp(-np.euler_gamma) * kB*self.Tc # eV, Delta(T)=self.Delta valid for approx. T<Tc/5
    
    def __str__(self):
        return self.name
    
    @property
    def fc(self):
        '''
        Derives the frequency corresponding to the critical temperature of the material
            using hf = 2*Delta0
        
        returns:
            frequency for photon energy equivalent to energy phonon of Tc
        '''
        return self.Delta0 / h_bar / np.pi
    
    @property
    def R_s_approx(self):
        '''
        Approximates the normal state sheet resistance of the material (valid for kT >> 2*Delta)
        '''
        return self.rho / self.d
    
    @property
    def Lk_s_approx(self):
        '''
        Approximates the sheet kinetic inductance of the material (valid for kT, hf << 2*Delta)
        '''
        return h_bar * self.R_s_approx / np.pi / self.Delta0
    
    '''
    METHODS
        f: float or int, or numpy array
            Frequency value(s), in Hz.
        T: float or int
            Temperature value, in K.
        mode: str, kwarg
            If "readout", uses approximation for sigma_1 and sigma_2 (valid for kT, hf << 2*Delta)
            If "normal", defines sigma as sigma_N (i.e., 1/rho)
    '''
        
    def sigma(self, f, T, mode=None, DeltaT=True):
        '''
        Derives the complex conductivity definition of the material using Mattis-Bardeen or normal metal.
        
        DeltaT: bool, kwarg
            If True, Delta is taken as a function of temperature, approximating Delta(T); 
            if False, Delta is taken as the constant Delta0. Only used in combination with mode is None

        returns:
            complex conductivity
        '''
        try:
            sigma_N = 1/self.rho
        except ZeroDivisionError:
            raise Exception("For rho=0, use class CPW.PEC() instead.")
        
        if   mode=='readout':
            sigma_1, sigma_2 = self.sigma12_approx(f,T)
        elif mode=='normal':
            sigma_1, sigma_2 = np.ones(np.shape(f)), np.zeros(np.shape(f))
        else:
            sigma_1, sigma_2 = self.sigma12(f,T,DeltaT)
        
        return sigma_N * (sigma_1 - 1j*sigma_2)
    
    def sigma12(self, f, T, DeltaT=True):
        '''
        Derives the real and complex conductivity factors sigma1/sigmaN and sigma2/sigmaN of
        the material using Mattis-Bardeen.

        Uses scipy.special.quad(integration_func(), lower_limit, upper_limit) for integral approximation.
        Uses scipy.special.expit() for fermi distribution definition

        DeltaT: bool, kwarg
            If True, Delta is taken as a function of temperature;
            if False, Delta is taken as the constant Delta0

        returns:
            sigma_1: normalized real part conductivity
            sigma_2: normalized superconducting (imaginary part) conductivity
        '''
        assert isinstance(T, (int, float)), "Only single temperatures are valid."
        if DeltaT:
            Delta = self.Delta(T) # approximation of Delta(T)
        else:
            Delta = self.Delta0
        
        hbar_omega = np.array(h_bar * 2*np.pi*f, ndmin=1)
        if Delta == 0: # T>Tc
            return np.ones(np.shape(f)), np.zeros(np.shape(f))

        kBT = kB*T
        def int_1a(E, hf):
            g = (E*(E+hf) + Delta**2) / np.sqrt((E**2-Delta**2) * ((E+hf)**2-Delta**2))
            return (2*expit(-E/kBT) - 2*expit(-(E+hf)/kBT)) / hf * g
        def int_1b(E, hf):
            g = (E*(E+hf) + Delta**2) / np.sqrt((E**2-Delta**2) * ((E+hf)**2-Delta**2)) # negative for all hf>2*Delta
            return (             1  - 2*expit(-(E+hf)/kBT)) / hf *-g
        def int_2(E, hf):
            g = (E*(E+hf) + Delta**2) / np.sqrt((Delta**2-E**2) * ((E+hf)**2-Delta**2)) # take 1/i = -i out
            return (             1  - 2*expit(-(E+hf)/kBT)) / hf * g

        sigma_1, sigma_2 = np.zeros(len(hbar_omega)), np.zeros(len(hbar_omega))
        intlim1 = max(10*Delta, Delta+10*kBT)
        intkwargs = dict(epsabs=1e-12, epsrel=1e-9, limit=100)

        for i, hf in enumerate(hbar_omega):
            sigma_1[i] =      quad(int_1a, Delta,   intlim1, args=(hf), points=Delta, **intkwargs)[0]

            if hf >= 2*Delta:
                sigma_1[i] += quad(int_1b, Delta-hf, -Delta, args=(hf), points=[Delta-hf, -Delta], **intkwargs)[0]
                sigma_2[i]  = quad(int_2, -Delta,     Delta, args=(hf), points=[Delta-hf, -Delta], **intkwargs)[0]
            else:
                sigma_2[i]  = quad(int_2,  Delta-hf,  Delta, args=(hf), points=[-Delta, Delta-hf, Delta], **intkwargs)[0]

        return np.reshape(sigma_1, np.shape(f)), np.reshape(sigma_2, np.shape(f))
    
    def sigma12_approx(self, f, T, ignoreException=False):
        '''
        Derives the approximated real and complex conductivity factors sigma1/sigmaN and sigma2/sigmaN of
        the material using Mattis-Bardeen, given that kT, hf << 2*Delta.
        
        Uses scipy.special.i0(x) which is the Bessel function of the first kind of order 0
         and scipy.special.kn(0, x) which is the Bessel function of the second kind of order 0
        
        ignoreException: bool, kwarg
            If True, ignores exception for kT and message for hf << 2*Delta

        returns:
            sigma_1: normalized real part conductivity
            sigma_2: normalized superconducting (imaginary part) conductivity
        '''
        hbar_omega = np.array(h_bar*2*np.pi*f, ndmin=1)
        sigma_1, sigma_2 = np.empty(len(hbar_omega)), np.empty(len(hbar_omega))
        sigma_1.fill(np.nan), sigma_2.fill(np.nan)
        
        valid = hbar_omega < 2*self.Delta0
        if not ignoreException:
            check_low_energy(self, f, T)
                
        ratio_E = hbar_omega[valid]/kB/T
        sigma_1[valid] =     4*self.Delta0/hbar_omega[valid] * np.exp(-self.Delta0/kB/T) * np.sinh(ratio_E/2) * kn(0, ratio_E/2)
        sigma_2[valid] = np.pi*self.Delta0/hbar_omega[valid] * (1 - 2 * np.exp(-self.Delta0/kB/T - ratio_E/2) * i0(ratio_E/2))
        return np.reshape(sigma_1, np.shape(f)), np.reshape(sigma_2, np.shape(f))
    
    def Delta(self, T):
        '''
        Approximates Delta as a function of temperature for T<Tc, in eV. Takes DeltaT(T>Tc) = 0.
        Delta(T) is found by solving the BCS self-consistency equation. Uses that Debye energy (kTD) >> gap energy (Delta).
        
        Uses scipy.special.quad(integration_func(), lower_limit, upper_limit) for integral approximation.
        Uses scipy.optimize.root(root_func(), initial_guess) for root finding.
            returns same shape as T
        '''
        tau = np.array(T / self.Tc, ndmin=1)
        Delta = np.zeros(len(tau))
        
        integrand = lambda E, Del, t: expit(-np.pi*np.exp(-np.euler_gamma)/t * np.sqrt(E**2+Del**2)) / np.sqrt(E**2+Del**2)
        integral  = lambda    Del, t: quad(integrand, 0, np.inf, args=(Del, t))[0]
        
        for i, t in enumerate(tau):
            if t<1:
                Delta[i] = root_scalar(lambda Del: integral(Del, t) + np.log(Del)/2, x0=1/2, x1=1, method="secant").root
        return np.reshape(self.Delta0 * Delta, np.shape(T))
    
    def Delta_approx(self, T):
        '''
        Approximation of Delta as a function of temperature for T<<Tc, in eV.
            returns same shape as T
        '''
        return self.Delta0 - np.sqrt(2*np.pi*self.Delta0*kB*T) * np.exp(-self.Delta0/(kB*T))
    
    def Z_s(self, f, T, mode=None):
        '''
        Complex surface impedance of the material in Ohms/sq
        '''
        sigma = self.sigma(f, T, mode)
        return np.sqrt( 1j*mu_0 * 2*np.pi*f / sigma ) / np.tanh(self.d * np.sqrt(1j*mu_0 * sigma * 2*np.pi*f))
    
    def R_s(self, f, T, mode=None):
        '''
        Sheet resistance of the material in Ohms/sq
        '''
        Z_s = self.Z_s(f, T, mode)
        return np.real(Z_s)
    
    def X_s(self, f, T, mode=None):
        '''
        Sheet inductive reactance of the material in Ohms/sq
        '''
        Z_s = self.Z_s(f, T, mode)
        return np.imag(Z_s)
    
    def Lk_s(self, f, T, mode=None):
        '''
        Sheet kinetic inductance of the material in H/sq
        '''
        return self.X_s(f, T, mode) / (2*np.pi*f)
    
    def n_qp(self, T):
        '''
        Quasiparticle density expressed in units of N0 ([eV^-1 m^-3]), in eV.
            returns same shape as T
        '''
        T_ = np.array(T, ndmin=1)
        Delta = self.Delta(T_)
        n_qp = np.zeros(np.shape(T_))
        
        integrand = lambda E, D, t: E / np.sqrt(E**2 - D**2) * expit(-E/kB/t)
        for i, t in enumerate(T_):
            n_qp[i] = quad(integrand, Delta[i], max(10*Delta[i],Delta[i]+10*kB*t), args=(Delta[i], t), points=Delta[i], epsabs=1e-20, epsrel=1e-12)[0]
        return np.reshape(4*n_qp, np.shape(T))
    
    def n_qp_approx(self, T):
        '''
        Used to approximate the quasiparticle density expressed in units of N0 ([eV^-1 m^-3])
        for low temperature limit (kT << 2*Delta), in eV.
            returns same shape as T
        '''
        Delta = self.Delta_approx(T)
        return np.sqrt(8*np.pi*kB*T*Delta) * np.exp(-Delta / (kB*T))
    
    def dsigmadnqp(self, f, T):
        '''
        Derives the derivatives of the real and complex conductivity factors 
        sigma1/sigmaN and sigma2/sigmaN with respect to quasiparticle density
            dsigma/dnqp expressed in units of sigmaN/N0

        returns:
            dsigma_1/dnqp: derivative of normalized real part conductivity
            dsigma_2/dnqp: derivative of normalized superconducting (imaginary part) conductivity
        '''
        if isinstance(T, (int, float)):
            sigma1, sigma2 = self.sigma12(f,T)
        else:
            sigma1, sigma2 = [i for i in zip(*[self.sigma12(f,t) for t in T])]
        nqp = self.n_qp(T)
        return np.gradient(sigma1, nqp), np.gradient(sigma2, nqp)
    
    def dsigmadnqp_approx(self, f, T, ignoreException=False):
        '''
        Derives the approximated derivatives of the real and complex conductivity factors 
        sigma1/sigmaN and sigma2/sigmaN with respect to quasiparticle density, given that kT, hf << 2*Delta.
            dsigma/dnqp expressed in units of sigmaN/N0
        
        Uses scipy.special.i0(x) which is the Bessel function of the first kind of order 0
         and scipy.special.kn(0, x) which is the Bessel function of the second kind of order 0
        
        ignoreException: bool, kwarg
            If True, ignores exception for kT and message for hf << 2*Delta

        returns:
            dsigma_1/dnqp: derivative of normalized real part conductivity
            dsigma_2/dnqp: derivative of normalized superconducting (imaginary part) conductivity
        '''
        hbar_omega = np.array(h_bar*2*np.pi*f, ndmin=1)
        dsigma_1, dsigma_2 = np.empty(len(hbar_omega)), np.empty(len(hbar_omega))
        dsigma_1.fill(np.nan), dsigma_2.fill(np.nan)
        
        valid = hbar_omega < 2*self.Delta0
        if not ignoreException:
            check_low_energy(self, f, T)
                
        ratio_E = hbar_omega[valid]/kB/T
        dsigma_1[valid] =  np.sqrt(2*self.Delta0/(np.pi*kB*T))/hbar_omega[valid] * np.sinh(ratio_E/2) * kn(0, ratio_E/2)
        dsigma_2[valid] = -np.pi/2/hbar_omega[valid] * ( 1+np.sqrt(2*self.Delta0/(np.pi*kB*T))*np.exp(-ratio_E/2) * i0(ratio_E/2) )
        return np.reshape(dsigma_1, np.shape(f)), np.reshape(dsigma_2, np.shape(f))
    

    
class PEC():
    def __init__(self, d):
        '''
        Simplified version of class CPW.material() made for perfect electric conductor (PEC)
        
        name: string 
            Material name
        Tc: float or int
            Critical temperature in K
        d: float or int
            Thickness superconductor in m
        '''
        self.name = "PEC"
        self.Tc  = 300 # K
        self.rho = 0   # Ohm m
        self.d   = d   # m
        self.Delta0 = np.pi*np.exp(-np.euler_gamma) * kB*self.Tc
    
    def __str__(self):
        return self.name
    
    def sigma(self, f, T, mode=None, Delta_approx=None):
        return np.inf*np.ones(np.shape(f))
    
    def Z_s(self, f, T, mode=None):
        return np.zeros(np.shape(f))
    
    def R_s(self, f, T, mode=None):
        return np.zeros(np.shape(f))
    
    def X_s(self, f, T, mode=None):
        return np.zeros(np.shape(f))
    
    def Lk_s(self, f, T, mode=None):
        return np.zeros(np.shape(f))
        
        

class create():
    def __init__(self, S, W, material, eps_dielectric = 11.44):
        '''
        S: float or int
            CPW central strip width in m
        W: float or int
            CPW gap width in m
        material: object of class material
            Material of the CPW groundplane
        eps_dielectric: float or int, kwarg
            Relative permittivity of the dielectric (substrate). Defaults to Silicon
        '''
        self.S = S
        self.W = W
        self.W_tot = self.S + 2*self.W
        
        self.eps_r = eps_dielectric                           # relative permittivity of the dielectric
        self.eps_eff = (1+self.eps_r)/2                       # geometric effective relative permittivity 
                                                              # (thick substrate limit), eps_air=1        
        self.k   = self.S/(self.W_tot)
        self.k2  = np.sqrt(1-self.k**2)
        self.Kk  = ellipk(self.k**2)
        self.Kk2 = ellipk(self.k2**2)                              
        
        self.mat_c = material
        self.d_c = material.d
        self.mat_g = material
        self.d_g = material.d
        self.geometric_contributions()
        
        self.hybrid = False
        self.name = "{0} {1:g}-{2:g}-{1:g}".format(self.mat_g.name, self.W*1e6, self.S*1e6)
    
    def __str__(self):
        return self.name
    
    @property
    def Lg_l(self): # geometric inductance per unit length
        return mu_0/4 * self.Kk2 / self.Kk
    
    @property
    def Cg_l(self): # geometric capacity per unit length
        return 4*eps_0*self.eps_eff * self.Kk / self.Kk2 
    
    @property
    def C_l(self):  # total capacity per unit length
        return self.Cg_l
    
    def geometric_contributions(self):
        '''
        Derives the Holloway geometric contributions to the kinetic inductance
        
        g_c: central line geometric contribution
        g_g: groundplane geometric contribution
        '''
        if not ((self.mat_c.d < self.S/20) and 
                (self.mat_g.d < self.S/20) and
                (self.W > 0.3*self.S)):
            warnings.warn("Approximation geometric contributions is not valid using these values for S, W, and d.")
        
        factor = 1/(4 * self.S * self.k2**2 * self.Kk**2)
        self.g_c = (np.pi+np.log(4*np.pi* self.S     / self.d_c) - np.log((self.S+self.W) / self.W)*self.k) * factor
        self.g_g = (np.pi+np.log(4*np.pi* self.W_tot / self.d_g) - np.log((self.S+self.W) / self.W)/self.k) * factor
    
    def make_hybrid(self, material):
        '''
        Update central line material and thickness
        
        material: object of class material
            Material of the CPW central line
        '''
        self.mat_c = material
        self.d_c = material.d
        self.geometric_contributions()
        self.hybrid = True
        self.name = "{0}/{1} hybrid {2:g}-{3:g}-{2:g}".format(self.mat_c.name, self.mat_g.name, self.W*1e6, self.S*1e6)
        
    def _eps_eff(self, t):
        '''
        Effective relative permittivity approximation Hilberg (1969). 
        For thick substrates, tanh(...)=1 and kW/t<<1, and thus converges to eps_eff
        We assume a thick substrate for d>>W (already holds for d>3W) and tanh(1.785*log(d/W)+1.75) = 1
        ''' 
        return self.eps_eff*(np.tanh( 1.785*np.log10(d/self.W) + 1.75 ) + 
                             self.k*self.W/d * (0.04 - 0.7*self.k + 0.01*(1-0.1*self.eps_r)*(0.25+self.k)))
    
    '''
    METHODS
        f: float or int, or numpy array
            Frequency value(s), in Hz.
        T: float or int
            Temperature value, in K.
        mode: str, kwarg
            If "readout", uses approximation for sigma_1 and sigma_2 (valid for kT, hf << 2*Delta)
            If "normal", defines sigma as sigma_N (i.e., 1/rho)
    '''
    
    def R_l(self, f, T, mode=None):
        '''
        R_l, the total resistance per unit length in Ohm/m
        '''
        return self.mat_c.R_s(f,T,mode) * self.g_c + self.mat_g.R_s(f,T,mode) * self.g_g
    
    def Lk_l(self, f, T, mode=None):
        '''
        Lk_l, the kinetic inductance per unit length in H/m
        '''
        return self.mat_c.Lk_s(f,T,mode) * self.g_c + self.mat_g.Lk_s(f,T,mode) * self.g_g
    
    def L_l(self, f, T, mode=None):
        '''
        L_l, the total inductance per unit length in H/m
        '''
        return self.Lk_l(f,T,mode) + self.Lg_l
    
    def alpha_k(self, f, T, mode=None):
        '''
        alpha_k, the kinetic inductance fraction
        '''
        Lk_l = self.Lk_l(f,T,mode)
        return Lk_l / (Lk_l + self.Lg_l)
    
    def radiative_angle(self, f, T, mode=None):
        '''
        Radiative angle in radians. If undefined, no radiative shockwave is formed.
        '''
        return np.arccos(np.sqrt( self.eps_eff / self.eps_r / (1-self.alpha_k(f,T,mode)) ))
    
    def Z0(self, f, T, mode=None):
        '''
        Characteristic impedance, taking G_l = 0
        
        Returns:
            Z0, the CPW characteristic impedance in Ohm, taking G_l = 0
        '''
        return np.sqrt( (self.R_l(f,T,mode) + 2j*np.pi*f*self.L_l(f,T,mode)) / (2j*np.pi*f*self.C_l) )
    
    def gamma(self, f, T, mode=None):
        '''
        Propagation constant gamma = alpha + 1j*beta, taking G_l = 0
        
        Returns:
            gamma, the (complex-valued) propagation constant
        '''
        return np.sqrt( (self.R_l(f,T,mode) + 2j*np.pi*f*self.L_l(f,T,mode)) * (2j*np.pi*f*self.C_l) )
    
    def Z_CPW(self, f, T, mode=None):
        '''
        Characteristic impedance, taking R_l, G_l = 0 (first-order approximate of Z0)
        
        Returns:
            Z_CPW, the CPW impedance in Ohm
        '''
        return np.sqrt(self.L_l(f,T,mode) / self.C_l)
    
    def alpha(self, f, T, mode=None):
        '''
        Attenuation constant in Np/m.
        '''
        return np.real(self.gamma(f,T,mode))
    
    def alpha_approx(self, f, T, mode=None, ignoreException=False):
        '''
        Attenuation constant in Np/m.
        First-order approximation, for R_l << omega*L_l (kT, hf < 2*Delta)
        '''
        if not ignoreException:
            check_low_energy(self.mat_c, f, T)
        return self.R_l(f,T,mode) / 2 / self.Z_CPW(f,T,mode)
    
    def alpha_frankel(self, f, T, mode=None):
        '''
        Attenuation constant in -dB/m, approximation Frankel (1991)
        Shown not to be valid for our superconducting regimes by Hähnle (2020)
        '''
        cos_psi = np.cos(self.radiative_angle(f,T,mode)) # np.sqrt( self.eps_eff / self.eps_r / (1-self.alpha_k(f,T,mode)) )
        return 2*(np.pi/2)**5 * ((1-cos_psi**2)**2 / cos_psi) * self.W_tot**2*self.eps_r**(3/2) / c**3/self.Kk/self.Kk2 * f**3
    
    def v_phase(self, f, T, mode=None):
        '''
        Phase velocity in m/s
        Derived from beta = omega/v_phase
        '''
        return 2*np.pi*f / np.imag(self.gamma(f,T,mode))
        
    def v_phase_approx(self, f, T, mode=None, ignoreException=False):
        '''
        Phase velocity in m/s
        First-order approximation, for R_l << omega*L_l (kT, hf < 2*Delta)
        '''
        if not ignoreException:
            check_low_energy(self.mat_c, f, T)
        return 1/np.sqrt(self.L_l(f,T,mode) * self.C_l)
    
    def attenuation(self, f, T, mode=None):
        '''
        Attenuation constant in -dB/m
        '''
        return -self.alpha(f,T,mode) * 20*np.log10(np.exp(1))
    
    def ABCD(self, f, T, l, mode=None):
        '''
        ABCD (transmission) matrix for the CPW as a two-port network
        
        l: number
            Length of the CPW, in m
        '''
        gamma = self.gamma(f,T,mode)
        Z0 = self.Z0(f,T,mode)
        return np.array([ [np.cosh(gamma*l), Z0*np.sinh(gamma*l)] , [np.sinh(gamma*l)/Z0, np.cosh(gamma*l)] ])
        
    def draw(self, savename=None, size=300, color_Si="#c0c0c0", color_g="#181932", color_c="#D53F15", template='plotly+base+light'):
        '''
        Create a schematic representation of the CPW.

        savename: string, kwarg
            file name for saving
        size: int, kwarg
            Width and height of plot
        color_Si: str, kwarg
            String containing color code for the dielectric
        color_c: str, kwarg
            String containing color code for the central line
        color_g: str, kwarg
            String containing color code for the groundplane
        template: str, kwarg
            string of go.layout.Template

        returns:
            go.Figure()
        '''
        dl, px = self.d_g/3, 5
        offset  = dict(right=-px,center=0,left=px,top=-px,middle=0,bottom=px) # distances from arrows to annotations
            
        fig = go.Figure()
        
        # Setting arrows and annotations:
        arrows = np.array([[[           -self.S/2, -self.d_g/3], [            self.S/2, -self.d_g/3]], # S
                           [[            self.S/2, -self.d_g/3], [   self.W  +self.S/2, -self.d_g/3]], # W
                           [[  -self.W  -self.S/2, -self.d_g/3], [           -self.S/2, -self.d_g/3]], # W
                           [[       -self.W_tot/2,7*self.d_g/6], [        self.W_tot/2,7*self.d_g/6]], # W_tot
                           [[-5*self.W/6-self.S/2,         0  ], [-5*self.W/6-self.S/2,  self.d_g  ]]])# d_g
        labels = ["S", "W", "W", "W_{tot}", "d_{%s}"%self.mat_g.name] # labels for annotations
        anchors = [["center", "top"   ], # S
                   ["center", "top"   ], # W
                   ["center", "top"   ], # W
                   ["center", "bottom"], # W_tot
                   ["left"  , "middle"]] # d_g
        
        if not self.hybrid:
            color_c = color_g
        else:
            arrows = np.append(arrows, [[[self.W/6+self.S/2, 0], [self.W/6+self.S/2, self.d_c]]], axis=0) # d_c
            labels.append("d_{%s}"%self.mat_c.name)
            anchors.append(["left", "middle"])
            fig.add_annotation(x=0, y=self.d_c, yshift=offset['bottom'], xanchor='center', yanchor='bottom',
                               text=self.mat_c.name, showarrow=False, font_size=13, font_color=color_c)
        
        annots  = np.mean(arrows, axis=1) # midpoints arrows for annotations
        for arrow, label, annot, anchor in zip(arrows, labels, annots, anchors): # Placing arrows and annotations
            marks = [arrow, arrow[::-1]] # arrows in both directions
            len_arrow = arrow[1]-arrow[0]
            if np.sum(len_arrow) < dl:   # displace arrows when space is too narrow
                marks += np.array([[-len_arrow], [len_arrow]])
            fig.add_annotation(x=annot[0], xanchor=anchor[0], xshift=offset[anchor[0]],
                               y=annot[1], yanchor=anchor[1], yshift=offset[anchor[1]],
                               text="$\large{\mathrm{"+label+"}}$", showarrow=False, font_color='rgba(0,0,0,0.3)')
            for mark in marks: # placing both arrows
                fig.add_trace(go.Scatter(x=mark[:,0], y=mark[:,1],
                                         marker=dict(symbol="arrow-up", angleref='previous', color='#929292', size=10)))
        
        # Central line
        fig.add_shape(type="rect", x0=-self.S/2, y0=0, x1=self.S/2, y1=self.d_c,
                      line_color="rgba(0,0,0,0)", fillcolor=color_c, layer='below')
        
        # Groundplanes
        for x0,x1 in [(-self.W_tot,-self.W-self.S/2), (self.W+self.S/2,self.W_tot)]:
            fig.add_shape(type="rect", x0=x0, y0=0, x1=x1, y1=self.d_g,
                          line_color="rgba(0,0,0,0)", fillcolor=color_g, layer='below')
        fig.add_annotation(x=3*self.W_tot/4, y=self.d_g/2, xanchor='center', yanchor='middle',
                           text=self.mat_g.name, showarrow=False, font_size=13, font_color='#929292')
        
        # Silicon substrate
        fig.add_shape(type="rect", x0=-self.W_tot, y0=-self.d_g, x1=self.W_tot, y1=0,
                      line_color="rgba(0,0,0,0)", fillcolor=color_Si, layer='below')
        fig.add_annotation(x=3*self.W_tot/4, y=-3*self.d_g/8, xanchor='center', yanchor='middle',
                           text="Si", showarrow=False, font_size=13)
        
        # Air
        fig.add_annotation(x=3*self.W_tot/4, y=11*self.d_g/8, xanchor='center', yanchor='middle',
                           text="Air", showarrow=False, font_size=13)
        
        # Layout
        fig.update_layout(width=2*size, height=size, template=template, showlegend=False,
                          title=dict(text=self.name+" CPW", yanchor="bottom", y=0.85))
        xticks = np.array([-self.W-self.S/2, -self.S/2, self.S/2, self.W+self.S/2])
        zticks = np.array([0, self.d_g, self.d_c])
        fig.update_xaxes(title="$\large{x\:\mathrm{[m]}}$", range=[-self.W_tot, self.W_tot],
                         tickmode='array', tickvals=xticks, #ticktext=xticks,
                         griddash='dash', gridcolor='rgba(0,0,0,0.3)', zeroline=False)
        fig.update_yaxes(title="$\large{z\:\mathrm{[m]}}$", range=[-3*self.d_g/4, 7*self.d_g/4],
                         tickmode='array', tickvals=zticks, #ticktext=zticks,
                         griddash='dash', gridcolor='rgba(0,0,0,0.3)', zeroline=False)
        fig.show()

        if savename is not None:
            fig.write_image(savename, width=2*size, height=size)



def check_low_energy(mat, f, T):
    '''
    Checks whether low energy approximation holds (kT, hf < 2*Delta)
    '''
    if np.all(kB*T > 2*mat.Delta0):
        raise Exception(f"Approximation is not valid for {mat.name} using T={T}K.")
    if not np.all(h_bar*2*np.pi*f < 2*mat.Delta0):
        try:
            warnings.warn(f"Approximation is not valid for {mat.name} in (part of) this frequency region (for f\u2A86{f[h_bar*2*np.pi*f > 2*mat.Delta0][0]:.3g}).")
        except:
            warnings.warn(f"Approximation is not valid for {mat.name} on this frequency (f={f:.3g}).")
