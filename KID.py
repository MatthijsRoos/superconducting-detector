'''
This module contains a way to create an object of the class 'KID.hybrid', which allows the user to create (quarterwave) hybrid KID objects to help in estimating CPW lengths and therefore KID lengths given design resonance frequencies and narrow CPW lengths. These estimations are based on narrow CPW inputs and/or SONNET simulation data of the narrow CPW (which can also include other (load) impedances added to it, like an antenna). Similarly, one can also make a 'KID.simple', consisting of a single CPW

### Example b-Ta/NbTiN hybrid KID ###
# Using the materials/CPWs used in the example in CPW.py

l_hybrid = 500e-6 # m
KID = sd.KID.hybrid(hybrid_CPW, NbTiN_CPW, l_hybrid)
'''
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from . import func, TL, plot, CPW
from copy import deepcopy
import warnings

c     = 2.99792458e8 # m/s

class _KID():
    def __init__(self, CPW, T, T0, mode):
        '''
        Base class for any quarterwave KID
        
        CPW: CPW.create object
            Instance of CPW.create, representing the base CPW the KID is made of
        T: number
            Effective temperature, in K
        T0: number
            Design temperature, in K
        mode: str, kwarg
            If "readout", uses approximation for sigma_1 and sigma_2 in superconducting limit (valid for kT, hf << 2*Delta)
            If "normal", defines sigma as sigma_N (i.e., 1/rho) (valid for kT, hf >> 2*Delta)
        '''
        self.CPW  = CPW
        self.T    = T  # K
        self.T0   = T0 # K
        self.mode = mode
        
        # Initialize Z_L, load impedance in Ohms:
        self.Z_L = lambda f: np.zeros(np.shape(f)) # shorted load
    
    def set_Z_L(self, f_L, Z_L):
        '''
        Z_L used to represent load impedance at the short end of the CPW.
        Can be used to represent input impedance at the short end using CST/SONNET data.
        
        f_L: array
            Frequencies Z_L is evaluated at, in Hz
        Z_L: array
            Impedance data at frequencies f_L, in Ohms
        '''        
        Z_L_re = lambda f: func.interp1D(f, f_L.flatten(), np.real(Z_L.flatten()))
        Z_L_im = lambda f: func.interp1D(f, f_L.flatten(), np.imag(Z_L.flatten()))
        self.Z_L = lambda f: Z_L_re(f) + 1j*Z_L_im(f)
        
    def l_KID_PEC(self, f0):
        '''
        Yield the PEC KID length given design resonance frequency f0.
        Found by taking v_phase=c/sqrt(eps_eff)

        returns:
            l_KID, in m. Same shape as f0
        '''
        return c/np.sqrt(self.CPW.eps_eff)/f0 / 4
    
    def f0_PEC(self, l_KID):
        '''
        Used to approximate resonance frequency of the hybrid quarterwave KID
        given a certain KID length. Sets v_phase = c and ignores kinetic inductance fraction.
        
        l_KID: number or array
            KID length, in m
        
        returns:
            Resonance frequency, in Hz
        '''
        return c/np.sqrt(self.CPW.eps_eff)/l_KID / 4
    
    def l_res(self, f0, Qc=1e10):
        '''
        Used to find the length of the KID to achieve resonance frequency f0.
        
        returns:
            l_res, in m
        '''
        res = minimize_scalar( lambda l_res: np.abs(f0 - self.f0(l_res, Qc)), bounds=(0, 2*self.l_KID(f0, Qc)) )
        return res.x
    
    def l_coupler(self, f0, Qc, savename=None, traces=[], **kwargs):
        '''
        Used to yield the interpolated coupler length based on design resonance frequency and
        coupling quality factor. Assumed is that the coupler design is built from the same CPW.
        Can also plot and save visual representation of this interpolation
        
        savename: str, kwarg
            If not None, savename is used as input for savename in sd.plot.heatmap() and a 
            representation of the interpolation is plotted and saved as savename.
        traces: list, kwarg
            traces to be passed on to sd.plot.heatmap()
        **kwargs:
            go to sd.plot.heatmap()
        
        returns:
            l_coupler, in m. Same shape as f0s
            If savename is not None, go.Heatmap figure
        '''
        f0 = np.atleast_1d(f0)*1e-9 # GHz
        logQc = np.log10(Qc)
        
        Qdata = Qdesign(self.CPW.name) # yield Qdata from Qdesign(), assuming wide CPW is used as coupler design
        f, l_c, logQ = Qdata[1:,0], Qdata[0,1:], Qdata[1:,1:]
        
        logQ0 = func.interp2D(f0, l_c, f, l_c, logQ) # logQ values at each f0 for l_c
        l_coupler  = np.zeros_like(f0)

        for i in range(len(f0)):
            l_coupler[i] = func.interp1D(logQc, logQ0[i], l_c)
        
        if savename is not None:
            plot.heatmap(logQ.T, f, l_c, labels=dict(col="f_0",col_unit="GHz",row="l_{coupler}",row_unit="\u03bcm",z="log(Qc)",z_unit=""),
                         traces=traces+[dict(type="scatter", mode="markers", x=f0, y=l_coupler, line_color="white")],
                         title_text=self.CPW.name+" coupler"+f"<br>Qc = {Qc:.1g} (log(Qc)={logQc:.2f})", savename=savename, **kwargs)
        else:
            return l_coupler*1e-6
    
    def C_coupler(self, f, f0, Qc):
        '''
        Used to yield the approximated capacitance based on quarter-wave resonator resonance frequency
        and coupling quality factor. Assumed is that the coupler design is built from the wide CPW.
        Simple adjustment to approximate coupler with different throughline CPW compared to wide CPW
            is to replace Z_CPW by sqrt(Z_CPW_tl * Z_CPW_wide).
        
        returns:
            Coupler capacitance, in F
        '''
        return np.sqrt(2/(np.pi*Qc)) / self.CPW.Z_CPW(f, self.T, self.mode) / f0 / 4
    
    def ABCD_coupler(self, f, f0, Qc):
        '''
        ABCD (transmission) matrix for the coupler as a two-port network
        '''
        Cc = self.C_coupler(f, f0, Qc)
        return np.array([[np.ones(np.shape(f)), 1/(2j*np.pi*f*Cc)], [np.zeros(np.shape(f)), np.ones(np.shape(f))]])
    
    def S(self, f, f0, Qc, correct_shift=True):
        '''
        Returns the transmission matrix of the circuit including capacitor.
        Assumed is that the coupler design is built from the wide CPW.
            
        returns:
            S, 2x2 matrix with each index being same shape as f
        '''
        ABCD = self.ABCD(f, f0, Qc, correct_shift)
        Zres = TL.ABCD_to_Zin(ABCD, self.Z_L(f))
        Zcpw = self.CPW.Z_CPW(f, self.T, self.mode)
        S11  =  -Zcpw / (2*Zres+Zcpw)
        S21  = 2*Zres / (2*Zres+Zcpw)
        return np.array([[S11, S21], [S21, S11]])
        
    def S21(self, f, f0, Qc, correct_shift=True):
        '''
        Returns the forward transmission S21 of the circuit including capacitor.
        Assumed is that the coupler design is built from the wide CPW
            
        returns:
            S21, same shape as f
        '''
        S = self.S(f, f0, Qc, correct_shift)
        return S[1,0]
    
    def S21_fit(self, f0, Qc, correct_shift=True, show_plot=False, bounds=[-10,10], xrange=[-3,3], **kwargs):
        '''
        Returns a dictionary containing multiple defining variables for the Lorentzian shape of |S21|^2.
            To obtain f0 and FWHM, uses minimization through scipy.optimize.minimize_scalar()
        
        bounds: list, kwarg
            bounds used for finding f0 and FWHM, in units of Ql
        show_plot: bool, kwarg
            If True, plots the resonance dip with |S21|^2 both linear and in dB.
            If correct_shift is False, the plot is adjusted accordingly.
        xrange: list, kwarg
            xrange used for plotting, in units of Ql
        **kwargs:
            go to sd.plot.plotly()
        
        returns:
            dict containing found values for f0, S21_min, FWHM, Ql, Qi, and Qc
        '''
        Ql_approx = Q_l(Qc, self.Qi_approx(f0))
        if correct_shift:
            bounds = f0*(np.array(bounds)/Ql_approx + 1)
        else:
            bounds = f0*(np.array(bounds)/Ql_approx + 1/(1+np.sqrt(2/np.pi/Qc)) )

        S21_lor = lambda f: np.abs( self.S21(f, f0, Qc, correct_shift) )**2
        res_S21 = minimize_scalar(S21_lor, bounds=bounds, options={'xatol':1})
        f0_found, S21_min = res_S21.x, np.sqrt(res_S21.fun)

        f1 = minimize_scalar(lambda f: np.abs( S21_lor(f)-(1+S21_min**2)/2 ), bounds=(bounds[0], f0_found), options={'xatol':1})
        f2 = minimize_scalar(lambda f: np.abs( S21_lor(f)-(1+S21_min**2)/2 ), bounds=(f0_found, bounds[1]), options={'xatol':1})
        FWHM = f2.x - f1.x
        Ql = f0_found / FWHM

        if show_plot:
            if correct_shift:
                xticks = [-2,-1,-0.5,0,0.5,1,2]
            else:
                xticks = np.array([-2,-1,-0.5,0,0.5,1,2]) + Ql*(1/(1+np.sqrt(2/np.pi/Qc)) - 1)
                xrange = np.array(xrange)                 + Ql*(1/(1+np.sqrt(2/np.pi/Qc)) - 1)
            
            xQl = np.linspace(*xrange, 1000 if self.mode is None else 50000) # mode=None is rather slow
            S21 = self.S21((xQl/Ql+1)*f0, f0, Qc, correct_shift)
            
            plot.plotly(xQl, np.abs(S21)**2, labels=dict(x="xQ_l",x_unit="",y="|S_{21}|^2",y_unit=""),
                        xaxis_tickvals=xticks, xrange=xrange, vlines=(np.array([f0_found, f1.x, f2.x])/f0-1)*Ql,
                        yrange=[0,1], hlines=[(1+S21_min**2)/2, S21_min**2], **kwargs)
            plot.plotly(xQl, 20*np.log10(np.abs(S21)), labels=dict(x="xQ_l",x_unit="",y="|S_{21}|^2",y_unit="dB"),
                        xaxis_tickvals=xticks, xrange=xrange, vlines=(np.array([f0_found, f1.x, f2.x])/f0-1)*Ql,
                        yrange=[-30,0], hlines=[10*np.log10((1+S21_min**2)/2), 20*np.log10(S21_min)], **kwargs)

        return dict(f0=f0_found, S21_min=S21_min, FWHM=FWHM, Ql=Ql, Qi=Ql/S21_min, Qc=Ql/(1-S21_min), r=(1-S21_min)/2)
    
    def observables(self, f, f0, Qc, correct_shift=True):
        '''
        Returns observables A en theta for the forward transmission S21
        
        returns: 
            A and theta, same shape as f
        '''
        T = deepcopy(self.T)
        self.T = self.T0 # set to T0
        r_0 = self.S21_fit(f0, Qc, correct_shift)["r"]
        
        self.T = T # change back to T
        S21 = self.S21(f, f0, Qc, correct_shift)
        
        A = 1 - np.sqrt( (np.real(S21) + r_0-1)**2 + np.imag(S21)**2 ) / r_0
        theta = np.arctan2( np.imag(S21), 1-r_0-np.real(S21) )
        return A, theta



class simple(_KID):
    def __init__(self, CPW, T, T0=None, mode=None, dl=0):
        '''
        CPW: CPW.create object
            Instance of CPW.create, the CPW the KID is made of
        T: number
            Temperature, in K
        T0: number, kwarg
            Design temperature, in K
            If None, T0 is set to equal the initial T. Only used in l_CPW() method
        mode: str, kwarg
            If "readout", uses approximation for sigma_1 and sigma_2 in superconducting limit (valid for kT, hf << 2*Delta)
            If "normal", defines sigma as sigma_N (i.e., 1/rho) (valid for kT, hf >> 2*Delta)
        dl: number, kwarg
            Any additional length to be added to the CPW length, in m.
            Can be used when a load impedance added through the method set_Z_L() contains
            a CPW of a certain length dl
        '''
        self.dl = dl # m
        
        if T0 is None:
            T0 = T
        
        super().__init__(CPW, T, T0, mode)
        
    def __str__(self):
        return self.CPW.name + " KID"
    
    '''
    METHODS
        f: number or array
            Readout frequency, in Hz
        f0: number or array
            Design resonance frequency of KIDs
        Qc: number
            Coupling quality factor
        l_KID: number or array
            KID length, in m
        correct_shift: bool, kwarg
            If True, Qc is passed on to l_CPW(). If False, Qc is set to None in l_CPW() method
    '''
        
    def l_CPW(self, f0, Qc=None):
        '''
        Used to estimate the length of the CPW to achieve resonance frequency f0 at temperature T0.
        Assumes that the wide CPW can be described by a lossless transmission line.
        Assumes Im{Z_in_CPW} = -Im{Z_TL_in} at f0 and open load for CPW (Z_L = inf).
            l_CPW = v_phase(f0) / (2pi*f0) * arccot(Im{Z_TL_in}(f0) / Z_CPW(f0))
        When no load impedance was added through set_Z_L() method, this reduces to 
            l_CPW = v_phase(f0) / (4*f0)
        
        Qc: number, kwarg
            Coupling quality factor. If not None, f0 is
            rescaled to account for an added C_coupler(f0, Qc)
        
        returns:
            l_CPW, in m. Same shape as f0
        '''
        f0_shifted = np.copy(f0)
        if Qc is not None:
            f0_shifted *= (1+np.sqrt(2 / np.pi / Qc))
        
        v_phase    = self.CPW.v_phase(f0_shifted, self.T0, self.mode)
        Z_CPW      = self.CPW.Z_CPW(f0_shifted, self.T0, self.mode)
        Z_TL_in_im = np.imag( self.Z_L(f0_shifted) ) # =0 if no load was added through set_Z_L() method
        
        return v_phase / (2*np.pi*f0_shifted) * np.arctan2(Z_CPW, Z_TL_in_im)
    
    def l_KID(self, f0, Qc=None):
        '''
        Used to estimate the length of the KID to achieve resonance frequency f0.
        If dl=0, l_KID() will yield the same length as l_CPW() method
        
        returns:
            l_KID, in m. Same shape as f0
        '''
        return self.l_CPW(f0, Qc) + self.dl
    
    def f0(self, l_KID, Qc):
        '''
        Used to find the resonance frequency of the hybrid quarterwave KID of length l_KID.
        Finds f0 by equating the complex part of the resonator's input impedance to zero
            minimizes the absolute value of Im{Zin} using scipy.optimize.minimize_scalar()
        Method based on Steven de Rooij's code
        
        returns:
            Resonance frequency, in Hz
        '''
        l_kid = np.array(l_KID,ndmin=1)
        f0_found = np.zeros(np.shape(l_kid))
        
        def min_func(f_i, l):
            abcd_coupler = self.ABCD_coupler(f_i, f_i, Qc)
            abcd_cpw     = self.CPW.ABCD(f_i, self.T, l-self.dl, self.mode)
            ABCD = abcd_coupler @ abcd_cpw
            Zres = TL.ABCD_to_Zin(ABCD, self.Z_L(f_i))
            return np.abs(np.imag( Zres ))
        
        for i, l in enumerate(l_kid):
            res = minimize_scalar(min_func, args=(l), bounds=(1, self.f0_PEC(l)), options={'xatol':1})
            f0_found[i] = res.x
        return np.reshape(f0_found, np.shape(l_KID))
    
    def f0_approx(self, l_KID):
        '''
        Used to find the approximate resonance frequency of the hybrid quarterwave KID at readout frequency f.
        Equals f0 = c*sqrt(1-alpha_k) / (4*l_KID*sqrt(eps_eff))
            where alpha_k is the kinetic inductance fraction of the KID
            and eps_eff is the effective relative permittivity
        Does one iteration with f_read=5GHz as first guess for f0
        
        returns:
            Resonance frequency, in Hz
        '''
        if self.dl != 0:
            warnings.warn("Extra length dl and load impedance are not taken into account.")
        f0_i = 1/np.sqrt( self.CPW.L_l(5e9, self.T, self.mode) * self.CPW.C_l ) / l_KID / 4
        return 1/np.sqrt( self.CPW.L_l(f0_i, self.T, self.mode) * self.CPW.C_l ) / l_KID / 4

    def Qi_approx(self, f0):
        '''
        Used to approximate the internal quality factor of the KID at f0 using Zin of the CPW.
        
        returns:
            Internal quality factor
        '''
        Z_TL_in = self.Z_L(f0)
        if not np.all(Z_TL_in): # use beta / (2 alpha)
            gamma = self.CPW.gamma(f0, self.T, self.mode)
            return np.imag(gamma) / (2*np.real(gamma))
        else:
            return np.imag(Z_TL_in) / np.real(Z_TL_in)
        
    def alpha_k(self, f0, Qc):
        '''
        Used to find the kinetic inductance fraction of the KID.
        
        returns:
            alpha_k, the kinetic inductance fraction
        '''
        l_KID = self.l_KID(f0, Qc)
        
        PEC_KID = deepcopy(self)
        PEC_KID.CPW.mat_c = CPW.PEC(self.CPW.mat_c.d)
        return 1 - (self.f0(l_KID, Qc) / PEC_KID.f0(l_KID, Qc))**2
    
    def ABCD_res(self, f, f0, Qc=None):
        '''
        ABCD (transmission) matrix for the resonator as a two-port network
        Length of wide CPW is based on input for f0
        
        returns:
            ABCD (transmission) matrix
        '''
        if self.dl != 0:
            warnings.warn("Extra length dl and load impedance are not taken into account.")
        return self.CPW.ABCD(f, self.T, self.l_CPW(f0, Qc), self.mode)
    
    def ABCD(self, f, f0, Qc, correct_shift=True):
        '''
        ABCD (transmission) matrix for the KID as a two-port network.
        Assumed is that the coupler design is built from the same CPW.
        Length of CPW is based on input for f0 and accounts for shift due to
        added capacitor through Qc
        
        returns:
            ABCD (transmission) matrix
        '''
        if correct_shift:
            abcd_coupler = self.ABCD_coupler(f, f0*(1+np.sqrt(2/np.pi/Qc)), Qc) # f_0 --> f_1/4
            abcd_cpw = self.CPW.ABCD(f, self.T, self.l_CPW(f0, Qc), self.mode)
        else:
            abcd_coupler = self.ABCD_coupler(f, f0, Qc)
            abcd_cpw = self.CPW.ABCD(f, self.T, self.l_CPW(f0, None), self.mode)
        
        if isinstance(f, (int,float)): 
            return abcd_coupler @ abcd_cpw # same as np.einsum('ij,jk->ik', ...)
        else: # last axis (frequency) has to be simple multiplication
            return np.einsum('ijl,jkl->ikl', abcd_coupler, abcd_cpw)
        
    def resp_approx(self, f, f0, Qc, correct_shift=True):
        '''
        Approximate change in observables A and theta for change in quasiparticle density, in N0
        Uses thin film limit for beta,
            1+2*self.CPW.mat_c.d*np.sqrt(mu_0*s2*2*np.pi*f)/np.sinh(2*self.CPW.mat_c.d*np.sqrt(mu_0*s2*2*np.pi*f)) = 2
        
        returns:
            dA/dnqp and dtheta/dnqp, both same shape as f
        
        '''
        _, s2 = self.CPW.mat_c.sigma12(f, self.T)
        ak = self.alpha_k(f0, Qc)
        Bl = 2 # thin film limit
        Ql = self.S21_fit(f0, Qc, correct_shift)["Ql"]
        
        factor = ak*Bl*Ql / s2
        ds1dnqp, ds2dnqp = self.CPW.mat_c.dsigmadnqp_approx(f, self.T)
        dAdnqp     =  factor*ds1dnqp
        dthetadnqp = -factor*ds2dnqp
        return dAdnqp, dthetadnqp



class hybrid(_KID):
    def __init__(self, narrow_CPW, wide_CPW, l_hybrid=1e-3, T=None, T0=None, mode=None):
        '''
        narrow_CPW: CPW.create object
            Instance of CPW.create, representing the narrow or hybrid CPW
        wide_CPW: CPW.create object
            Instance of CPW.create, representing the wide CPW
        l_hybrid: number or array, kwarg
            Length of the narrow CPW, in m, default is 1 mm.
            Can be a number or an array of length #design resonance frequencies
        T: number, kwarg
            If None, T is set to be at 'operating temperature', or Tc/10 for the active metal.
        T0: number, kwarg
            Design temperature, in K.
            If None, T0 is set to equal the initial T. Only used in l_wide() method
        mode: str, kwarg
            If "readout", uses approximation for sigma_1 and sigma_2 in superconducting limit (valid for kT, hf << 2*Delta)
            If "normal", defines sigma as sigma_N (i.e., 1/rho) (valid for kT, hf >> 2*Delta)
        '''
        self.narrow   = narrow_CPW
        self.wide     = wide_CPW
        self.l_hybrid = l_hybrid # m
        
        if T is None:
            T = self.narrow.mat_c.Tc / 10 # K
        if T0 is None:
            T0 = T
            
        super().__init__(self.wide, T, T0, mode)
        
    def __str__(self):
        return "{0:g}\u03bcm {1} - l_wide(f0,Qc) {2}".format(self.l_hybrid*1e6, self.narrow.name, self.wide.name)
    
    '''
    METHODS
        f: number or array
            Readout frequency, in Hz
        f0: number or array
            Design resonance frequency of KIDs
        l_KID: number or array
            KID length, in m
        Qc: number, kwarg
            Coupling quality factor
        correct_shift: bool, kwarg
            If True, Qc is passed on to l_wide(). If False, Qc is set to None in l_wide() method
    '''
        
    def l_wide(self, f0, Qc=None):
        '''
        Used to estimate the length of the wide CPW to achieve resonance frequency f0 at temperature T0.
        Assumes that the wide CPW can be described by a lossless transmission line.
        Assumes Im{Z_in_wide} = -Im{Z_TL_in} at f0 and open load for wide CPW (Z_L = inf).
            l_wide = v_phase(f0) / (2pi*f0) * arccot(Im{Z_TL_in}(f0) / Z_CPW(f0))
        Also works if l_hybrid is an array corresponding to the different values of f0, but
        has to have the same length as f0.
        
        Qc: number, kwarg
            Coupling quality factor. If not None, f0 is
            rescaled to account for an added C_coupler(f0, Qc)
        
        returns:
            l_wide, in m. Same shape as f0
        '''
        if not isinstance(self.l_hybrid, (int, float)):
            if len(l_hybrid) != len(f0):
                raise Exception("Lengths of l_hybrid and f0 should match.")
        
        f0_shifted = np.copy(f0)
        if Qc is not None:
            f0_shifted *= (1+np.sqrt(2 / np.pi / Qc))
        
        v_phase    = self.wide.v_phase(f0_shifted, self.T0, self.mode)
        Z_CPW      = self.wide.Z_CPW(f0_shifted, self.T0, self.mode)
        Z_TL_in_im = np.imag( TL.Z_in(f0_shifted, self.T0, self.narrow, self.Z_L(f0_shifted), self.l_hybrid, self.mode) )
        
        return v_phase / (2*np.pi*f0_shifted) * np.arctan2(Z_CPW, Z_TL_in_im)
        
    def l_KID(self, f0, Qc=None):
        '''
        Used to estimate the total length of the hybrid KID to achieve resonance frequency f0.
        Uses l_wide() and thus Z_L() in the process.
        
        returns:
            l_KID, in m. Same shape as f0
        '''
        return self.l_wide(f0,Qc) + self.l_hybrid
    
    def l_KID_approx(self, f0):
        '''
        Yield an approximate for the KID length given design resonance frequency f0.
        Found by solving for f0=1/sqrt(L*C)/4 with l_wide being the only unknown.

        returns:
            l_KID, in m. Same shape as f0
        '''
        L_l_wide = self.wide.L_l(f0, self.T0, self.mode)
        L_l_narrow = self.narrow.L_l(f0, self.T0, self.mode)

        a = L_l_wide * self.wide.C_l
        b = self.l_hybrid * (L_l_wide*self.narrow.C_l + L_l_narrow*self.wide.C_l)
        c = L_l_narrow * self.narrow.C_l * self.l_hybrid**2 - 1/(16*f0**2)

        l_wide = (np.sqrt(b**2-4*a*c)-b)/2/a
        return l_wide + self.l_hybrid
    
    def L(self, f, l_KID):
        '''
        Used to find the inductance of the KID at frequency f given a certain KID length.
        
        returns:
            Inductance, in H
        '''
        return self.wide.L_l(f, self.T, self.mode) * (l_KID-self.l_hybrid) + self.narrow.L_l(f, self.T, self.mode) * self.l_hybrid
    
    def C(self, f, l_KID):
        '''
        Used to find the capacitance of the KID at frequency f given a certain KID length.
        
        returns:
            Capacitance, in F
        '''
        return self.wide.C_l * (l_KID-self.l_hybrid) + self.narrow.C_l * self.l_hybrid
    
    def f0(self, l_KID, Qc):
        '''
        Used to find the resonance frequency of the hybrid quarterwave KID of length l_KID.
        Finds f0 by equating the complex part of the resonator's input impedance to zero
            minimizes the absolute value of Im{Zin} using scipy.optimize.minimize_scalar()
        Method based on Steven de Rooij's code
        
        returns:
            Resonance frequency, in Hz
        '''
        l_kid = np.array(l_KID,ndmin=1)
        f0_found = np.zeros(np.shape(l_kid))
        
        def min_func(f_i, l):
            abcd_coupler = self.ABCD_coupler(f_i, f_i, Qc)
            abcd_wide    = self.wide.ABCD(f_i, self.T, l-self.l_hybrid, self.mode)
            abcd_narrow  = self.narrow.ABCD(f_i, self.T, self.l_hybrid, self.mode)
            ABCD = abcd_coupler @ abcd_wide @ abcd_narrow
            Zres = TL.ABCD_to_Zin(ABCD, self.Z_L(f_i))
            return np.abs(np.imag( Zres ))
        
        for i, l in enumerate(l_kid):
            res = minimize_scalar(min_func, args=(l), bounds=(1, self.f0_PEC(l)), options={'xatol':1})
            f0_found[i] = res.x
        return np.reshape(f0_found, np.shape(l_KID))
    
    def f0_approx(self, l_KID):
        '''
        Used to find the approximate resonance frequency of the hybrid quarterwave KID at readout frequency.
        Equals f0 = c*sqrt(1-alpha_k) / (4*l_KID*sqrt(eps_eff))
            where alpha_k is the kinetic inductance fraction of the KID
            and eps_eff is the effective relative permittivity
        Does one iteration with f_read=5GHz as first guess for f0
        
        returns:
            Resonance frequency, in Hz
        '''
        f0_i = 1/np.sqrt(self.L(5e9, l_KID) * self.C(5e9, l_KID)) / 4
        return 1/np.sqrt(self.L(f0_i, l_KID) * self.C(f0_i, l_KID)) / 4

    def Qi_approx(self, f0, Qc=None):
        '''
        Used to approximate the internal quality factor of the KID at f0 using Zin of the narrow CPW.
        
        returns:
            Internal quality factor
        '''
        Z_TL_in_narrow = TL.Z_in(f0, self.T, self.narrow, self.Z_L(f0), self.l_hybrid, self.mode)
        Zc = 1/(2j*np.pi*f0*self.C_coupler(f0,f0,Qc)) if Qc is not None else np.inf
        Z_TL_in_wide = TL.Z_in(f0, self.T, self.wide, Zc, self.l_wide(f0,Qc), self.mode)
        return (np.imag(Z_TL_in_narrow) - np.imag(Z_TL_in_wide)) / (np.real(Z_TL_in_narrow) + np.real(Z_TL_in_wide))
    
    def alpha_k(self, f0, Qc):
        '''
        Used to find the kinetic inductance fraction of the KID.
        
        returns:
            alpha_k, the kinetic inductance fraction
        '''
        l_KID = self.l_KID(f0, Qc)
        
        PEC_KID = deepcopy(self)
        PEC_KID.narrow.mat_c = CPW.PEC(self.narrow.mat_c.d)
        return 1 - (self.f0(l_KID, Qc) / PEC_KID.f0(l_KID, Qc))**2
    
    def alpha_k_approx(self, f, Qc=None):
        '''
        Used to approximate the kinetic inductance fraction of the KID.
        Only takes the wide and narrow CPWs into account. 
        
        returns:
            alpha_k, the kinetic inductance fraction
        '''
        l_KID = self.l_KID(f, Qc)
        return (self.wide.Lk_l(f, self.T, self.mode) * (l_KID-self.l_hybrid) + 
                self.narrow.Lk_l(f, self.T, self.mode) * self.l_hybrid) / self.L(f, l_KID)
    
    def ABCD_res(self, f, f0, Qc=None):
        '''
        ABCD (transmission) matrix for the resonator as a two-port network
        Length of wide CPW is based on input for f0
        
        returns:
            ABCD (transmission) matrix
        '''
        abcd_wide = self.wide.ABCD(f, self.T, self.l_wide(f0, Qc), self.mode)
        abcd_narrow = self.narrow.ABCD(f, self.T, self.l_hybrid, self.mode)
        
        if isinstance(f, (int,float)): 
            return abcd_wide @ abcd_narrow # same as np.einsum('ij,jk->ik', ...)
        else: # last axis (frequency) has to be simple multiplication
            return np.einsum('ijm,jkm->ikm', abcd_wide, abcd_narrow)
    
    def ABCD(self, f, f0, Qc, correct_shift=True):
        '''
        ABCD (transmission) matrix for the KID as a two-port network.
        Assumed is that the coupler design is built from the wide CPW.
        Length of wide CPW is based on input for f0 and accounts for shift due to
        added capacitor through Qc
        
        returns:
            ABCD (transmission) matrix
        '''
        if correct_shift:
            abcd_coupler = self.ABCD_coupler(f, f0*(1+np.sqrt(2/np.pi/Qc)), Qc) # f_0 --> f_1/4
            abcd_wide = self.wide.ABCD(f, self.T, self.l_wide(f0, Qc), self.mode)
        else:
            abcd_coupler = self.ABCD_coupler(f, f0, Qc)
            abcd_wide = self.wide.ABCD(f, self.T, self.l_wide(f0, None), self.mode)
        abcd_narrow = self.narrow.ABCD(f, self.T, self.l_hybrid, self.mode)
        
        if isinstance(f, (int,float)): 
            return abcd_coupler @ abcd_wide @ abcd_narrow # same as np.einsum('ij,jk,kl->il', ...)
        else: # last axis (frequency) has to be simple multiplication
            return np.einsum('ijm,jkm,klm->ilm', abcd_coupler, abcd_wide, abcd_narrow)
        
    def resp_approx(self, f, f0, Qc, correct_shift=True):
        '''
        Approximate change in observables A and theta for change in quasiparticle density, in N0
        Uses thin film limit for beta,
            1+2*self.narrow.mat_c.d/np.sqrt(mu_0*s2*2*np.pi*f)/np.sinh(2*self.narrow.mat_c.d/np.sqrt(mu_0*s2*2*np.pi*f))
        
        returns:
            dA/dnqp and dtheta/dnqp, both same shape as f
        
        '''
        _, s2 = self.narrow.mat_c.sigma12(f, self.T)
        ak = self.alpha_k(f0, Qc)
        Bl = 2 # thin film limit
        Ql = self.S21_fit(f0, Qc, correct_shift)["Ql"]
        
        factor = ak*Bl*Ql / s2
        ds1dnqp, ds2dnqp = self.narrow.mat_c.dsigmadnqp_approx(f, self.T)
        dAdnqp     =  factor*ds1dnqp
        dthetadnqp = -factor*ds2dnqp
        print(factor, ds1dnqp, ds2dnqp)
        return dAdnqp, dthetadnqp



def resonator(f, f0, Qc, Qi):
    '''
    Approximate resonator S12(f).
    
    f: array
        Readout frequency
    f0: number
        Resonance frequency, same unit as f
    Qc: number
        Coupling quality factor
    Qi: number
        Internal quality factor
    
    returns:
        S12(f). Same shape as f
    '''
    x = (f-f0)/f0
    Ql = Q_l(Qc, Qi)
    return (Ql/Qi + 2j*Ql*x)/(1 + 2j*Ql*x) 
    
def Q_l(*args):
    '''
    Loaded quality factor
    
    Qc: number or array
        Coupling quality factor
    Qi: number or array
        Internal quality factor
    Qi ...
        
    returns:
        Loaded quality factor
    '''
    return 1 / sum([1/q for q in args])

def fit_S21(f, S21, Qc_guess=1e4, Qi_guess=1e4):
    '''
    Fit S21(f) data and return found variables and their standard deviations.
    
    f: array
        Readout frequency
    S21: array
        Forward transmission S21, same shape as f
    Qc_guess: number
        Coupling quality factor
    Qi_guess: number
        Internal quality factor
    
    returns:
        pvar: array containing fit variables [f0, Qc, Qi]
        pstd: array containing standard deviation in fit variables
    '''
    def S12_fitfunc(f, f0, Qc, Qi): # f and f0 in GHz
        complex_vals = resonator(f, f0, Qc, Qi)
        return np.concatenate([np.real(complex_vals), np.imag(complex_vals)])

    pvar, pcov = curve_fit(S12_fitfunc, f, np.concatenate([np.real(S21), np.imag(S21)]),
                           p0    = (f[np.argmin(S21)], Qc_guess, Qi_guess),
                           bounds=([np.min(f),                0,        0],
                                   [np.max(f),           np.inf,   np.inf]))
    pstd = np.sqrt(np.diag(pcov))
    return pvar, pstd

def Qdesign(design):
    '''
    Same input formatting used as in Jochem Baselmans' mask code.
    
    design: str
        Name of CPW the coupler is made of.
        If the CPW is an instance of CPW.create(), the CPW name can be called using .name
    
    returns:
        matrix containing interpolation data for logQ:
            Qdesign[1:,0], first column contains f data (in GHz) from index 1 onward
            Qdesign[0,1:], first row contains l_c data (in um) from index 1 onward
            Qdesign[1:,1:], matrix contains logQ data (in log10(Qc)) with rows defined by and columns by l_c
    '''
    if design == "NbTiN 8-20-8":
        return np.array([[0,20,30,40,50,60,80,100,120,140,160,180,200,250,300,400,600,800,1000,1200],[1,7.23495985760615,7.01595289885715,6.89533679665315,6.74083774881715,6.65120023870115,6.46069492782915,6.30453175366915,6.17227506855115,6.05764865839915,5.95654447882415,5.86611163070515,5.78428849537915,5.60178948779015,5.46106474843415,5.22507199419915,4.88846342891915,4.64561453365915,4.45664750650015,4.30150895721915],[2,6.63285598779115,6.41384069468415,6.29322312286315,6.13872355771315,6.04908578632115,5.85858278844315,5.70242025773515,5.57016368542115,5.45553839093815,5.35443564033615,5.26400216538415,5.18217794824715,4.99968667368915,4.85900368873615,4.62309006833015,4.28652521546615,4.04391030022215,3.85518057486715,3.70031756122315],[3,6.28059843499315,6.06156948659915,5.94094946318415,5.78644904775915,5.69681085726915,5.50631176214715,5.35015038049015,5.21789408963215,5.10327075835515,5.00217050227115,4.91173610856715,4.82991022530915,4.64743222369315,4.50681924054715,4.27103870676215,3.93455070725015,3.69233264699415,3.50400971124415,3.34962340943715],[4,6.03061231049715,5.81156473417615,5.69094127604415,5.53643969882015,5.44680095832715,5.25630743325315,5.10014782729315,4.96789213278515,4.85327177374415,4.75217525269915,4.66173984229015,4.57991191867615,4.39745331577815,4.25693920306615,4.02134750797615,3.68497546790015,3.44332735004215,3.25559734782215,3.10191463101915],[5,5.83664679947715,5.61757609664015,5.49694822076815,5.34244520298415,5.25280582103515,5.06231964646115,4.90616261726215,4.77390804368315,4.65929189656415,4.55820060012015,4.46776434958415,4.38593430759715,4.20350204891215,4.06311653695415,3.82777192417415,3.49156316071315,3.25067205575415,3.06374381806615,2.91102636746015],[6,5.67809822340515,5.45900048269915,5.33836720903815,5.18386251881615,5.09422245781815,4.90374556762415,4.74759214923315,4.61533949955015,4.50072910866715,4.39964485392215,4.30920829828315,4.22737644491615,4.04497853049315,3.90475244498815,3.66971631674515,3.33372846867815,3.09379888446215,2.90790930879115,2.75646149678315],[7,5.54397388876115,5.32484588489315,5.20420624135715,5.04969971122115,4.96005900407815,4.76959353151615,4.61344505580115,4.48119548641115,4.36659278059115,4.26551779600515,4.17508191680915,4.09324903603815,3.91089475612815,3.77086024270915,3.53619779340015,3.20050075361715,2.96175767156215,2.77717623113715,2.62735135182015],[8,5.42771045879315,5.20854973252215,5.08790276099115,4.93339430797915,4.84375307668415,4.65330140594215,4.49715957447415,4.36491467780915,4.25032205745415,4.14925907342515,4.05882539232215,3.97699284159515,3.79469301533815,3.65488375450615,3.42066456648115,3.08534220436415,2.84803360279515,2.66506576842915,2.51727018833915]])
    
    else:
        raise Exception("Coupler design does not exist in Qdesign(). Add to KID.Qdesign() in required format.")