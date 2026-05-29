'''
This module contains several methods from transmission line theory. See Pozar.
'''
import numpy as np

def ABCD_to_Zin(ABCD, ZL):
    '''
    Returns input impedance from ABCD and load impedances
    '''
    return (ABCD[0,0]*ZL+ABCD[0,1]) / (ABCD[1,0]*ZL+ABCD[1,1])

def ABCD_to_Z(ABCD):
    '''
    Convert ABCD matrix to impedance matrix
    '''
    [[A,B],[C,D]] = ABCD
    return np.array([[A/C, (A*D-B*C)/C], [1/C, D/C]])
   
def Z_in(f, T, TL, ZL, l, mode=None):
    '''
    Returns transmission line input impedance
    
    f: number or array
        Frequency, in Hz
    T: number
        Temperature in K
    TL: object of a transmission line class
        Currently only supports CPW.create()
    ZL: number (or np.inf) or array
        Load impedance in Ohms. Can also be an array of the same shape as f
    l: number
        Length of transmission line in m
    mode: None or str, kwarg
        Input used for TL methods

    Returns:
        Zin, the input impedance in Ohms
    '''
    Z0 = TL.Z0(f,T,mode)
    tanh_yl = np.tanh( TL.gamma(f,T,mode)*l )
    
    if np.isinf(ZL).all():
        return Z0 / tanh_yl
    else:
        return Z0 * (ZL + Z0*tanh_yl) / (Z0 + ZL*tanh_yl)

def Gamma(Z0, ZL, power=True):
    '''
    Voltage reflection coefficient
    
    Power: bool, kwarg
       If True, the conjugate is taken of Z0 to consider power waves
       Else, considering voltage waves. Yields the same when Z0 is real
    '''
    if power:
        return (ZL-np.conjugate(Z0)) / (ZL+Z0)
    else:
        return (ZL-Z0) / (ZL+Z0)

def reflection(Z0, ZL, power=True):
    '''
    Power reflection
    '''
    return np.abs(Gamma(Z0, ZL, power))**2

def transmission(Z0, ZL, power=True):
    '''
    Power transmission
    '''
    return 1-reflection(Z0, ZL, power)