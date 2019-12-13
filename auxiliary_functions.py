import os,sys
import numpy as np


def directory():
	path  = os.getcwd()
	Updir = os.path.dirname(path)
	
	if not sys.path[-1]==Updir:
		sys.path += [Updir]
			
	
def read_parameters():
	if sys.version_info[0]==2:
		import ConfigParser
		config = ConfigParser.RawConfigParser()
	elif sys.version_info[0]==3:
		import configparser
		config = configparser.ConfigParser()
	
	directory()
	import Cosmo_functions as cf
	
	#import Cosmo_functions as cf
	INI         = "parameters.ini"
	name_params = os.path.join(os.path.dirname(os.getcwd()),INI)
	config.read(name_params)
	#Cosmology
	Ob_h2           = config.getfloat("Cosmology","Obh2")
	Oc_h2           = config.getfloat("Cosmology","Och2")
	Ok_h2           = config.getfloat("Cosmology","Okh2")
	h               = config.getfloat("Cosmology","h")
	TCMB            = config.getfloat("Cosmology","TCMB")
	Neff            = config.getfloat("Cosmology","Neff")
	model_w         = config.get(     "Cosmology","model_w")
	w0              = config.getfloat("Cosmology","w0")
	wa              = config.getfloat("Cosmology","wa")
	h2              = h**2
	Om_h2           = Ob_h2+Oc_h2
	params = {"Ob_h2":Ob_h2,"Oc_h2":Oc_h2,"Ok_h2":Ok_h2, "Om_h2":Om_h2,"h":h,"h2":h2,
	          "TCMB":TCMB,"Neff":Neff,"w0":w0,"wa":wa,"model_w":model_w}
	Or_h2           = cf.Orh2(params)
	Od_h2           = h2 - (Om_h2 + Ok_h2 + Or_h2)
	params["Or_h2"] = Or_h2
	params["Od_h2"] = Od_h2

	return params
	
def comoving_distance(z,params):
	directory()
	import Cosmo_functions as cf
	return cf.comoving_distance(z,params)

def cosmic_variance(L,CL,fsky):
    return np.sqrt(2./(2.*l+1)/np.pi/fsky)*CL

def data_distortion(L,BAO,N,fsky):
    loc  = np.sort(np.random.choice(np.arange(N),N))
    x    = L[loc]
    yerr = 0.04 * np.random.rand(N)
    y    = BAO[loc]
    y    += yerr* np.random.randn(N)
    yerr = cosmic_variance(L,BAO,fsky) 
    return x,y,yerr	
	
def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1 #i*2, return i with the bits shifted to the left by "1" places
    return i


def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f    = np.fft.fft(x - np.mean(x), n=2 * n)
    acf  = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def outputs_path(nbins):
	
	path    = os.getcwd()
	pathout = os.path.join(path, "outputs")
	pathbin = os.path.join(pathout, str(nbins)+"bin")
	
	if not os.path.isdir(pathout):
		os.mkdir(pathout)
		os.mkdir(pathbin)
	else:
		if not os.path.isdir(pathbin):
			os.mkdir(pathbin)
	return pathbin

def remove_space_input(var):
	var = np.char.strip(var)
	return np.asarray(var.astype(np.float))
	

