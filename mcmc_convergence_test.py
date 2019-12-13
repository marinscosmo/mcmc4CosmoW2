#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys
import numpy as np
import auxiliary_functions as auxf
from time import time,strftime, gmtime

def run_emcee(params,paramsOut,prior_range):
	import emcee
    
	print("\n")
	print("================================================================================================")
	print("PROGRAM: MCMC TO CALCULATE THE CL PHENOMENOLOGICAL FITS PARAMETERS FOR THE BAOs\n")
	print("================================================================================================")
	print("\n")

	A0         = params['A0']
	alpha0     = params['alpha0']
	SigmaSilk0 = params['Silk0']

	a_fid      = np.array([A0,alpha0,SigmaSilk0])

	ndim         = len(a_fid)  
	nwalkers     = params['nwalkers']
	nsteps       = params['nsteps']
	steps        = params['steps']
	burn_steps   = params['burn_steps']	
	n_bins       = params['nbins']
	bin_analysis = params['bin_analysis']
	lmin         = params["use_range_l"][0]
	lmax         = params["use_range_l"][1]
	factor_N     = params["factor_N"]

	########################################################################
	#
	# Way of inputs and outputs
	#
	########################################################################
	
	path    = paramsOut["path_datas"]
	path_cl = os.path.join(path,paramsOut["filename_data"])
	path_z  = os.path.join(path,paramsOut["filename_redshifts"])

	########################################################################
	#
	# DATAS/INPUT
	#
	########################################################################
	fsky  = 1.
	datas = np.loadtxt(path_cl,unpack=True)
	l     = datas[0,:]
	lmin_range = np.amin(l)
	lmax_range = np.amax(l)
	l     = l[int(lmin-lmin_range):int(lmax-lmin_range+1)]
	cl    = datas[bin_analysis,:]
	cls   = datas[bin_analysis + n_bins,:]
	bao   = (cl/cls)[int(lmin-lmin_range):int(lmax-lmin_range+1)]
	err   = np.sqrt(2./(2.*l+1.)/fsky)*bao	
	
	if params["data_distortion"]:
		l,bao,err = auxf.data_distortion(l,bao,len(l),fsky)

	params_cosmo = auxf.read_parameters()

	print("=================================================================")
	print("Biggest  error:   " + str(int(1e4*np.amax(err))/1e4))
	print("Smallest error:   " + str(int(1e4*np.amin(err))/1e4))
	print("=================================================================")
	print("\n")

	########################################################################
	#
	# Define EMCEE parameters
	#
	########################################################################

	print("=================================================================")
	print("Num. loops  : " + str(nsteps))
	print("Num. steps  : " + str(steps))
	print("Num. walkers: " + str(nwalkers))
	print("Num. burning: " + str(burn_steps))
	print("Num. points : " + str(nwalkers*nsteps*steps))
	print("=================================================================")
	print("\n")
	


	########################################################################
	#
	# Vector of binned redshits values and sound horizon rs
	#
	########################################################################
	

	z_vector  = np.loadtxt(path_z)
	zeff      = z_vector[bin_analysis]
	rs        = params["rs"]


	print("=================================================================")
	print("Survey redshift range:(" + str(int(np.amin(z_vector)*1000.)/1000.) + "," + str(int(np.amax(z_vector)*1000.)/1000.) + ")")
	print("Num. bins            : " + str(n_bins))
	print("Bin analyzed         : " + str(bin_analysis))
	print("Effective redshift   : " + str(int(zeff*1000.)/1000.))
	print("=================================================================")
	print("\n\n")



	########################################################################
	#
	# Define functions: Likelihood, Prior, data theory
	#
	########################################################################

	def theory(a,l):

		A         = a[0]
		alpha     = a[1]
		SigmaSilk = a[2] 
		
		kr   = np.sqrt(l*(l+1))
		k    =  kr/auxf.comoving_distance(zeff,params_cosmo)
		f_l  =  1. + A*k*np.exp(-(k*SigmaSilk)**1.4)*np.sin(alpha*k*rs)
		return f_l

	def lnprior(a_try):
		if ((prior_range[:,0]<a_try)*(prior_range[:,1]>a_try)).all()==False: 
			return -np.inf
		return 0.0

	def lnprob(a_try,data_vec):
		lp = lnprior(a_try) 
		if not np.isfinite(lp): 
			return -np.inf      
		try_vec  = theory(a_try,l)
		diff_vec = try_vec - data_vec
		chi2_try = np.sum(diff_vec**2/err**2)	
		return -chi2_try/2.0

	########################################################################
	#
	# 
	#
	########################################################################

	def file_chains():
		path_chainsdir  = os.path.join(os.getcwd() , 'chains')
		path_file_bin   = os.path.join(path_chainsdir , str(n_bins) + '_bins')
		path_chains_bin = os.path.join(path_file_bin,'bin_' + str(bin_analysis))
		
		if not os.path.isdir(path_chainsdir):
			os.mkdir(path_chainsdir)
		
		if os.path.isdir(path_file_bin):
			if os.path.isdir(path_chains_bin):
				dirr = os.listdir(path_chains_bin)
				for filee in dirr:
					os.remove(path_chains_bin + os.sep + filee)
			else:
				os.mkdir(path_chains_bin)
				
		else:
			os.mkdir(path_file_bin)
			os.mkdir(path_chains_bin)

		g = open(os.path.join(path_chains_bin,'zeff.dat'),'w')
		g.write(str(zeff))
		g.close()

		return path_chains_bin
		
	########################################################################
	#
	# Convergence Diagnostic: Gelman-Rubin and autocorrelation test
	#
	########################################################################

	def Gelman_Rubin(chains_f=None):
		#calculate mean to mth chains:     Sigma_hat_m
		#calculate mean between chains:    Sigma_hat
		#calculate variance to mth chains: Var_hat_m
		#Let N be elements chains number in a chains
		#Let M be chains number
		#B = frac(N)(M-1) Sum_{m=1}^{N} (Sigma_hat_m - Sigma_hat)**2 , between-chains variance
		#W = frac(1)(M)Sum_{m=1}^{M} (Var_hat_m)^{2} ,                 within-chains variance
		#V_hat = frac(N-1)(M)W + frac(M+1)(NM)B
		return None
		
	def autocorrelation(y, c=5.0): #y=chain
		f = np.zeros(y.shape[1])
		for yy in y:
			f += auxf.autocorr_func_1d(yy)
		f /= len(y)
		taus   = 2.0 * np.cumsum(f) - 1.0
		window = auxf.auto_window(taus, c)
		return taus[window]
		
	########################################################################
	#
	# Initial chains and conditions
	#
	########################################################################

	#prior_range = np.array([a0_range,a1_range,a2_range])

	a0_0  = np.random.uniform(prior_range[0][0],prior_range[0][1],nwalkers)
	a1_0  = np.random.uniform(prior_range[1][0],prior_range[1][1],nwalkers)
	a2_0  = np.random.uniform(prior_range[2][0],prior_range[2][1],nwalkers)
	par_0 = np.array([a0_0,a1_0,a2_0]).T 
	
	########################################################################
	#
	# Begin program
	#
	########################################################################
	print("=================================================================")
	print("Burning...")

	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[bao])
	
	timei=time()
	pos, prob, state = sampler.run_mcmc(par_0, burn_steps)  
	burnsamp = sampler.chain 
	print("=================================================================")
	print("\n\n")
	
	path_chains = file_chains()
	f           = open(os.path.join(path_chains,"chain0.dat"), "w")
	f.close()

	sampler.reset() 
	posf, probf, statef = sampler.run_mcmc(pos, steps)
	flat_sample         = sampler.chain.reshape((-1, ndim))

	f = open(os.path.join(path_chains,"chain0.dat"), "a")
	for k in range(flat_sample.shape[0]):
		f.write(str(flat_sample[k])[2:-2] + "\n")
	f.close()

	print("=================================================================")
	print("Building chains:")
	
	
	count     = 1
	autocorr  = []
	autocorr0 = []
	autocorr1 = []
	autocorr2 = []
	N_50      = []
	nconvergence = 0
	
	for nstore in range(nsteps):
		posf, probf, statef = sampler.run_mcmc(posf, steps)
		chain1              = sampler.chain[:,:,1]
		N                   = np.exp(np.log(chain1.shape[1])).astype(int)
				
		#This part will only work if: params['chains_txt'] = True
		if paramsOut['chains_txt']:
			f = open(os.path.join(path_chains,"chain"+str(nstore+1)+ ".dat"), "a")
			this_sample = sampler.chain.reshape((-1, ndim))
			for k in range(this_sample.shape[0]):
				f.write(str(this_sample[k])[2:-2] + "\n")
			f.close()
		#
		if count==params['steps_convergence_test']:
			count=0
			if params["type_convergence"]=="EMCEE_autocorrelation":
				if nstore==0: 
					print("EMCEE autocorrelation analysis.")
				N_50 = np.append(N_50,N/factor_N)
				print("chain "+ str(nstore + 1) + "/" + str(nsteps))
				try:
					if np.where(params['convergence_test_parameter']==1)[0]>=0:
						chain0 = sampler.chain[:,:,0]
						autocorr0  = np.append(autocorr0,autocorrelation(chain0))
						if autocorr0[-1]<N/factor_N:
							print("-->(A)mplitude converged")
							nconvergence += 1
							num = np.where(params['convergence_test_parameter']==1)[0][0]
							params['convergence_test_parameter'][num]=-1
						else:
							print("-->(A)mplitude did not converged")
				except:
					pass
				try:
					if  np.where(params['convergence_test_parameter']==2)[0]>=0:
						autocorr1  = np.append(autocorr1,autocorrelation(chain1))
						if autocorr1[-1]<N/factor_N:
							print("-->alpha converged")
							nconvergence += 1
							num = np.where(params['convergence_test_parameter']==2)[0][0]
							params['convergence_test_parameter'][num]=-1
						else:
							print("-->alpha did not converged")
				except:
					pass
				try:
					if  np.where(params['convergence_test_parameter']==3)[0]>=0:
						chain2 = sampler.chain[:,:,2]
						autocorr2  = np.append(autocorr2,autocorrelation(chain2))
						if autocorr2[-1]<N/factor_N:
							print("-->Sigma converged")
							nconvergence += 1
							num = np.where(params['convergence_test_parameter']==3)[0][0]
							params['convergence_test_parameter'][num]=-1
						else:
							print("-->Sigma did not converged")
				except:
					pass
			
			elif params["type_convergence"]=="Gelman-Rubin":
				raise NameError("Not implemented yet.")
			else:
				raise NameError("There is not.")

		if nconvergence==len(params['convergence_test_parameter']):
			print("Converged system.")
			break
				
		count+=1
	
	print("Chains created.")
	print("=================================================================")
	print("\n\n")
	print("=================================================================")
	print("Outputs...")
	import matplotlib
	import matplotlib.pyplot as plt
	from matplotlib import rc
	import corner
	rc('text', usetex=True)

	pathbin = auxf.outputs_path(n_bins)
		
	mylabels=["A", r"$\alpha$", r"$\Sigma$"]
	fig=corner.corner(flat_sample, labels=mylabels, range=prior_range ,truths=a_fid, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
	plt.savefig(os.path.join(pathbin,'triangle'+str(bin_analysis)+'.png'))
	plt.show()
	
	try:
		plt.plot(autocorr0,label="autocorrelation - (A)mplitude")
	except:
		pass
	try:
		plt.plot(autocorr1,label="autocorrelation - alpha")
	except:
		pass
	try:
		plt.plot(autocorr2,label="autocorrelation - sigma")
	except:
		pass
	plt.plot(N_50,    label="N/"+str(int(factor_N)))
	plt.xlabel("number of samples, $N$")
	plt.ylabel(r"$\tau$ estimates")
	plt.legend(loc="best")
	plt.savefig(os.path.join(pathbin,'autocorrelation_evolution'+str(bin_analysis)+'.png'))
	plt.show()
	print("End program.")
	print("=================================================================")
