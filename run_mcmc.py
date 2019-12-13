import os,sys
import json
import numpy as np
import argparse
import mcmc_convergence_test as mcmc
import auxiliary_functions as auxf
from time import time,strftime, gmtime

###################################################################
# Check the python version and import configparser
###################################################################

if sys.version_info[0]==2:
	import ConfigParser
	config = ConfigParser.RawConfigParser()
elif sys.version_info[0]==3:
	import configparser
	config = configparser.ConfigParser()

###################################################################
# This part is for extracting information from mcmc_parameters.ini file
###################################################################
timei       = time()
INI         = "mcmc_parameters.ini"
name_params = os.path.join(os.getcwd(),INI)
config.read(name_params)

#Fitting
A0          = config.getfloat("Fitting","A0")
alpha0      = config.getfloat("Fitting","alpha0")
Silk0       = config.getfloat("Fitting","Silk0")

try:
	A_range     = json.loads(config.get("Fitting", "A_range"))
	alpha_range = json.loads(config.get("Fitting", "alpha_range"))
	Silk_range  = json.loads(config.get("Fitting", "Silk_range"))
	prior_range = np.array([np.array([A_range[0],A_range[1]]),np.array([alpha_range[0],alpha_range[1]]),np.array([Silk_range[0],Silk_range[1]])])
except:
	raise NameError
	
#MCMC
nwalkers               = config.getint("MCMC","nwalkers")
nsteps                 = config.getint("MCMC","nsteps")
steps                  = config.getint("MCMC","steps")
burn_steps             = config.getint("MCMC","burn_steps")
steps_convergence_test = config.getint("MCMC","steps_convergence_test")

#Data_information
rs                 = config.getfloat("Data_information","rs")
nbins              = config.getint(  "Data_information","nbins")
bin_analysis       = config.getint(  "Data_information","bin_analysis")
path_datas         = config.get(     "Data_information","path_datas")
filename_data      = config.get(     "Data_information","filename_data")
filename_redshifts = config.get(     "Data_information","filename_redshifts")
filename_error     = config.get(     "Data_information","filename_error").split(",")
try:
	use_range_l    = json.loads(config.get("Data_information", "use_range_l"))
except:
	raise NameError
data_distortion    = config.getboolean("Data_information","data_distortion")

#Convergence
type_convergence           = config.get("Convergence","type_convergence")
convergence_test_parameter = config.get("Convergence","convergence_test_parameter").split(",")
convergence_test_parameter = auxf.remove_space_input(convergence_test_parameter)
factor_N                   = config.getint("Convergence","factor_N")

#Output
chains_txt                     = config.getboolean("Outputs","chains_txt")
chains_convergence_plot_save   = config.getboolean("Outputs","chains_convergence_plot_save")
chains_convergence_plot_show   = config.getboolean("Outputs","chains_convergence_plot_show")
bidimension_triangle_plot_save = config.getboolean("Outputs","bidimension_triangle_plot_save")
bidimension_triangle_plot_show = config.getboolean("Outputs","bidimension_triangle_plot_show")
###############################################################################
# You can modify any options in the parameters.ini file by the command terminal
###############################################################################

parser = argparse.ArgumentParser(description='Modify by the command terminal parameters in mcmc_parameters.ini file')

#Fitting
parser.add_argument('--A0'    , action = 'store', dest = 'A0'     , default = A0,     help = 'Amplitude')
parser.add_argument('--alpha0', action = 'store', dest = 'alpha0' , default = alpha0, help = 'dilation term')
parser.add_argument('--Silk0' , action = 'store', dest = 'Silk0'  , default = Silk0,  help = 'decay term')

#MCMC
parser.add_argument('--nwalkers'              , action = 'store', dest = 'nwalkers'              , default = nwalkers              , help = 'number of walkers')
parser.add_argument('--nsteps'                , action = 'store', dest = 'nsteps'                , default = nsteps                , help = 'number of steps')
parser.add_argument('--steps'                 , action = 'store', dest = 'steps'                 , default = steps                 , help = 'number of steps per iteration')
parser.add_argument('--burn_steps'            , action = 'store', dest = 'burn_steps'            , default = burn_steps            , help = 'discarded steps')
parser.add_argument('--steps_convergence_test', action = 'store', dest = 'steps_convergence_test', default = steps_convergence_test, help = 'steps to convengence test')

#Data_information
parser.add_argument('--rs'                 , action = 'store', dest = 'rs'                 , default = rs                 , help = 'sound horizon')
parser.add_argument('--nbins'              , action = 'store', dest = 'nbins'              , default = nbins              , help = 'number of bins')
parser.add_argument('--bin_analysis'       , action = 'store', dest = 'bin_analysis'       , default = bin_analysis       , help = 'bin to analysis')
parser.add_argument('--path_datas'         , action = 'store', dest = 'path_datas'         , default = path_datas         , help = '')
parser.add_argument('--filename_data'      , action = 'store', dest = 'filename_data'      , default = filename_data      , help = '')
parser.add_argument('--filename_error'     , action = 'store', dest = 'filename_error'     , default = filename_error     , help = '')
parser.add_argument('--filename_redshifts' , action = 'store', dest = 'filename_redshifts' , default = filename_redshifts , help = '')
#parser.add_argument('--use_range_l'        , action = 'store', dest = 'use_range_l'        , default = use_range_l        , help = '')
parser.add_argument('--data_distortion'    , action = 'store', dest = 'data_distortion'    , default = data_distortion    , help = '')

#Convergence
parser.add_argument('--type_convergence'           , action = 'store', dest = 'type_convergence'           , default = type_convergence           , help = '')
#parser.add_argument('--convergence_test_parameter' , action = 'store', dest = 'convergence_test_parameter' , default = convergence_test_parameter , help = '')
parser.add_argument('--factor_N'                   , action = 'store', dest = 'factor_N'                   , default = factor_N                   , help = '')

#Output
parser.add_argument('--chains_txt'                     , action = 'store', dest = 'chains_txt'                    , default = chains_txt                     , help = '')
parser.add_argument('--chains_convergence_plot_save'   , action = 'store', dest = 'chains_convergence_plot_save'  , default = chains_convergence_plot_save   , help = '')
parser.add_argument('--chains_convergence_plot_show'   , action = 'store', dest = 'chains_convergence_plot_show'  , default = chains_convergence_plot_show   , help = '')
parser.add_argument('--bidimension_triangle_plot_save' , action = 'store', dest = 'bidimension_triangle_plot_save', default = bidimension_triangle_plot_save , help = '')
parser.add_argument('--bidimension_triangle_plot_show' , action = 'store', dest = 'bidimension_triangle_plot_show', default = bidimension_triangle_plot_show , help = '')


arguments = parser.parse_args()


###############################################################################
# Variables
###############################################################################
#Fitting
A0     = float(arguments.A0)
alpha0 = float(arguments.alpha0)
Silk0  = float(arguments.Silk0)

#MCMC
nwalkers               = int(arguments.nwalkers)
nsteps                 = int(arguments.nsteps)
steps                  = int(arguments.steps)
burn_steps             = int(arguments.burn_steps)
steps_convergence_test = int(arguments.steps_convergence_test)

#Data_information
rs                 = float(arguments.rs)
nbins              = int(arguments.nbins)
bin_analysis       = int(arguments.bin_analysis)
path_datas         = str(arguments.path_datas)
filename_data      = str(arguments.filename_data)
filename_redshifts = str(arguments.filename_redshifts)
filename_error     = str(arguments.filename_error)
#use_range_l = list(arguments.use_range_l)
data_distortion    = bool(arguments.data_distortion)

#Convegence
type_convergence           = str(arguments.type_convergence)
#convergence_test_parameter = list(arguments.convergence_test_parameter)
factor_N                   = int(arguments.factor_N)

#Output
chains_txt                     = bool(arguments.chains_txt)
chains_convergence_plot_save   = bool(arguments.chains_convergence_plot_save)
chains_convergence_plot_show   = bool(arguments.chains_convergence_plot_show)
bidimension_triangle_plot_save = bool(arguments.bidimension_triangle_plot_save)
bidimension_triangle_plot_show = bool(arguments.bidimension_triangle_plot_show)

use_range_l                = np.asarray(use_range_l).astype(int)
convergence_test_parameter = np.asarray(convergence_test_parameter).astype(int)
###############################################################################
# Building "params" 
###############################################################################

params = {"A0":A0,"alpha0":alpha0,"Silk0":Silk0,
          "prior_range":prior_range,
          "nwalkers":nwalkers, "nsteps":nsteps,"steps":steps, "burn_steps":burn_steps,
          "steps_convergence_test":steps_convergence_test,
          "rs":rs,"nbins":nbins,"bin_analysis":bin_analysis,
          "type_convergence":type_convergence,
          "use_range_l":use_range_l,"convergence_test_parameter":convergence_test_parameter,
          "factor_N":factor_N, "data_distortion":data_distortion
          }
paramsOut ={"path_datas":path_datas, "filename_data":filename_data,"filename_redshifts":filename_redshifts, "filename_error":filename_error,
            "chains_txt":chains_txt,
            "chains_convergence_plot_save":chains_convergence_plot_save,"chains_convergence_plot_show":chains_convergence_plot_show,
            "bidimension_triangle_plot_save":bidimension_triangle_plot_save,
            "bidimension_triangle_plot_show":bidimension_triangle_plot_show
            }

###############################################################################
# Running MCMC
###############################################################################
mcmc.run_emcee(params,paramsOut,prior_range)
