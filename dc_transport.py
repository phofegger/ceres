import numpy as np
import re, os, glob, configparser
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# read settings from config file
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config.ini')

rxx = map(int, config['Setup']['rxx'].replace(' ','').split(','))
ryx = map(int, config['Setup']['ryx'].replace(' ','').split(','))
show = config['Analyzer'].getboolean('show')
work_folder = '../ppms_import/first/'

# analyze IV curves to find best excitation current (can be temperature dependent), diff B in diff files, also applies to ryx
# TODO if no file just allow all
for file in glob.glob(work_folder+'*IV*'):
	func = lambda x, c0: c0
	with open(file) as fh:
		names = fh.next().split(',')
		comment = fh.next()[2:].split(',')

	# TODO make function to shorten the index finding
	regex = re.compile(".*(Temp).*")
	ind_t = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	regex = re.compile(".*(Bridge %d E).*"%(rxx[0]))
	ind_e = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	regex = re.compile(".*(Bridge %d )(S|R).*"%(rxx[0]))
	ind_r = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]

	data = np.genfromtxt(file, delimiter=',')

	temp, exc, res, resStd = [np.zeros((len(ind_e))) for i in range(4)]
	minE = 3 # min excitation
	for i in range(len(ind_e)):
		indExc = ind_e[i]
		indTmp = ind_t[i]
		indRes = ind_r[2*i]
		indRst = ind_r[2*i+1]

		mask = ~np.isnan(data[:,indExc])
		mdata = data[mask,:]
		tmp = np.mean(mdata[:,indTmp])
		if len(glob.glob(work_folder+'*IV*')) == 1:
			arrE, arrR, arrS = [np.zeros((mdata.shape[0]-minE)) for z in range(3)]
		else:
			pass # if IV curves at diff B, dont think so
		for l in range(minE, len(arrE)):			
			popt, pcov = curve_fit(func, mdata[:l,indExc], mdata[:l,indRes], sigma=mdata[:l,indRst])
			pstd = np.diag(pcov)**0.5
			arrE[l], arrR[l], arrS[l] = mdata[minE+l,indExc], popt[0], pstd[0]

		avg_std = np.mean(arrS)
		for l in range(len(arrE)-1, minE-1, -1):
			if arrS[l] < 1.5*avg_std:
				temp[i] = tmp
				exc[i] = arrE[l]
				res[i] = arrR[l]
				resStd[i] = arrS[l]
				break
	if show:
		plt.plot(temp, res)
		plt.show()
		plt.plot(temp, exc)
		plt.xlim([np.min(temp), np.max(temp)])
		plt.ylim([0,np.max(exc)*1.1])
		plt.show()

# analyze Hall measurements
# TODO define labels such as Bscan, ....
Bfiles = glob.glob(work_folder+'*Bscan*')
maxB = config['Analyzer'].getfloat('maxB')
if len(Bfiles) != 1:
	pass # somehow combine them so i can do it in one go
elif len(Bfiles) == 0:
	pass # no Bsweeps
else:
	file = Bfiles[0]

	with open(file) as fh:
		names = fh.next().split(',')
		comment = fh.next()[2:].split(',')

	regex = re.compile(".*(Temp).*")
	ind_t = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	regex = re.compile(".*(Magnetic).*")
	ind_b = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	regex = re.compile(".*(Bridge %d E).*"%(ryx[0]))
	ind_e = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	regex = re.compile(".*(Bridge %d )(S|R).*"%(ryx[0]))
	ind_r = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	data = np.genfromtxt(file, delimiter=',')

	# TODO find better names :)
	c1_arr, c1s_arr, c1t_arr = [np.zeros((len(ind_e))) for z in range(3)]
	for i in range(len(ind_e)):
		indExc = ind_e[i]
		indTmp = ind_t[i]
		indB = ind_b[i]
		indRes, indRst = ind_r[2*i], ind_r[2*i+1]

		mask = ~np.isnan(data[:,indExc])
		mdata = data[mask,:]
		# TODO make function probably
		max_ind = np.argmin(np.abs(np.mean(mdata[:,indTmp])-temp))
		mask = mdata[:,indExc] <= exc[max_ind]
		mmdata = mdata[mask,:]

		s_ind = np.argmin(np.abs(mmdata[:,indB]-maxB))
		e_ind = np.argmin(np.abs(mmdata[:,indB]+maxB))

		# TODO move to better location
		func = lambda x, c0, c1: c0+c1*x
		popt, pcov = curve_fit(func, mmdata[s_ind:e_ind,indB], mmdata[s_ind:e_ind,indRes], sigma=mmdata[s_ind:e_ind,indRst])
		pstd = np.diag(pcov)**0.5

		c1t_arr[i] = np.mean(mmdata[:,indTmp])
		c1_arr[i] = popt[1]
		c1s_arr[i] = pstd[1]

	plt.plot(c1t_arr, 1/(c1_arr*1.6022e-19)*1e-6)
	plt.show()
