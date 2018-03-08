import numpy as np
import re, os, glob, configparser
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# read settings from config file
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config.ini')

q = config['Analyzer'].getfloat('q')
rxx = map(int, config['Setup']['rxx'].replace(' ','').split(','))
ryx = map(int, config['Setup']['ryx'].replace(' ','').split(','))
show = config['Analyzer'].getboolean('show')
work_folder = '../ppms_import/first/' # TODO folder?

# helper functions
def getAllIndex(col_names, label):
	regex = re.compile('.*'+label+'.*')
	return [i for i, l in enumerate(col_names) for m in [regex.search(l)] if m]

def maskCurrent(data, indExc, indTmp):
	if restrictCurr:
		max_ind = np.argmin(np.abs(np.mean(data[:,indTmp])-temp))
		return data[data[:,indExc] <= exc[max_ind],:]
	else:
		return data

# TODO make plots better? low priority
# analyze IV curves to find best excitation current (can be temperature dependent), diff B in diff files, also applies to ryx
restrictCurr = False
for file in glob.glob(work_folder+'*IV*'):
	func = lambda x, c0: c0
	with open(file) as fh:
		fh.next() # TODO maybe needed
		names = fh.next().split(',')
		comment = fh.next()[2:].split(',')

	# get right column indices
	ind_t = getAllIndex(names, '(Temp)')
	ind_e = getAllIndex(names, '(Bridge %d E)'%(rxx[0]))
	ind_r = getAllIndex(names, '(Bridge %d )(S|R)'%(rxx[0]))

	data = np.genfromtxt(file, delimiter=',')
	temp, exc, res, resStd = [np.zeros((len(ind_e))) for i in range(4)] # TODO give better names to avoid replacing
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
			pass # if IV curves at diff B, dont think so TODO
		for l in range(minE, len(arrE)):			
			popt, pcov = curve_fit(func, mdata[:l,indExc], mdata[:l,indRes], sigma=mdata[:l,indRst])
			pstd = np.diag(pcov)**0.5
			arrE[l], arrR[l], arrS[l] = mdata[minE+l,indExc], popt[0], pstd[0]

		avg_std = np.mean(arrS)
		for l in range(len(arrE)-1, minE-1, -1):
			if arrS[l] < 1.5*avg_std: 		# TODO integrate into config?
				temp[i] = tmp
				exc[i] = arrE[l]
				res[i] = arrR[l]
				resStd[i] = arrS[l]
				break

	restrictCurr = True
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
		fh.next() # TODO maybe needed
		names = fh.next().split(',')
		comment = fh.next()[2:].split(',')

	# get right column indices
	ind_t = getAllIndex(names, '(Temp)')
	ind_b = getAllIndex(names, '(Magnetic)')
	ind_e = getAllIndex(names, '(Bridge %d E)'%(ryx[0]))
	ind_r = getAllIndex(names, '(Bridge %d )(S|R)'%(ryx[0]))
	
	data = np.genfromtxt(file, delimiter=',')
	func = lambda x, c0, c1: c0+c1*x
	tmp, c0, c1, c0_std, c1_std, rsq = [np.zeros((len(ind_e))) for z in range(6)]
	for i in range(len(ind_e)):
		indTmp, indB, indExc = ind_t[i], ind_b[i], ind_e[i]
		indRes, indRst = ind_r[2*i], ind_r[2*i+1]

		mdata = data[~np.isnan(data[:,indExc]),:]
		mdata = maskCurrent(mdata, indExc, indTmp) # restrict to only allowed values

		s_ind = np.argmin(np.abs(mdata[:,indB]-maxB))
		e_ind = np.argmin(np.abs(mdata[:,indB]+maxB))

		popt, pcov = curve_fit(func, mdata[s_ind:e_ind,indB], mdata[s_ind:e_ind,indRes], sigma=mdata[s_ind:e_ind,indRst])
		pstd = np.diag(pcov)**0.5
		tmp[i] = np.mean(mdata[:,indTmp])
		c0[i], c0_std[i] = popt[0], pstd[0]
		c1[i], c1_std[i] = popt[1], pstd[1]

		yh = func(mdata[s_ind:e_ind,indB],*popt)
		yi = mdata[s_ind:e_ind,indRes]
		ybar = np.mean(yi)/len(yi)
		rsq[i] = np.sum((yh-ybar)**2)/np.sum((yi-ybar)**2) # R^2 value
	
	c0_to_max = np.abs(c0/np.nanmax(data[:,[ind_r[2*l] for l in range(len(ind_e))]], axis=0))
	print 'Mean Hall offset ratio c0/max(ryx)=%.3f'%(np.mean(c0_to_max))

	n = 1./(c1*q)*1e-6
	dn = (-1./(q*c1**2))*c1_std*1e-6
	if np.mean(n) < 0.0:
		print 'Carrier type: hole'
		n *= -1
	else:
		print 'Carrier type: electron'

	# use iv-res to calculate muh
	print tmp.shape, temp.shape # TODO IV counts and bcounts are different so damn

	if show:
		plt.plot(tmp, c1)
		plt.show()
		plt.errorbar(tmp, n, yerr=dn, fmt='o')
		plt.show()


	header = 'Temperature (K),Hall Coefficient (1/m^2),Hall Coefficient Std (1/m^2),Carrier Density (1/cm^3),Carrier Density Std (1/cm^3)'
	#np.savetxt('final.data', np.vstack((tmp, c1, c1_std, n, dn)).T, header=header, delimiter=',')

"""
# analyze resistivity
Tfiles = sorted(glob.glob(work_folder+'*Tscan*'), key=lambda x: float(x[x.rfind('/')+1:x.rfind('T_')]))
if len(Tfiles) == 0:		# TODO do nothing lel
	pass
elif len(Tfiles) == 1:		# all in one file
	pass # TODO
else:
	for x in Tfiles:
		with open(x) as fh:
			loop = fh.next().split(' ')[1:]
			var, start, end, steps, mode, lc = [fun(loop[i]) for i, fun in enumerate([str, float, float, int, int , int])]
			names = fh.next()[2:].split(',')

		data = np.genfromtxt(x, comments='#', delimiter=',')
		indT = getAllIndex(names, '(Temperature)')[0]
		indB = getAllIndex(names, '(Magnetic)')[0]
		indR = getAllIndex(names, '(Bridge %d R)'%(rxx[0]))[0]
		indE = getAllIndex(names, '(Bridge %d E)'%(rxx[0]))[0]
		mdata = data[~np.isnan(data[:,indE]),:]
		exc *= 0.7 # cause very temperature sensitive at low K see curve splitting TODO maybe show plot to better illustrate point
		mdata = maskCurrent(mdata, indE, indT)
		exc /= 0.7
		#plt.plot(mdata[:,indT], mdata[:,indR], '.', label='%fT'%(np.nanmean(data[:,indB])))

		if np.abs(np.mean(mdata[:,indB])) < 0.01:
			print "We no field baby"
			plt.plot(mdata[:,indT], mdata[:,indR], '.', label='%fT'%(np.nanmean(data[:,indB])))
			

		plt.legend()
		plt.show()
	#data = np.hstack([np.genfromtxt(x, comments='#', delimiter=',') for x in Tfiles])
"""