import numpy as np
import re, os, glob, configparser
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from sklearn import linear_model

# read settings from config file
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config.ini')

rxx = map(int, config['Setup']['rxx'].replace(' ','').split(','))
ryx = map(int, config['Setup']['ryx'].replace(' ','').split(','))
work_folder = os.path.split(config['Files']['data'])[0] + '/' + config['Files']['title'] + '/'
unit = config['Options']['unit']
show = config['Options'].getboolean('show')
symmetrize = config['Options'].getboolean('symmetrize')

q = config['Analyzer'].getfloat('q')
min_exc = config['Analyzer'].getfloat('min_exc')
max_exc = config['Analyzer'].getfloat('max_exc')
restrict_exc = config['Analyzer'].getboolean('restrict_exc')

# helper functions
def getAllIndex(col_names, label):
	regex = re.compile('.*'+label+'.*')
	return [i for i, l in enumerate(col_names) for m in [regex.search(l)] if m]

def maskCurrent(data, indExc, indTmp):
	if restrict_exc:
		max_ind = np.argmin(np.abs(np.mean(data[:,indTmp])-temp))
		max_e = np.max([exc[max_ind], max_exc])
		return data[(data[:,indExc] <= max_e) & (data[:,indExc] >= min_exc),:]
	else:
		return data

def fit_ransac(datax, datay, function, datay_err=None):
	ransac = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(fit_intercept=True))	

	# find outlier
	score = -np.inf	
	for lll in range(10):		
		ransac.fit(datax.reshape(-1,1), datay.reshape(-1,1))
		nscore = ransac.score(datax.reshape(-1,1), datay.reshape(-1,1)) 
		if nscore > score:
			pfit = [ransac.estimator_.intercept_[0], ransac.estimator_.coef_[0][0]]
			inlier_mask = ransac.inlier_mask_
			outlier_mask = np.logical_not(inlier_mask)
			score = nscore
	
	# calculate fit parameter and error
	popt, pcov = curve_fit(function, datax[inlier_mask], datay[inlier_mask], sigma=datay_err[inlier_mask])
	return popt, np.sqrt(np.diag(pcov)), inlier_mask, outlier_mask 
	
# analyze IV curves to find best excitation current for each temperature
units = {'Ohm': 1, 'Ohm*m': 1, 'Ohm*cm': 1e-2, 'mOhm*cm': 1e-5}
for file in glob.glob(work_folder+'*IV*'):
	func = lambda x, c0: c0+x*0
	with open(file) as fh:
		fh.next() # skip sweep parameter
		names = fh.next().split(',')
		comment = fh.next()[2:].split(',')

	# get right column indices
	ind_t = getAllIndex(names, '(Temp)')
	ind_e = getAllIndex(names, '(Bridge %d E)'%(rxx[0]))
	ind_r = getAllIndex(names, '(Bridge %d )(S|R)'%(rxx[0]))

	data = np.genfromtxt(file, delimiter=',')
	data[:,ind_r] *= units[unit]	# convert back to regular SI unit
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
		#if show:
		#	plt.plot(arrE[minE:],arrR[minE:],'ro')
		#	plt.show()

		avg_std = np.mean(arrS[minE:])
		for l in range(len(arrE)-1, minE-1, -1):
			if arrS[l] < 1.5*avg_std:
				temp[i] = tmp
				exc[i] = arrE[l]
				res[i] = arrR[l]
				resStd[i] = arrS[l]
				break
	
	if show:
		fig, ax = plt.subplots()
		ax.plot(temp, exc, 'o')
		ax.set_xlim([np.min(temp), np.max(temp)])
		ax.set_ylim([0,np.max(exc)*1.1])
		ax.set_title('Maximum allowed excitation current')
		ax.set_xlabel('Temperature (K)')
		ax.set_ylabel(r'max. excitation current ($\mu A$)')
		plt.show()

# analyze Hall measurements
Bfiles = glob.glob(work_folder+'*Bscan*')
bRange = map(float, config['Analyzer']['bRange'].split(','))

if len(Bfiles) == 0:
	print "No Bsweeps!"
elif len(Bfiles) != 1:
	pass # calculate for each sweep and combine at the end
else:
	file = Bfiles[0]

	with open(file) as fh:
		fh.next() # TODO maybe needed
		names = fh.next().split(',')
		comment = fh.next()[2:].split(',')
	data = np.genfromtxt(file, delimiter=',')

	# get right column indices
	ind_t = getAllIndex(names, '(Temp)')
	ind_b = getAllIndex(names, '(Magnetic)')
	ind_e = getAllIndex(names, '(Bridge %d E)'%(ryx[0]))
	if symmetrize:
		ind_r = getAllIndex(names, '(Bridge %d )(Std\. Dev\. a|Resistivity a)'%(ryx[0]))
	else:
		ind_r = getAllIndex(names, '(Bridge %d )(S|R)'%(ryx[0]))
	
	fun = lambda x, a, b: a+b*x
	tmp, c0, c1, c0_std, c1_std = [np.zeros((len(ind_e))) for z in range(5)]
	for i in range(len(ind_e)):
		# index variables
		indTmp, indB, indExc = ind_t[i], ind_b[i], ind_e[i]
		indRes, indRst = ind_r[2*i], ind_r[2*i+1]

		# restrict to only allowed values
		mdata = data[~np.isnan(data[:,indExc]),:]
		mdata = maskCurrent(mdata, indExc, indTmp)

		# restrict magnetic field range
		s_ind = np.argmin(np.abs(mdata[:,indB]-bRange[0]))
		e_ind = np.argmin(np.abs(mdata[:,indB]-bRange[1]))
		if s_ind > e_ind:
			s_ind, e_ind = e_ind, s_ind
		mdata = mdata[s_ind:e_ind,:]
		
		# find outlier and fit linear function
		fit_data = fit_ransac(mdata[:,indB], mdata[:,indRes], fun, mdata[:,indRst])
		pfit, perr, inlier_mask, outlier_mask = fit_data
		
		tmp[i] = np.mean(mdata[:,indTmp])
		c0[i], c0_std[i] = pfit[0], perr[0]
		c1[i], c1_std[i] = pfit[1], perr[1]
		
		if show:
			plt.errorbar(mdata[inlier_mask,indB], mdata[inlier_mask,indRes], yerr=mdata[inlier_mask,indRst],
						fmt='o', ms=10, alpha=0.5, c='yellowgreen', label='Inliers')
			plt.errorbar(mdata[outlier_mask,indB], mdata[outlier_mask,indRes], yerr=mdata[outlier_mask,indRst],
						fmt='o', ms=10, alpha=0.5, c='gold', label='Outliers')
			plt.scatter(mdata[:,indB], fun(mdata[:,indB], pfit[0], pfit[1]),
						c='violet', s=50, label='fit')			
			plt.title("%f"%(tmp[i]))
			plt.ylim([np.min(mdata[:,indRes]), np.max(mdata[:,indRes])])
			plt.legend()
			plt.show()
	
	c0_to_max = np.abs(c0/np.nanmax(data[:,[ind_r[2*l] for l in range(len(ind_e))]], axis=0))
	print 'Mean Hall offset ratio c0/max(ryx)=%.3f'%(np.mean(c0_to_max))

	n = 1./(c1*q)
	dn = np.abs((-1./(q*c1**2))*c1_std)
	carrier_type = 'electron' if np.mean(n) < 0.0 else 'hole'
	print 'Carrier type: ' + carrier_type
	if np.mean(n) < 0.0:
		n *= -1

	# use iv-res to calculate muh
	int_res, int_res_std = [np.zeros((len(tmp))) for z in range(2)]
	int_res[1:-1] = interp1d(temp, res, kind='linear')(tmp[1:-1])
	int_res_std[1:-1] = interp1d(temp, resStd, kind='linear')(tmp[1:-1])
	int_res[0], int_res[-1] = res[-1], res[0]
	int_res_std[0], int_res_std[-1] = resStd[-1], resStd[0]

	muh = np.abs(c1/int_res)
	dmuh = np.sqrt(((1./int_res)**2*c1_std**2+(+c1/int_res**2)**2*int_res_std**2))
	
	header = 'Hall data for single carrier (type=%s)'%(carrier_type)
	header += '\nTemperature (K),Resistivity (mOhm*cm),Resistivity Std (mOhm*cm),Hall Coefficient (cm^3/C),Hall Coefficient Std (cm^3/C)'
	header += ',Carrier Density (1/cm^3),Carrier Density Std (1/cm^3),Hall Mobility (cm^2/V*s),Hall Mobility Std (cm^2/V*s)'
	header += '\n'
	np.savetxt(work_folder + 'final.dat', np.vstack((tmp, int_res*1e5, int_res_std*1e5, c1*1e6, c1_std*1e6, n*1e-6, dn*1e-6, muh*1e4, dmuh*1e4)).T, header=header, delimiter=',')

	if show:
		plt.plot(tmp, c1)
		plt.show()
		plt.errorbar(tmp, n, yerr=dn, fmt='o')
		plt.show()

# analyze resistivity # TODO
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
		exc *= 0.6 # cause very temperature sensitive at low K see curve splitting TODO maybe show plot to better illustrate point
		mdata = maskCurrent(mdata, indE, indT)
		exc /= 0.6
		#plt.plot(mdata[:,indT], mdata[:,indR], '.', label='%fT'%(np.nanmean(data[:,indB])))

		#if np.abs(np.mean(mdata[:,indB])) < 0.01:
		#plt.plot(mdata[:,indT], mdata[:,indR], '.', label='%fT'%(np.nanmean(data[:,indB])))
			

	#plt.legend()
	#plt.show()
	#data = np.hstack([np.genfromtxt(x, comments='#', delimiter=',') for x in Tfiles])
