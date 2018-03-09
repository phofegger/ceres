import numpy as np
import re, os, fileinput
import configparser

# read settings from config file
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config.ini')
path_data = config['Files']['data']
path_seq = config['Files']['sequence']
add_cols = config['Files']['ignore_columns'].split(', ')

strip_cols = config['Options'].getboolean('strip_cols')
combine_loop = config['Options'].getboolean('combine_loops')
remove_outlier = config['Options'].getboolean('remove_outlier')
reverse_polarity = ~config['Setup'].getboolean('pol')

skip_data = config['Files'].getint('skip_data')
max_nest_depth = config['Options'].getint('max_depth')
Rxx = map(int, config['Setup']['rxx'].replace(' ','').split(','))
Ryx = map(int, config['Setup']['ryx'].replace(' ','').split(','))

distance = config['Setup'].getfloat('d')*1e-3
cross_section = config['Setup'].getfloat('A')*1e-6
outlier_perc = config['Options'].getfloat('outlier_perc')
T = config['Options'].getfloat('T')
B = config['Options'].getfloat('B')

# determine data file structure TODO make uniform labels for scans
start_data = 0
with open(path_data, 'r') as file:
	for line in file:
		start_data += 1
		if '[Data]' in line:
			names = np.array(file.next().replace('Ohm','Ohm*m').split(','))
			cols = len(file.next().split(','))
			break

data = np.genfromtxt(path_data, skip_header=start_data + skip_data + 1, delimiter=',')

# remove unused columns if wanted
if strip_cols:
	mask = ~np.isnan(data[0])
	for add_col in add_cols:
		for i, name in enumerate(names):
			if add_col in name:
				mask[i] = False

	names = names[:cols][mask]
	data = data[:,mask]
	cols = data.shape[1]

# convert values
data[:,np.where(names=='Magnetic Field (Oe)')] /= 1e+4
names[np.where(names=='Magnetic Field (Oe)')] = 'Magnetic Field (T)'
for r in Rxx:
	regex = re.compile(".*(Bridge %d )(S|R).*"%(r))
	ind = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	if len(ind) == 0:
		raise UserWarning("Invalid channel setup")
	data[:,ind] *= cross_section/distance
for r in Ryx:
	regex = re.compile(".*(Bridge %d )(S|R).*"%(r))
	ind = [i for i, l in enumerate(names) for m in [regex.search(l)] if m]
	if len(ind) == 0:
		raise UserWarning("Invalid channel setup")
	if reverse_polarity:
		data[:,ind] *= -1.

# remove the outliers using IQR method due to big absolute value difference, relative error would be better f.e. big MR values
if remove_outlier:
	tot = 0
	for i, name in enumerate(names):
		if 'Std.' in name:
			q1,q2 = np.nanpercentile(data[:,i], [0,outlier_perc])
			outlier = ~np.isnan(data[:,i])
			outlier[outlier] &= data[outlier,i] > q2*1.5
			data[outlier, :] = np.nan
			avg, std = np.nanmean(data[:,i]), np.nanstd(data[:,i])
			print avg,std
			tot += np.sum(outlier)
	print "Removed %d outliers from %d data points"%(tot, data.shape[0])

# analyze sequence file
ind = 0
def search_loop(fo, depth):
	count = 0
	comment = ""
	containsloop = False
	global T, B, ind, data_folder
	while(True):
		line = fo.readline()

		if line == "":
			print "reached EOF with ", count
			return count
		com = line[:3]

		if com in ['ENT', 'ENB']:
			return count, containsloop

		elif com == 'REM':
			if combine_loop and depth == max_nest_depth:
				comment = (line[4:]).replace('-','').replace(' ','').replace('\n','').replace('\r','')

		elif com in ['LPI']:
			lc = int(line.split(' ')[13])
			containsloop = True
			if combine_loop:
				count += lc
				if depth == max_nest_depth:
					print "make IV file with size %d and T=%f and B=%f"%(count, T, B)
					ind += count
					count = 0
			else:
				if depth > max_nest_depth:
					print "make IV file with size %d and T=%f and B=%f"%(count, T, B)
					ind += count
				else:
					count += lc

		elif com in ['LPT', 'LPB']:
			var, fun = {'T': [B,'T'], 'B': [T,'K']}, {0: np.linspace, 2: lambda x,y,z: np.logspace(np.log10(x), np.log10(y), z)}
			start, end, steps, mode = [func(line.split(' ')[i]) for func,i in [[float,2],[float,3],[int,5],[int,-2]]]
			points = fun[mode](start, end, steps) 
			lc, cl = search_loop(fo, depth+1)
			lc *= steps
			containsloop = True
			if combine_loop:
				count += lc
				if depth == max_nest_depth:
					cdata = data[ind:ind+count]
					if cl:
						cdata = np.hstack(np.vsplit(cdata, steps))
						commentline = '%s %f %f %d %d %d'%(comment, start, end, steps, mode, lc/steps)
						commentline += '\n'+','.join([item for sublist in [names]*steps for item in sublist])
						commentline += '\n'+','.join([item for sublist in [['%.1f%s'%(x, {'T': 'K', 'B': 'T'}[com[2]])]*cols for x in points] for item in sublist])
					else:
						commentline = '%s %f %f %d %d %d'%(comment, start, end, steps, mode, lc/steps)
						commentline += '\n'+','.join(names)
						commentline += '\n'
					filename = '%s/%s.dat'%(data_folder,'%.1f%s_%s'%(var[com[2]][0],var[com[2]][1],comment))
					np.savetxt(filename, cdata, header=commentline, delimiter=',')
					for line in fileinput.input(filename, inplace=True):
						print re.sub('nan', '', line.rstrip())
					ind += count
					count = 0
			else:
				if depth > max_nest_depth:
					print "make file with size %d and %s=%f and name=%s"%(lc, 'T' if com[2] == 'B' else 'B', T if com[2] == 'B' else B, comment)
					ind += lc
				else:
					count += lc

		elif com == 'FLD':
			B = float(line.split(' ')[2])*1e-4

		elif com == 'TMP':
			T = float(line.split(' ')[2])

		elif com == 'RES':
			count += 1

		elif com in ['CHN', 'CDF', 'WAI']:
			pass

		else:
			print "unknown command"

data_folder = os.path.splitext(path_data)[0]
with open(path_seq, 'r') as seq_file:
	if not os.path.exists(data_folder):
		os.makedirs(data_folder)
	search_loop(seq_file, 0)

print data.shape, ind