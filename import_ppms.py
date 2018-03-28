import numpy as np
import re, os, fileinput
import configparser
import matplotlib.pyplot as plt

# read settings from config file
config = configparser.ConfigParser(inline_comment_prefixes='#')
config.read('config.ini')
path_data = config['Files']['data']
path_seq = config['Files']['sequence']
add_cols = config['Files']['ignore_columns'].split(', ')

strip_cols = config['Options'].getboolean('strip_cols')
combine_loop = config['Options'].getboolean('combine_loops')
remove_outlier = config['Options'].getboolean('remove_outlier')
symmetrize = config['Options'].getboolean('symmetrize')
calc_mr = config['Options'].getboolean('calc_mr')
reverse_polarity = not config['Setup'].getboolean('pol')
weighted_avg = config['Options'].getboolean('weighted_avg')
show = config['Options'].getboolean('show')
unit = config['Options']['unit']
units = {'Ohm*m': 1, 'Ohm*cm': 1e2, 'mOhm*cm': 1e5}
unit, conv_factor = (unit, units[unit]) if unit in units else ('Ohm*m', 1)

skip_data = config['Files'].getint('skip_data')
max_nest_depth = config['Options'].getint('max_depth')
Rxx = map(int, config['Setup']['rxx'].replace(' ','').split(','))
Ryx = map(int, config['Setup']['ryx'].replace(' ','').split(','))

distance = config['Setup'].getfloat('d')*1e-3
cross_section = config['Setup'].getfloat('A')*1e-6
outlier_perc = config['Options'].getfloat('outlier_perc')
T = config['Options'].getfloat('T')
B = config['Options'].getfloat('B')

# helper functions
def getAllIndex(col_names, label):
	regex = re.compile('.*'+label+'.*')
	return [i for i, l in enumerate(col_names) for m in [regex.search(l)] if m]

def saveData(comment, start, end, steps, mode, lc, cl, com):
	global T, B, data_folder, names, cols
	global symmetrize, calc_mr
	B, T = args['B'], args['T']

	var, fun = {'T': [B, 'T'], 'B': [T, 'K']}, {0: np.linspace, 2: lambda x,y,z: np.logspace(np.log10(x), np.log10(y), z)}
	points = fun[mode](start, end, steps)

	if comment == 'Bscan':
		pass
	commentline = '%s %f %f %d %d %d'%(comment, start, end, steps, mode, lc/steps)
	if cl:
		cdata = np.hstack(np.vsplit(cdata, steps))		
		commentline += '\n'+','.join([item for sublist in [names]*steps for item in sublist])
		commentline += '\n'+','.join([item for sublist in [['%.1f%s'%(x, {'T': 'K', 'B': 'T'}[com[2]])]*cols for x in points] for item in sublist])
	else:
		commentline += '\n'+','.join(names)
		commentline += '\n'
	filename = '%s/%s.dat'%(data_folder, '%.1f%s_%s'%(var[com[2]][0], var[com[2]][1], comment))
	np.savetxt(filename, cdata, header=commentline, delimiter=',')
	for line in fileinput.input(filename, inplace=True):
		print re.sub('nan', '', line.rstrip())

class TreeNode(list):
	def __init__(self, iterable=(), type='', comment='', args=[]):
		self.type = type
		self.comment = comment
		self.args = args
		list.__init__(self, iterable)

	def getDenominator(self):
		if len(self) != 0 and any(len(child) != 0 for child in self):
			return self[0].getDenominator()
		if len(self) != 0 and not any(len(child) != 0 for child in self):
			return np.array([child.args for child in self])

	def has_children(self):
		return True if len(self) > 0 else False

	def __repr__(self):
		if len(self) != 0:
			return '%s [' % (self.type) + ', '.join(str(item) for item in self) + ']'
		else:
			return '%s' % (self.type)
root = TreeNode(type='root')

# determine data file structure
start_data = 0
with open(path_data, 'r') as file:
	for line in file:
		start_data += 1
		if '[Data]' in line:
			names = np.array(file.next().split(','), dtype='|S48')
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
	ind = getAllIndex(names, '(Bridge %d )(S|R)'%(r))
	for i in ind:
		names[i] = names[i].replace('Ohm*m', 'Ohm').replace('Ohm', unit)
	if len(ind) == 0:
		raise UserWarning("Invalid channel setup")
	data[:,ind] *= conv_factor*cross_section/distance
for r in Ryx:
	ind = getAllIndex(names, '(Bridge %d )(R)'%(r))
	if len(ind) == 0:
		raise UserWarning("Invalid channel setup")
	for i in ind:
		names[i] = names[i].replace('Ohm*m', 'Ohm')
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
			#avg, std = np.nanmean(data[:,i]), np.nanstd(data[:,i])
			tot += np.sum(outlier)
	print "Removed %d outliers from %d data points"%(tot, data.shape[0])

# analyze sequence file
ind = 0
def search_loop(fo, depth):
	count = 0
	comment = ""
	containsloop = False
	loop_info = []
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
			args = [func(line.split(' ')[i]) for func,i in [[float,2],[float,3],[int,5],[int,-2]]]
			lc, cl = search_loop(fo, depth+1)
			lc *= steps
			containsloop = True
			if combine_loop:
				count += lc
				if depth == max_nest_depth:
					cdata = data[ind:ind+count]
					saveData(comment, start, end, steps, mode, lc, cl, com)
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

def make_tree(fo, node, comment=''):
	while(True):
		line = fo.readline()
		
		if line == '':
			return
		com = line[:3]

		if com in ['ENT', 'ENB']:
			return
		elif com == 'REM':
			tmp = (line[4:]).replace('-','').replace(' ','').replace('\n','').replace('\r','')
			comment = tmp if tmp != '' else comment
		elif com == 'LPI':
			exc_s, exc_e = map(float, line.split(' ')[1:3])
			counts = int(line.split(' ')[13])
			nnode = TreeNode(type=com, comment=comment, args=[exc_s, exc_e, 1, 0])
			for exc in np.linspace(exc_s, exc_e, counts):
				nnode.append(TreeNode(type='RES', args=exc))
			node.append(nnode)
		elif com in ['LPT', 'LPB']:
			nnode = TreeNode(type=com, comment=comment, args=[func(line.split(' ')[i]) for func,i in [[float,2], [float,3], [int,5], [int,-2]]])
			make_tree(fo, nnode, comment)
			node.append(nnode)
		elif com == 'FLD':
			node.append(TreeNode(type=com, args=float(line.split(' ')[2])*1e-4))
		elif com == 'TMP':
			node.append(TreeNode(type=com, args=float(line.split(' ')[2])))
		elif com == 'RES':
			node.append(TreeNode(type=com, args=float(line.split(' ')[4])))
		elif com in ['CHN', 'CDF', 'WAI']:
			pass
		else:
			print "Unknown command"

def calc_tree(node):
	count = 0
	if len(node) != 0:
		for child in node:
			if len(child) == 0:
				if child.type == 'RES':
					count += 1
			else:
				if 'LP' in child.type:
					child.size = calc_tree(child)
					count += child.size*child.args[2]
	return count

def save_tree_data(node, depth=0, pos=0):
	global B, T
	if len(node) == 0:
		return
	
	for child in node:
		if len(child) == 0:
			if child.type == 'FLD':
				B = child.args
			elif child.type == 'TMP':
				T = child.args
		elif depth <= max_nest_depth:
			if 'LP' in child.type:
				var, fun = {'T': '%.1fT'%(B), 'B': '%.1fK'%(T)}, {0: np.linspace, 2: lambda x,y,z: np.logspace(np.log10(x), np.log10(y), z)}
				start, end, steps, mode = child.args
				points = fun[mode](start, end, steps)

				add_names = []
				cdata = data[pos:pos+child.size*child.args[2]]
				filename = '%s/%s_%s.dat'%(data_folder, var[child.type[2]], child.comment)
				header = '%s %f %f %d %d %d'%(child.comment, start, end, steps, mode, child.size*steps/child.args[2])
				if any(len(cchild) != 0 for cchild in child):
					if 'Bscan' in child.comment:						
						exc = child.getDenominator()
						un_exc, un_ind = np.unique(exc, return_inverse=True)
						ind_t = getAllIndex(names, '(Temp)')
						ind_rxx = getAllIndex(names, '(Bridge %d )(S|R)'%(Rxx[0]))
						ind_ryx = getAllIndex(names, '(Bridge %d )(S|R)'%(Ryx[0]))
						ind_b = getAllIndex(names, '(Magnetic)')

						sdata = np.vsplit(cdata, child.args[2])
						slen = sdata[0].shape[0]
						pol = (np.nanmean(sdata[0][:slen/2,ind_b]) - np.nanmean(sdata[0][slen/2:,ind_b])) > 0
						if symmetrize:							
							for k in range(len(sdata)):
								srxx, srxx_std, sryx, sryx_std = [np.full((slen), np.nan) for z in range(4)]
								for i, e in enumerate(un_exc):
									ind = np.where(un_ind == i)[0]
									mask = np.full((sdata[k].shape[0]), False, dtype=np.bool)
									for z in ind:
										mask[z::len(un_ind)] = True
									maskdata = np.full(np.sum(mask), np.nan)
									data_points = (len(maskdata)+len(ind))/(2*len(ind))
									num_exc = len(un_ind)
									for z in range(data_points):
										left_rxx, right_rxx = sdata[k][z*num_exc+ind, ind_rxx[0]], sdata[k][-(z+1)*num_exc+ind, ind_rxx[0]]
										left_rxx_std, right_rxx_std = sdata[k][z*num_exc+ind, ind_rxx[1]], sdata[k][-(z+1)*num_exc+ind, ind_rxx[1]]
										left_ryx, right_ryx = sdata[k][z*num_exc+ind, ind_ryx[0]], sdata[k][-(z+1)*num_exc+ind, ind_ryx[0]]
										left_ryx_std, right_ryx_std = sdata[k][z*num_exc+ind, ind_ryx[1]], sdata[k][-(z+1)*num_exc+ind, ind_ryx[1]]
										if not pol:
											left_rxx, right_rxx = right_rxx, left_rxx
											left_rxx_std, right_rxx_std = right_rxx_std, left_rxx_std
											left_ryx, right_ryx = right_ryx, right_ryx
											left_ryx_std, right_ryx_std = right_ryx_std, left_ryx_std
										xx_tmp, xx_tmp2 = np.ravel([left_rxx, right_rxx]), np.ravel([left_rxx_std, right_rxx_std])
										yx_tmp, yx_tmp2 = np.ravel([-left_ryx, right_ryx]), np.ravel([left_ryx_std, right_ryx_std])
										if weighted_avg:											
											symm_rxx_std = np.sqrt(1./np.nansum(1/xx_tmp2**2)) if not np.isnan(xx_tmp2).all() else np.nan
											symm_rxx = np.nansum(xx_tmp/xx_tmp2**2)*symm_rxx_std**2
											symm_ryx_std = np.sqrt(1./np.nansum(1/yx_tmp2**2)) if not np.isnan(yx_tmp2).all() else np.nan
											symm_ryx = np.nansum(yx_tmp/yx_tmp2**2)*symm_ryx_std**2																					
										else:
											symm_rxx = np.nanmean(xx_tmp) if not np.isnan(xx_tmp2).all() else np.nan
											symm_rxx_std = np.nanmean(xx_tmp2) if not np.isnan(xx_tmp2).all() else np.nan
											symm_ryx = np.nanmean(yx_tmp) if not np.isnan(yx_tmp2).all() else np.nan
											symm_ryx_std = np.nanmean(yx_tmp2) if not np.isnan(yx_tmp2).all() else np.nan										

										srxx[z*num_exc+ind] = symm_rxx
										srxx[-(z+1)*num_exc+ind] = symm_rxx										
										srxx_std[z*num_exc+ind] = srxx_std[-(z+1)*num_exc+ind] = symm_rxx_std
										sryx[z*num_exc+ind] = symm_ryx if not pol else -symm_ryx
										sryx[-(z+1)*num_exc+ind] = -symm_ryx if not pol else symm_ryx
										sryx_std[z*num_exc+ind] = sryx_std[-(z+1)*num_exc+ind] = symm_ryx_std
									if show:
										plt.subplot(2, 1, 1)
										#plt.plot(sdata[k][mask,ind_b[0]], sdata[k][mask,ind_rxx[0]], 'o', label='%f'%(e))
										plt.plot(sdata[k][mask,ind_b[0]], srxx[mask], 'o', label='sym. %f'%(e))
										plt.subplot(2, 1, 2)
										#plt.plot(sdata[k][mask,ind_b[0]], sdata[k][mask,ind_ryx[0]], 'o', label='%f'%(e))
										plt.plot(sdata[k][mask,ind_b[0]], sryx[mask], 'o', label='asym. %f'%(e))
								sdata[k] = np.hstack([sdata[k], srxx.reshape(-1,1), srxx_std.reshape(-1,1), sryx.reshape(-1,1), sryx_std.reshape(-1,1)])
								if show:
									plt.legend()
									plt.title('Anti/Symmetrize at %.1fK'%(np.nanmean(sdata[k][:,ind_t[0]])))								
									plt.show()
							cdata = np.hstack(sdata)
							add_names = np.concatenate([add_names, [i.replace('(', 'symm. (') for i in names[getAllIndex(names, '(Bridge %d )(S|R)'%(Rxx[0]))]]])
							add_names = np.concatenate([add_names, [i.replace('(', 'antisymm. (') for i in names[getAllIndex(names, '(Bridge %d )(S|R)'%(Ryx[0]))]]])	
										
						if calc_mr:
							ind_r = len(names) if symmetrize else ind_rxx[0]							
							for k in range(len(sdata)):
								mr = np.full((slen), np.nan)
								for i, e in enumerate(un_exc):
									ind = np.where(un_ind == i)[0]
									mask = np.full((sdata[k].shape[0]), False, dtype=np.bool)
									for z in ind:
										mask[z::len(un_ind)] = True
									maskdata = np.full(np.sum(mask), np.nan)
									data_points = len(maskdata)/len(ind)
									min_val = sdata[k][mask][np.nanargmin(np.abs(sdata[k][mask,ind_b[0]])), ind_r]
									num_exc = len(un_ind)
									for z in range(data_points):
										for u in ind:
											mr[z*num_exc+u] = (sdata[k][z*num_exc+u, ind_r]-min_val)/min_val
									if show:
										plt.plot(sdata[k][mask, ind_b[0]], mr[mask], 'o', label='%f'%(e))
								sdata[k] = np.hstack([sdata[k], mr.reshape(-1,1)])
								if show:
									plt.legend()
									plt.title('MR at %.1fK'%(np.nanmean(sdata[k][:,ind_t[0]])))								
									plt.show()
							cdata = np.hstack(sdata)
							add_names = np.concatenate([add_names, ['Bridge %d MR. %s (a.u.)'%(Rxx[0], 'symm.' if symmetrize else '')]])

					else:
						cdata = np.hstack(np.vsplit(cdata, child.args[2]))
					header += '\n'+','.join([item for sublist in [np.concatenate([names, add_names])]*steps for item in sublist])
					header += '\n'+','.join([item for sublist in [['%.1f%s'%(x, {'T': 'K', 'B': 'T'}[child.type[2]])]*(cols+len(add_names)) for x in points] for item in sublist])					
				else:
					header += '\n'+','.join(names)
					header += '\n'

				#np.savetxt(filename, cdata, header=header, delimiter=',')
				for line in fileinput.input(filename, inplace=True):
					print re.sub('nan', '', line.rstrip())
				pos += child.size*child.args[2]
		else:
			pos = save_tree_data(child, depth+1, pos)
			
data_folder = os.path.splitext(path_data)[0]
with open(path_seq, 'r') as seq_file:
	make_tree(seq_file, root)
	root.size = calc_tree(root)
if not os.path.exists(data_folder):
	os.makedirs(data_folder)
save_tree_data(root)

print data.shape, root.size