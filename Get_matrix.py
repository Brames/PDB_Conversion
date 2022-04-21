import os
import numpy as np
import scipy as sp
from array import array

#Parsing dataset info
datadir = '../data/'
idfile = open('./id.all')
id_list = idfile.read().splitlines()
idfile.close()
typatm = np.dtype([('typ', 'S2'), ('pos', float, (3,)), ('rad', float), ('id', int)])

lig_ele_list = ['C','N','O','S','P','F','Cl','Br','I']
pro_ele_list = ['C','N','O','S']

#Natural distance and large number for self correlation
cut = 12.0
large = 1000.0
for i in range(0,len(id_list)):
	pdbid = id_list[i]
	infilename = pdbid+'_'+str(cut)+'.struct'
	if not os.path.exists(datadir+pdbid+'_C-C_'+str(cut)+'.mat'):
		print i, pdbid
		infile = open(datadir+infilename)
		data = np.load(infile)
		PRO = data['PRO']; LIG = data['LIG']
		infile.close()
		atoms = np.concatenate((LIG,PRO),axis=0)
		for pe in pro_ele_list:
			for le in lig_ele_list:
				matname = pe+'-'+le+'_'+str(cut)+'.mat'
				pair_atoms = []
				for l in range(0,len(atoms)):
					if (atoms[l]['typ'].replace(" ","")==le and atoms[l]['id']==1) or (atoms[l]['typ'].replace(" ","")==pe and atoms[l]['id']==-1):
						pair_atoms.append(atoms[l])
				n = len(pair_atoms)
				mat = np.zeros([n,n], float)
				for j in range(0,n):
					for k in range(0,n):
						if pair_atoms[j]['id']*pair_atoms[k]['id'] == -1:
							posj = pair_atoms[j]['pos']; posk = pair_atoms[k]['pos']
							dis = np.linalg.norm(posj - posk)
							mat[j,k] = dis
						else:
							mat[j,k] = large
				matfile = open(datadir+pdbid+'_'+matname, 'w')
				mat_array = np.zeros([n*n], float)
				for j in range(0,n):
					for k in range(0,n):
						mat_array[j*n+k] = mat[j,k]
				float_array = array('d', mat_array)
				int_array = array('l', [8067171840, 7, n])
				int_array.tofile(matfile)
				float_array.tofile(matfile)
				matfile.close()

cut = 6.0
for i in range(0,len(id_list)):
	pdbid = id_list[i]
	infilename = pdbid+'_'+str(cut)+'.struct'
	if not os.path.exists(datadir+pdbid+'_all_'+str(cut)+'.mat'):
		print i, pdbid
		infile = open(datadir+infilename)
		data = np.load(infile)
		PRO = data['PRO']; LIG = data['LIG']
		atoms = np.concatenate((LIG,PRO),axis=0)
		matname = 'all_'+str(cut)+'.mat'
		n = len(atoms)
		mat = np.zeros([n,n], float)
		for j in range(0,n):
			for k in range(0,n):
				posj = atoms[j]['pos']; posk = atoms[k]['pos']
				dis = np.linalg.norm(posj - posk)
				mat[j,k] = dis
		matfile = open(datadir+pdbid+'_'+matname, 'w')
		mat_array = np.zeros([n*n], float)
		for j in range(0,n):
			for k in range(0,n):
				mat_array[j*n+k] = mat[j,k]
		float_array = array('d', mat_array)
		int_array = array('l', [8067171840, 7, n])
		int_array.tofile(matfile)
		float_array.tofile(matfile)
		matfile.close()

rad_dict = {'C':1.7,'N':1.55,'O':1.52,'S':1.8,'P':1.8,'F':1.47,'Cl':1.75,'Br':1.85,'I':1.98}
p = 5
alpha = 1.0
cut = 12.0
large = 1000.0
for i in range(0,len(id_list)):
	pdbid = id_list[i]
	infilename = pdbid+'_'+str(cut)+'.struct'
	if not os.path.exists(datadir+pdbid+'_C-C_'+str(cut)+'_fri_'+str(p)+'_'+str(alpha)+'.mat'):
		print i, pdbid
		infile = open(datadir+infilename)
		data = np.load(infile)
		PRO = data['PRO']; LIG = data['LIG']
		infile.close()
		atoms = np.concatenate((LIG,PRO),axis=0)
		for pe in pro_ele_list:
			for le in lig_ele_list:
				matname = pe+'-'+le+'_'+str(cut)+'_fri_'+str(p)+'_'+str(alpha)+'.mat'
				pair_atoms = []
				for l in range(0,len(atoms)):
					if (atoms[l]['typ'].replace(" ","")==le and atoms[l]['id']==1) or (atoms[l]['typ'].replace(" ","")==pe and atoms[l]['id']==-1):
						pair_atoms.append(atoms[l])
				n = len(pair_atoms)
				mat = np.zeros([n,n], float)
				for j in range(0,n):
					for k in range(0,n):
						if pair_atoms[j]['id']*pair_atoms[k]['id'] == -1:
							posj = pair_atoms[j]['pos']; posk = pair_atoms[k]['pos']
							dis = np.linalg.norm(posj - posk)
							mat[j,k] = 1. - 1./(1.+np.power(alpha*dis/(rad_dict[pe]+rad_dict[le]),p))
						else:
							mat[j,k] = large
				matfile = open(datadir+pdbid+'_'+matname, 'w')
				mat_array = np.zeros([n*n], float)
				for j in range(0,n):
					for k in range(0,n):
						mat_array[j*n+k] = mat[j,k]
				float_array = array('d', mat_array)
				int_array = array('l', [8067171840, 7, n])
				int_array.tofile(matfile)
				float_array.tofile(matfile)
				matfile.close()

kk = 1
cut = 12.0
large = 10.0
for i in range(0,len(id_list)):
	pdbid = id_list[i]
	infilename = pdbid+'_'+str(cut)+'.struct'
	infile = open(datadir+infilename)
	data = np.load(infile)
	PRO = data['PRO']; LIG = data['LIG']
	infile.close()
	atoms = np.concatenate((LIG,PRO),axis=0)
	for pe in pro_ele_list:
		for le in lig_ele_list:
			if not os.path.exists(datadir+pdbid+'_'+pe+'-'+le+'_'+str(cut)+'_fri_exp_'+str(kk)+'.mat'):
				print i, pdbid
				matname = pe+'-'+le+'_'+str(cut)+'_fri_exp_'+str(kk)+'.mat'
				pair_atoms = []
				for l in range(0,len(atoms)):
					if (atoms[l]['typ'].replace(" ","")==le and atoms[l]['id']==1) or (atoms[l]['typ'].replace(" ","")==pe and atoms[l]['id']==-1):
						pair_atoms.append(atoms[l])
				n = len(pair_atoms)
				mat = np.zeros([n,n], float)
				for j in range(0,n):
					for k in range(0,n):
						if pair_atoms[j]['id']*pair_atoms[k]['id'] == -1:
							posj = pair_atoms[j]['pos']; posk = pair_atoms[k]['pos']
							dis = np.linalg.norm(posj - posk)
							mat[j,k] = 1. - np.exp(-np.power(dis/(rad_dict[pe]+rad_dict[le]),kk))
						else:
							mat[j,k] = large
				matfile = open(datadir+pdbid+'_'+matname, 'w')
				mat_array = np.zeros([n*n], float)
				for j in range(0,n):
					for k in range(0,n):
						mat_array[j*n+k] = mat[j,k]
				float_array = array('d', mat_array)
				int_array = array('l', [8067171840, 7, n])
				int_array.tofile(matfile)
				float_array.tofile(matfile)
				matfile.close()

cut = 6.0
p = 3
alpha = 0.5
for i in range(0,len(id_list)):
	pdbid = id_list[i]
	infilename = pdbid+'_'+str(cut)+'.struct'
	if not os.path.exists(datadir+pdbid+'_all_'+str(cut)+'_fri_3_0.5.mat'):
		print i, pdbid
		infile = open(datadir+infilename)
		data = np.load(infile)
		PRO = data['PRO']; LIG = data['LIG']
		atoms = np.concatenate((LIG,PRO),axis=0)
		matname = 'all_'+str(cut)+'_fri_3_0.5.mat'
		n = len(atoms)
		mat = np.zeros([n,n], float)
		for j in range(0,n):
			for k in range(0,n):
				posj = atoms[j]['pos']; posk = atoms[k]['pos']
				dis = np.linalg.norm(posj - posk)
				pe = atoms[j]['typ'].replace(" ","")
				le = atoms[k]['typ'].replace(" ","")
				mat[j,k] = 1. - 1./(1.+np.power(alpha*dis/(rad_dict[pe]+rad_dict[le]),p))
		matfile = open(datadir+pdbid+'_'+matname, 'w')
		mat_array = np.zeros([n*n], float)
		for j in range(0,n):
			for k in range(0,n):
				mat_array[j*n+k] = mat[j,k]
		float_array = array('d', mat_array)
		int_array = array('l', [8067171840, 7, n])
		int_array.tofile(matfile)
		float_array.tofile(matfile)
		matfile.close()

cut = 6.0
p = 3
alpha = 0.5
for i in range(0,len(id_list)):
	pdbid = id_list[i]
	infilename = pdbid+'_'+str(cut)+'.struct'
	if not os.path.exists(datadir+pdbid+'_pro_'+str(cut)+'_fri_3_0.5.mat'):
		print i, pdbid
		infile = open(datadir+infilename)
		data = np.load(infile)
		PRO = data['PRO']; LIG = data['LIG']
		atoms = PRO
		matname = 'pro_'+str(cut)+'_fri_3_0.5.mat'
		n = len(atoms)
		mat = np.zeros([n,n], float)
		for j in range(0,n):
			for k in range(0,n):
				posj = atoms[j]['pos']; posk = atoms[k]['pos']
				dis = np.linalg.norm(posj - posk)
				pe = atoms[j]['typ'].replace(" ","")
				le = atoms[k]['typ'].replace(" ","")
				mat[j,k] = 1. - 1./(1.+np.power(alpha*dis/(rad_dict[pe]+rad_dict[le]),p))
		matfile = open(datadir+pdbid+'_'+matname, 'w')
		mat_array = np.zeros([n*n], float)
		for j in range(0,n):
			for k in range(0,n):
				mat_array[j*n+k] = mat[j,k]
		float_array = array('d', mat_array)
		int_array = array('l', [8067171840, 7, n])
		int_array.tofile(matfile)
		float_array.tofile(matfile)
		matfile.close()
