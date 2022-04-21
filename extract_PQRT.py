import numpy as np
import sys,os

# ---- get data from pqrt file
#print('Get data from pqrt file...')
def getarray(str):
 lines=[]
 lists=[]
 a1=[]
 s1=[]
 tf=open(str)
 for lines in tf:
	if "ATOM" in lines:
		lists=lines.split()
		a1.append(lists)
 s1=np.array(a1)
 return s1

pro = getarray("pro_out.pqrt")
lig = getarray("lig_out.pqrt")
com = getarray("com_out.pqrt")

r1 = pro.shape[0]
r2 = lig.shape[0]
r3 = com.shape[0]

# ---- Calculate Coulombic and van der Waals
#print("Calculate Coulombic and van der Waals...")

coul = np.zeros((r1,r2))
van  = np.zeros((r1,r2))

for i in range(0,r1):
	for j in range(0,r2):
		dist=np.sqrt((float(pro[i,5])-float(lig[j,5]))**2+(float(pro[i,6])-float(lig[j,6]))**2+(float(pro[i,7])-float(lig[j,7]))**2)
		coul[i,j]=float(pro[i,8])*float(lig[j,8])/dist
		van[i,j]=((float(pro[i,9])+float(lig[j,9]))/dist)**12-2*(((float(pro[i,9])+float(lig[j,9]))/dist)**6)		


# ---- Create features by atom type
#print('Create features by atom type')
label1=['C','N','O','S']
label2=['C','N','O','S','P','F','Cl','Br','I']
label3=['CC','CN','CO','CS','CP','CF','CCl','CBr','CI','NN','NO','NS','NP','NF','NCl','NBr','NI','OO','OS','OP','OF','OCl','OBr','OI','SS','SP','SF','SCl','SBr','SI']

fecoulp=[[] for x in range(4)]
fecoull=[[] for x in range(9)]
fevan=[[] for x in range(30)]
coulp=coul.sum(axis=1)
coull=coul.sum(axis=0)
print(coulp)


# ---- Sorted coulombic of protein by atom type
for i in range(0,r1):
	for k in range(0,4):
		if (label1[k]==pro[i,10]):
			fecoulp[k].append(coulp[i])

# ---- Sorted coulombic of ligand by atom type
for i in range(0,r2):
	for k in range(0,9):
		if (label2[k]==lig[i,10]):
			fecoull[k].append(coull[i])
			

# ---- Sorted van der waals interaction by atom pair type
for i in range(0,r1):
	for j in range(0,r2):
		for k in range(0,30):
			if ((label3[k]==(pro[i,10]+lig[j,10])) or (label3[k]==(lig[j,10]+pro[i,10]))):
				fevan[k].append(van[i,j])
	

# ---- Calculate Atomic reaction field energy
#print('Calculate Atomic reaction field energy')

fesolig=[[] for x in range(9)]
fearlig=[[] for x in range(9)]
fesopro=[[] for x in range(4)]
fearpro=[[] for x in range(4)]
fesocom=[[] for x in range(9)]
fearcom=[[] for x in range(9)]

# ---- sorted soleng and area for protein
for i in range(0,r1):
	for k in range(0,4):
		if label1[k]==pro[i,10]:
			fesopro[k].append(float(pro[i,13]))
			fearpro[k].append(float(pro[i,12]))

# ---- sorted soleng and area for ligand
for i in range(0,r2):
	for k in range(0,9):
		if label2[k]==lig[i,10]:
			fesolig[k].append(float(lig[i,13]))
			fearlig[k].append(float(lig[i,12]))

# ---- sorted soleng and area for complex
for i in range(0,r3):
	for k in range(0,9):
		if label2[k]==com[i,10]:

			fesocom[k].append(float(com[i,13]))
			fearcom[k].append(float(com[i,12]))

# ---- get volume of ligand protein and complex

def getvolume(str):
	line=[]
	list=[]
	f=open(str)
	for line in f:
		if "VOLUMES" in line:
			list=line.split()
		        break
	return list[2]

vollig = float(getvolume("lig_out.pqrt"))
volpro = float(getvolume("pro_out.pqrt"))
volcom = float(getvolume("com_out.pqrt"))


# ---- get area of ligand protein and complex

def getarea(str):
        line=[]
        list=[]
        f=open(str)
        for line in f:
                if "AREAS" in line:
                        list=line.split()
                        break
        return list[2]

arlig = float(getarea("lig_out.pqrt"))
arpro = float(getarea("pro_out.pqrt"))
arcom = float(getarea("com_out.pqrt"))

# ---- get soleng of ligand protein and complex

def getsoleng(str):
        line=[]
        list=[]
        f=open(str)
        for line in f:
                if "MIBPB" in line:
                        list=line.split()
                        break
        return list[2]

solig=float(getsoleng("lig_out.pqrt"))
sopro=float(getsoleng("pro_out.pqrt"))
socom=float(getsoleng("com_out.pqrt"))


# ---- calculate feature vector
feature=[]

#calculate coulmobic feature
for i in range(0,4):
	if fecoulp[i]==[]:
		feature.append(0)
		feature.append(0)
		feature.append(0)
		feature.append(0)
		feature.append(0)
		feature.append(0)
	else:
		feature.append(sum(fecoulp[i]))
		feature.append(max(fecoulp[i]))
		feature.append(max([abs(x) for x in fecoulp[i]]))
		feature.append(min(fecoulp[i]))
		feature.append(min([abs(x) for x in fecoulp[i]]))
		feature.append(sum(fecoulp[i])/len(fecoulp[i]))

for i in range(0,9):
	if fecoull[i]==[]:
		feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
		feature.append(0)
		feature.append(0)
	else:
		feature.append(sum(fecoull[i]))
                feature.append(max(fecoull[i]))
		feature.append(max([abs(x) for x in fecoull[i]]))
                feature.append(min(fecoull[i]))
		feature.append(min([abs(x) for x in fecoull[i]]))
                feature.append(sum(fecoull[i])/len(fecoull[i]))

# ---- calculate van der waals feature
for i in range(0,30):
	if fevan[i]==[]:
		feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
	else:
		feature.append(sum(fevan[i]))
                feature.append(max(fevan[i]))
                feature.append(min(fevan[i]))
                feature.append(sum(fevan[i])/len(fevan[i]))

# ---- calculate soleng feature
for i in range(0,4):
        if fesopro[i]==[]:
                feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
		feature.append(0)
		feature.append(0)
        else:
                feature.append(sum(fesopro[i]))
                feature.append(max(fesopro[i]))
		feature.append(max([abs(x) for x in fesopro[i]]))
                feature.append(min(fesopro[i]))
		feature.append(min([abs(x) for x in fesopro[i]]))
                feature.append(sum(fesopro[i])/len(fesopro[i]))

for i in range(0,9):
        if fesolig[i]==[]:
                feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
		feature.append(0)
		feature.append(0)
        else:
                feature.append(sum(fesolig[i]))
                feature.append(max(fesolig[i]))
		feature.append(max([abs(x) for x in fesolig[i]]))
                feature.append(min(fesolig[i]))
		feature.append(min([abs(x) for x in fesolig[i]]))
                feature.append(sum(fesolig[i])/len(fesolig[i]))

for i in range(0,9):
        if fesocom[i]==[]:
                feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
		feature.append(0)
		feature.append(0)
        else:
                feature.append(sum(fesocom[i]))
                feature.append(max(fesocom[i]))
		feature.append(max([abs(x) for x in fesocom[i]]))
                feature.append(min(fesocom[i]))
		feature.append(min([abs(x) for x in fesocom[i]]))
                feature.append(sum(fesocom[i])/len(fesocom[i]))

feature.append(solig+sopro-socom)

# ---- calculate area feature

for i in range(0,4):
        if fearpro[i]==[]:
                feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
        else:
                feature.append(sum(fearpro[i]))
                feature.append(max(fearpro[i]))
                feature.append(min(fearpro[i]))
                feature.append(sum(fearpro[i])/len(fearpro[i]))

for i in range(0,9):
        if fearlig[i]==[]:
                feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
        else:
                feature.append(sum(fearlig[i]))
                feature.append(max(fearlig[i]))
                feature.append(min(fearlig[i]))
                feature.append(sum(fearlig[i])/len(fearlig[i]))

for i in range(0,9):
        if fearcom[i]==[]:
                feature.append(0)
                feature.append(0)
                feature.append(0)
                feature.append(0)
        else:
                feature.append(sum(fearcom[i]))
                feature.append(max(fearcom[i]))
                feature.append(min(fearcom[i]))
                feature.append(sum(fearcom[i])/len(fearcom[i]))

feature.append(arpro+arlig-arcom)

#calculate volume feature

feature.append(vollig)
feature.append(volpro)
feature.append(volcom)
feature.append(volcom-vollig-volpro)


#calculate electrostatic binding feature

feature.append(socom-sopro-solig+sum(coulp))

#for i in range(0,len(feature)):
#	print(feature[i])

f_handle = file('features.txt', 'a')
np.savetxt(f_handle, feature)
f_handle.close()

