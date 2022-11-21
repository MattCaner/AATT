f = open('pol.txt','r',encoding = 'utf-8')

lines = f.readlines()

f.close()

cut = []

for l in lines:
    cut.append(l.split('\t'))

f1 = open('simpleen2.txt','w',encoding = 'utf-8')
f2 = open('simplepl2.txt','w',encoding = 'utf-8')

cpl = [c[1]+'\n' for c in cut]
cen = [c[0]+'\n' for c in cut]

f1.writelines(cen)
f2.writelines(cpl)

f1.close()
f2.close()