f = open('sentences.tsv','r',encoding = 'utf-8')

lines = f.readlines()

f.close()

cut = []

for l in lines:
    cut.append(l.split('\t'))

f1 = open('english_tatoeba.txt','w',encoding = 'utf-8')
f2 = open('german_tatoeba.txt','w',encoding = 'utf-8')

cpl = [c[3] for c in cut]
cen = [c[1]+'\n' for c in cut]

f1.writelines(cen)
f2.writelines(cpl)

f1.close()
f2.close()