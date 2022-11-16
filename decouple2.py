import csv

c1 = []
c2 = []
c3 = []
with open('pol.txt', 'r') as f:
    lines = f.readlines()
    for l in lines:
        ll = l.split('\t')
        c1.append(ll[0])
        c2.append(ll[1])
    f1 = open('spol.txt','w')
    f2 = open('seng.txt','w')
    f1.writelines(c1)
    f2.writelines(c2)