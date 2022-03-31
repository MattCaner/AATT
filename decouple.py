import xml.dom.minidom as x


def dec1():
    file = open("simpleTranslations.txt","r",encoding="utf-8")
    file2 = open("simplepl","w",encoding="utf-8")
    file3 = open("simpleen","w",encoding="utf-8")

    lines = file.readlines()

    for i in range(0,len(lines)):
        if i%4 == 0:
            file2.writelines([lines[i]])
        if i%4 == 2:
            file3.writelines([lines[i]])

def dec2():

    l = 40

    lines1 = []
    lines2 = []

    data = x.parse("text.xlf").getElementsByTagName("file")
    for iter, shred in enumerate(data):
        if iter > l:
            break
        if shred._attrs['original'].firstChild.data == '/2007-09-26/just-ahead-early-parliamentary-elections-ukraine' or shred._attrs['original'].firstChild.data == '/2007-10-03/political-crisis-georgia':
            continue
        for i, v in enumerate(shred.getElementsByTagName("body")[0].getElementsByTagName("trans-unit")):
            if i < 2:
                continue
            if  v.getElementsByTagName("source")[0].firstChild == None:
                continue
            if  v.getElementsByTagName("target")[0].firstChild == None:
                continue
            if len(v.getElementsByTagName("source")[0].firstChild.data) < 5:
                continue
            pl = v.getElementsByTagName("source")[0].firstChild.data.replace("\n"," ") + "\n"
            en = v.getElementsByTagName("target")[0].firstChild.data.replace("\n"," ") + "\n"

            lines1.append(pl)
            lines2.append(en)

    


    file2 = open("pl.txt","w",encoding = "utf-8")
    file3 = open("en.txt","w", encoding = "utf-8")

    file2.writelines(lines1)
    file3.writelines(lines2)

    file2.close()
    file3.close()


dec2()