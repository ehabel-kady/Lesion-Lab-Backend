# import os
# import pydicom
# # from pydicom.data import get_test_files


import os
import pydicom

path = '/home/ehab/Desktop/TCGA-02-0006/'
files = []
x = 0
names = []
for r, d, f in os.walk(path):
    if x == 1:
        names = d
    x +=1
    for file in f:
        if '.dcm' in file:
            files.append(os.path.join(r, file))

print(names)
ds = pydicom.dcmread('/home/ehab/Desktop/braintumor/braintumor/scripts/TCGA-02-0006/08-23-1996-MRI BRAIN W WO CONTRAMR-42545/1-LOC-49191/000000.dcm')
print(ds)