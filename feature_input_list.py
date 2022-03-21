from PIL import Image
import model as f
import numpy as np
import os
import sys
import multiprocessing as mtp

global imgSrc
imgSrc='/research/byu2/hgeng/metric-learning/vias/test'
inList=sys.argv[1]
outFolder=sys.argv[2]
blocksize=int(sys.argv[3])  #100
blockdim=int(sys.argv[4])  #12
fealen=int(sys.argv[5])  #24

quant = np.empty(blocksize*blocksize, dtype=int).reshape(blocksize,blocksize)
quant.fill(1)
with open(inList) as hs:
    filenames = hs.readlines()
def get_label(files):
    files =files.split('/')[-1][:-1]
    if files[0]=='N':
        label=0
    else:
        label=1
    return label

def get_feature(files):
    files =files.split('/')[-1][:-1]
    im = Image.open(os.path.join(imgSrc,files))
    print("processing files %s"%files)
    imdata=np.asarray(im.convert('L'))
    tempfeature=f.feature(imdata, blocksize, blockdim, fealen)
    return tempfeature


cpu_count=mtp.cpu_count()
p=mtp.Pool(cpu_count)
tmp_data=[]
tmp_data.append(p.map(get_feature, filenames))
tmp_data=np.array(tmp_data[0])

labels=np.array(filenames)
tmp_label=[]
tmp_label.append(p.map(get_label, filenames))
tmp_label=np.array(tmp_label[0])

f.writecsv(outFolder, tmp_data, tmp_label, fealen)
