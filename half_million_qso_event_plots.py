import numpy as np
import os
import matplotlib.path as mplPath
import astropy.units as u
from astropy.coordinates import Galactic, ICRS
import matplotlib.pyplot as plt

def countQ(x):
  y1=len(x[x==b'Q'])
  y2=len(x[x==b'QR'])
  y3=len(x[x==b'QX'])
  y4=len(x[x==b'QRX'])
  y5=len(x[x==b'QR2'])
  return y1+y2+y3+y4+y5

def Qlocs(x):
  outind = np.array([],dtype=int)
  y1=np.where(x==b'Q')[0]
  y2=np.where(x==b'QR')[0]
  y3=np.where(x==b'QX')[0]
  y4=np.where(x==b'QRX')[0]
  y5=np.where(x==b'QR2')[0]
  outind=np.append(outind,y1)
  outind=np.append(outind,y2)
  outind=np.append(outind,y3)
  outind=np.append(outind,y4)
  outind=np.append(outind,y5)
  return outind

def Alocs(x):
  outind = np.array([],dtype=int)
  y1=np.where(x==b'A')[0]
  y2=np.where(x==b'AR')[0]
  y4=np.where(x==b'ARX')[0]
  y5=np.where(x==b'AR2')[0]
  outind=np.append(outind,y1)
  outind=np.append(outind,y2)
  outind=np.append(outind,y4)
  outind=np.append(outind,y5)
  return outind

def Blocs(x):
  outind = np.array([],dtype=int)
  y1=np.where(x==b'B')[0]
  y2=np.where(x==b'BR')[0]
  y3=np.where(x==b'BRX')[0]
  y4=np.where(x==b'BR2')[0]
  y5=np.where(x==b'B2')[0]
  outind=np.append(outind,y1)
  outind=np.append(outind,y2)
  outind=np.append(outind,y3)
  outind=np.append(outind,y4)
  outind=np.append(outind,y5)
  return outind

def qlocs(x):
  outind = np.array([],dtype=int)
  y1=np.where(x==b'q')[0]
  y2=np.where(x==b'qR')[0]
  y3=np.where(x==b'qX')[0]
  y4=np.where(x==b'qRX')[0]
  y5=np.where(x==b'qR2')[0]
  outind=np.append(outind,y1)
  outind=np.append(outind,y2)
  outind=np.append(outind,y3)
  outind=np.append(outind,y4)
  outind=np.append(outind,y5)
  return outind

def Klocs(x):
  outind = np.array([],dtype=int)
  y1=np.where(x==b'K')[0]
  y2=np.where(x==b'KR')[0]
  y3=np.where(x==b'KX')[0]
  y4=np.where(x==b'KRX')[0]
  y5=np.where(x==b'KR2')[0]
  y6=np.where(x==b'K2')[0]
  outind=np.append(outind,y1)
  outind=np.append(outind,y2)
  outind=np.append(outind,y3)
  outind=np.append(outind,y4)
  outind=np.append(outind,y5)
  outind=np.append(outind,y6)
  return outind

def countA(x):
  y1=len(x[x==b'A'])
  y2=len(x[x==b'AR'])
  y3=len(x[x==b'ARX'])
  y4=len(x[x==b'A2'])
  y5=len(x[x==b'AR2'])
  return y1+y2+y3+y4+y5

def countB(x):
  y1=len(x[x==b'B'])
  y2=len(x[x==b'BR'])
  y3=len(x[x==b'BRX'])
  y4=len(x[x==b'BR2'])
  y5=len(x[x==b'B2'])
  return y1+y2+y3+y4+y5

def countq(x):
  y1=len(x[x==b'q'])
  y2=len(x[x==b'qR'])
  y3=len(x[x==b'qX'])
  y4=len(x[x==b'qRX'])
  y5=len(x[x==b'qR2'])
  return y1+y2+y3+y4+y5

def countK(x):
  y1=len(x[x==b'K'])
  y2=len(x[x==b'KR'])
  y3=len(x[x==b'KX'])
  y4=len(x[x==b'KRX'])
  y5=len(x[x==b'KR2'])
  y6=len(x[x==b'K2'])
  return y1+y2+y3+y4+y5+y6

filelist=[]
datadir='./ellipse_coords/'
dirlist=os.listdir(datadir)
for s in dirlist:
  if ('ellipse' in s) and ('.npy' in s):
    filelist.append(datadir+s)

f=open("/home/sqnn/Downloads/HMQ.txt",'r')
radec=np.zeros((510764,2))
gtype=np.zeros(510764,dtype='|S5')
z=np.zeros(510764)
for i in range(510764):
  try:
    raw=f.readline()
    radec[i,0]=float(raw[0:12])
    radec[i,1]=float(raw[12:24])
    gtype[i]=raw[50:55].split()[0]
    try:
      z[i]=float(raw[75:80])
    except:
      z[i]=np.nan
  except:
    continue
f.close()

ind=np.where((np.abs(radec)>(0,0)).all(axis=1))
radec=radec[ind]
gtype=gtype[ind]
z=z[ind]

icrs=ICRS(radec[:,0],radec[:,1],unit=['deg','deg'])
galconv=icrs.galactic
l=galconv.l.deg
b=galconv.b.deg

l[l>180.]=l[l>180]-360.

lb=np.zeros((len(l),2))
lb[:,0]=l
lb[:,1]=b
filelist=os.listdir('./ellipse_coords/')

evtnum=np.zeros(202,dtype=int)
i=0
for evt in filelist:
  evtnum[i]=int(''.join([k for k in evt if k.isdigit()]))
  i+=1

filelist=np.array(filelist,dtype='|S100')
sortind=np.argsort(evtnum)
filelist=filelist[sortind]

#f=open("hmq_table.txt",'a')
for evt in filelist:
  rawdata=np.load('./ellipse_coords/'+evt.decode())
  bb=mplPath.Path(rawdata)
  evtnum=int(''.join([k for k in evt.decode() if k.isdigit()]))
  hits=bb.contains_points(lb)
  plt.plot(rawdata[:,0],rawdata[:,1],label='99.7%')
  ghits=gtype[hits]
  lhits=l[hits]
  bhits=b[hits]
  Qind=Qlocs(ghits)
  Aind=Alocs(ghits)
  Bind=Blocs(ghits)
  qind=qlocs(ghits)
  Kind=Klocs(ghits)
  N=len(Qind)+len(Aind)+len(Bind)+len(qind)+len(Kind)
  plt.scatter(lhits[Qind],bhits[Qind],marker='.',s=30,label='QSO')
  plt.scatter(lhits[Aind],bhits[Aind],marker='+',s=50,label='AGN',color='indigo')
  plt.scatter(lhits[Bind],bhits[Bind],marker='*',s=30,label='BL Lac',color='green')
  plt.scatter(lhits[qind],bhits[qind],marker='s',s=30,label='Hi conf. AGN',color='red')
  plt.scatter(lhits[Kind],bhits[Kind],marker='o',s=30,label='Type II AGN',color='gold')
  plt.legend(loc=2)
  plt.title('%i $N=$%i'%(evtnum,N),fontsize=20)
  plt.xlabel(r'$\ell$ (deg.)',fontsize=20)
  plt.ylabel(r'$b$ (deg.)',fontsize=20)
  plt.show()
  


  MZ = np.nanmean(z[hits])
  nz=len(z[hits])-len(np.where(np.isnan(z[hits])==1)[0])
  dMZ = np.nanstd(z[hits])/np.sqrt(len(np.not_equal(z[hits],np.nan)))
  N=numQ+numA+numB+numq+numK
  plt.plot(rawdata[:,0],rawdata[:,1])
  plt.scatter(l[hits],b[hits],marker='.',s=4)
  plt.show()
  #f.write("%i & %i & %i & %i & %i & %i & %.2f & %.2f & %i \\\\ \n" %(evtnum,numQ,numA,numB,numq,numK,MZ,dMZ,N))
#f.close()