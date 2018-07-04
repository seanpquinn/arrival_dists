import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import os
import itertools
import scipy.stats as ss
import scipy.optimize as so
import numpy.linalg as linalg
import scipy.spatial.distance as ssd
import healpy as hp
from scipy.spatial import ConvexHull
from matplotlib.ticker import NullFormatter
from rpy2.robjects.packages import importr
import rpy2
dc=importr('Directional')
from rpy2.robjects import numpy2ri
numpy2ri.activate()
import matplotlib.path as mplPath
from mpl_toolkits.basemap import Basemap
import pickle
from fastkde import fastKDE
from scipy.interpolate import interp1d

colorlist = ['#3B4CC0',
'#445ACC',
'#4D68D7',
'#5775E1',
'#6282EA',
'#6C8EF1',
'#779AF7',
'#82A5FB',
'#8DB0FE',
'#98B9FF',
'#A3C2FF',
'#AEC9FD',
'#B8D0F9',
'#C2D5F4',
'#CCD9EE',
'#D5DBE6',
'#DDDDDD',
'#E5D8D1',
'#ECD3C5',
'#F1CCB9',
'#F5C4AD',
'#F7B194',
'#F7A687',
'#F49A7B',
'#F18D6F',
'#EC7F63',
'#E57058',
'#DE604D',
'#D55042',
'#CB3E38',
'#C0282F',
'#B40426']
new_cmap = matplotlib.colors.ListedColormap(colorlist,name='custom_cmap')
plt.register_cmap(cmap=new_cmap)

datadir = 'events50_new_icrc.txt'
darray=np.genfromtxt(datadir,dtype=None,names=('id','E','lon','lat'))

filelist=[]
datadir='events_to_be_used/'
dirlist=os.listdir(datadir)

for s in dirlist:
  if ('evt' in s) and ('.npy' in s):
    filelist.append(datadir+s)

def PolyArea2D(pts):
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area

def onesig(x):
  return znew[znew>x].sum()-0.68*N_const

def twosig(x):
  return znew[znew>x].sum()-0.95*N_const

def threesig(x):
  return znew[znew>x].sum()-0.99*N_const


def kde2D(x, y, bandwidth, xbins=175j, ybins=175j, **kwargs): 
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins, 
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train  = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)

crpE = 0.1602176487

gzk_coords = np.loadtxt("gzk_horizon_pdf_coords.txt")
D_gzk = (50./29.79)*(gzk_coords[:,1]-39.85)
E_gzk = (10./24.8)*(gzk_coords[:,0]-43.75) + 55
gzk_interp = interp1d(E_gzk,D_gzk,kind='cubic')

def perturb_angle(xx,yy):
	#Calculates the deflection due to EGF
	dtheta = 0.8 * (np.pi / 180.) * (1 / (xx / 100)) * np.sqrt(yy / 10) * 0.9
	return dtheta

for evt in filelist:
	print "Loading %i of %i" %(filelist.index(evt),len(filelist))
	evt_split = evt.split('evt')[1]
	evtnum=int(''.join([k for k in evt_split if k.isdigit()]))
	rawdata=np.load(evt)
	if rawdata.max() == 0.:
		continue
	# Only use samples
	good_index = np.where(rawdata[:,-1] == 0.)[0]
	xyz1 = rawdata[good_index,1:4]
	xyz1l2norm = np.sqrt((xyz1*xyz1).sum(axis=1))
	pxpypz = rawdata[good_index,4:7]
	pl2norm = np.sqrt((pxpypz*pxpypz).sum(axis=1))
	pxpypz_norm = pxpypz/pl2norm.reshape(len(pxpypz),1)
	E1 = rawdata[good_index,0] / crpE
	if E1.mean() < 56:
		GZK_dist_scale = 250.
	elif E1.mean() > 109:
		GZK_dist_scale = 50.
	else:
		GZK_dist_scale = gzk_interp(E1.mean())
	#FIND PERTURBED VECTOR
	size_vec = len(xyz1)
	delta_theta = perturb_angle(E1.mean(),GZK_dist_scale)
	new_theta = np.random.uniform(max(0,0.666667*delta_theta),1.333333*delta_theta,size=size_vec)
	new_phi = np.random.uniform(-np.pi,np.pi,size=size_vec)
	rand_vec = np.zeros((size_vec,3))
	rand_vec[:,0] = np.random.uniform(0.,1.,size=size_vec)
	rand_vec[:,1] = np.random.uniform(0.,1.,size=size_vec)
	rand_vec[:,2] = np.random.uniform(0.,1.,size=size_vec)
	rv_norm = np.sqrt((rand_vec * rand_vec).sum(axis=1))
	rand_vec_norm = rand_vec / rv_norm.reshape(size_vec,1)
	u_vec = np.cross(pxpypz_norm,rand_vec_norm,axis=1)
	v_vec = np.cross(pxpypz_norm,u_vec,axis=1)
	pert_p1 = u_vec * np.cos(new_phi)[:,np.newaxis] + v_vec * np.sin(new_phi)[:,np.newaxis]
	pert_p2 = pert_p1 * np.sin(new_theta)[:,np.newaxis]
	pert_p = pert_p2 + pxpypz_norm * np.cos(new_theta)[:,np.newaxis]
	pert_norm = np.sqrt((pert_p*pert_p).sum(axis=1))
	pert_p_norm = pert_p / pert_norm.reshape(size_vec,1)
	pert_xyz = xyz1/xyz1l2norm.reshape(len(xyz1),1)*20e-3 + pert_p_norm*GZK_dist_scale
	#END OF PERTURBED VECTOR
	#SAME CUTS ALSO APPLIED TO PERTURBED VECTOR
	xyz = xyz1/xyz1l2norm.reshape(len(xyz1),1)*20e-3 + pxpypz_norm*GZK_dist_scale
	l2norm = np.sqrt((xyz*xyz).sum(axis=1))
	xyz_norm = xyz/l2norm.reshape(len(xyz),1)
	lon = np.zeros(len(xyz))
	lat = np.zeros(len(xyz))
	#PERTURBED
	pert_l2norm = np.sqrt((pert_xyz*pert_xyz).sum(axis=1))
	pert_xyz_norm = pert_xyz / pert_l2norm.reshape(len(pert_xyz),1)
	p_lon = np.zeros(len(pert_xyz))
	p_lat = np.zeros(len(pert_xyz))
	for i in range(len(xyz)):
		lon[i],lat[i] = hp.vec2dir(xyz[i],lonlat=True) 
		p_lon[i],p_lat[i] = hp.vec2dir(pert_xyz[i],lonlat=True) 
	#Test for nans (might be a result of run stoppage)
	tmp = np.isnan(lat)
	if len(lat[tmp]) > 0:
		num_nans = len(lat[tmp])
		print "found %i nans in lat array. Exlcuding." %num_nans
		lon = lon[~tmp]
		lat = lat[~tmp]
		xyz = xyz[~tmp]
		xyz_norm = xyz_norm[~tmp]
		pxpypz_norm = pxpypz_norm[~tmp]
		E1 = E1[~tmp]
		p_lon = p_lon[~tmp]
		p_lat = p_lat[~tmp]
	#Wrap coords to [0,360] if necessary
	n1 = len(np.where(lon<-175)[0])
	n2 = len(np.where(lon>175)[0])
	# Grab observed values
	obs_ind=np.where(darray['id']==evtnum)[0][0]
	obs_E=darray['E'][obs_ind]
	obs_lon = darray['lon'][obs_ind]
	obs_lat = darray['lat'][obs_ind]
#	if n1 > 500 and n2 > 500:
#		neg_ind = np.where(lon<0)[0]
#		lon[neg_ind] = 360 - np.abs(lon[neg_ind])
#	else:
#		obs_lon = 180. - obs_lon
	#Cut outliers
	dt = np.zeros(len(lon),dtype=np.float32)
	#sampmean=np.array([lon.mean(),lat.mean()])
	#Calculate spherical mean
	R_xyz_norm = rpy2.robjects.Matrix(xyz_norm)
	R_median_dir = rpy2.robjects.r['mediandir']
	m_dir = np.array(R_median_dir(R_xyz_norm))
	sampmean = hp.vec2dir(m_dir,lonlat=True)
	for i in range(len(lon)):
		dt[i] = hp.rotator.angdist([lon[i],lat[i]],sampmean,lonlat=True)*180./np.pi
	#QUALITY CUT: REMOVE POINTS WITH ANG DIST > 3SIGMA
	cut1 = np.where(dt < np.median(dt) + 3*dt.std())[0]
	dt1 = dt[cut1]
	xyz1 = xyz[cut1]
	xyz_norm1 = xyz_norm[cut1]
	pxpypz_norm1 = pxpypz_norm[cut1]
	E2 = E1[cut1]
	lon1 = lon[cut1]
	lat1 = lat[cut1]
	p_lon1 = p_lon[cut1] #Perturbed lon cut
	p_lat1 = p_lat[cut1] #Perturbed lat cut
	nnn_pct = np.percentile(dt,95)
	cut2 = np.where(dt1 < nnn_pct)[0]
	dt2 = dt1[cut2]
	xyz2 = xyz1[cut2]
	xyz_norm2 = xyz_norm1[cut2]
	pxpypz_norm2 = pxpypz_norm1[cut2]
	E3 = E2[cut2]
	lon2 = lon1[cut2]
	lat2 = lat1[cut2]
	p_lon2 = p_lon1[cut2] #Perturbed lon cut
	p_lat2 = p_lat1[cut2] #Perturbed lat cut
	m=Basemap(projection='moll',lon_0=0,resolution='c',celestial=True)
	pars = np.arange(-90,90,30)
	mers = np.arange(-180,180,30)
	plt.subplot(121)
	plt.subplots_adjust(left=0.05,right=0.95)
	proj_lon,proj_lat = m(lon,lat)
	m.scatter(proj_lon,proj_lat,marker='.',s=3)
	#num_cut = 10**5 - len(lon2)
	num_cut = len(lon) - len(lon2)
	#pct_cut = num_cut / 100000. * 100
	pct_cut = num_cut / len(lon) * 100.
	# Draw hull around cut region
	ha = np.zeros((len(lon2),2))
	ha[:,0]=lon2
	ha[:,1]=lat2
	hull = ConvexHull(ha)
	x = ha[hull.vertices,0]
	y = ha[hull.vertices,1]
	x = np.append(x,x[0])
	y = np.append(y,y[0])
	x1,y1 = m(x,y)
	m.plot(x1,y1,'r-',lw=2)
	m.drawparallels(pars,labels=[1,0,0,0],labelstyle='+/-')
	m.drawmeridians(mers)
	plt.title("Event %i" %evtnum)
	xmin,xmax,ymin,ymax=p_lon2.min()-1,p_lon2.max()+1,p_lat2.min()-1,p_lat2.max()+1
	#Fast KDE based on O'Brien et al., Comput. Stat. Data Anal. 101, 148-160 (2016)
	xax = np.linspace(xmin,xmax,513)
	yax = np.linspace(ymin,ymax,513)
	myPDF,axes=fastKDE.pdf(p_lon2,p_lat2,axes=[xax,yax],numPoints=513)
	zz = myPDF
	ax1 = np.zeros(len(axes[0]))
	ax2 = np.zeros(len(axes[1]))
	ax1 = axes[0]
	ax2 = axes[1]
	xx,yy = np.meshgrid(ax1,ax2)
	xy = np.zeros((len(x),2))
	xy[:,0] = x
	xy[:,1] = y
	bbp = mplPath.Path(xy)
	mask_array = np.zeros((len(ax1),len(ax2)),dtype=int)
	for i in range(len(ax1)):
		for j in range(len(ax2)):
			if bbp.contains_point((ax1[i],ax2[j])):
				mask_array[i,j] = 0
			else:
				mask_array[i,j] = 1
	#Remove probability density beyond cut boundary
	znew = np.ma.array(zz, mask=mask_array)
	zflat = znew.flatten()
	zflat = zflat[~zflat.mask].data
	N_const = zflat.sum()
	plt.subplot(122)
	x0,y0=sampmean
	#Apparently impossible to zoom in on Mollweide, use Lambert equal area
	#Care must be taken for events near the poles
	if y0 > 70.:
		m2 = Basemap(projection='nplaea',lon_0=x0,boundinglat=y0-12)
	elif y0 < -70.:
		m2 = Basemap(projection='splaea',lon_0=x0,boundinglat=y0+12)
	else:
		m2 = Basemap(projection='laea',lon_0=x0,lat_0=y0,llcrnrlat=ymin,
			urcrnrlat=ymax,llcrnrlon=xmin,urcrnrlon=xmax)
	proj_xx,proj_yy = m2(xx,yy)
	m2.contourf(proj_xx,proj_yy,znew,15,cmap='custom_cmap')
	pars2 = np.arange(-90,90,15)
	mers2 = np.arange(-180,180,15)
	m2.drawparallels(pars2,labels=[1,0,0,0],labelstyle='+/-')
	m2.drawmeridians(mers2,labels=[0,0,0,1],labelstyle='+/-')
	plt.title('%i of 100000 cut, %.2f pct' %(num_cut,pct_cut))
	plt.tight_layout(h_pad=0.4,rect=[0.02,0.1,0.97,0.9])
	plt.savefig('cut_%i.png' %evtnum)
	plt.close('all')
	cset=plt.contour(xx,yy,znew,15,cmap='custom_cmap')
	# Save all level curves, just to have on hand
	# it's a pickle of paths. To unpack, just go through and get vertices
	#with open('./levels/level_data_%i' %(evtnum),'w') as F:
	#	pickle.dump(cset.collections,F)
	#To get the vertices simply load the file
	#>>unpick=pickle.load(open('./levels/level_data_60815505400','rb'))
	#>>t1=unpick[0].get_paths()
	#>>verts=t1[0].vertices  First element of this list are the vertices
	#Rinse and repeat for length of unpick
	x1=cset.levels[1] / 10.
	x2=x1
	x3=x1
	l3=so.fsolve(onesig,x1)[0]
	l2=so.fsolve(twosig,x2)[0]
	l1=so.fsolve(threesig,x3)[0]
	# Clear dummy contour plot
	plt.close('all')
	max_lat = max([obs_lat+2,ymax])
	max_lon = max([obs_lon+2,xmax])
	min_lat = min([obs_lat-2,ymin])
	min_lon = min([obs_lon-2,xmin])
	print "Max: (%.2f,%.2f) Min: (%.2f,%.2f)" %(max_lon,max_lat,min_lon,min_lat)
	cset=plt.contour(xx,yy,znew,[l2,l3],colors='k',fignum=1)
	sig_areas = [0] * 2
	for i in range(2):# 2 contours
		ps = cset.collections[i].get_paths()
		if len(ps) < 1:# Possible that plt.contour doesn't draw the curve
			continue
		vert = ps[0].vertices #Vertices are the points of the contour
		ellarea=PolyArea2D(vert)
		sig_areas[i] = ellarea
	plt.close('all')
	m2 = Basemap(projection='moll',lon_0=0.,lat_0=0.,celestial=True)
	proj_xx,proj_yy = m2(xx,yy)
	obs_x,obs_y = m2(obs_lon,obs_lat)
	fig=plt.figure()
	ax=fig.add_subplot(111)
	m2.contourf(proj_xx,proj_yy,znew,15,cmap='custom_cmap')
	#Add arrival direction point
	m2.plot(obs_x,obs_y,marker='.',c='gray',markersize=16)
	m2.colorbar(pad=0.12)
	m2.drawparallels(pars2,labels=[1,0,0,0],labelstyle='+/-')
	m2.drawmeridians(mers2,labels=[0,0,0,1],labelstyle='+/-')
	#Add arrow from arrival direction to distribution mean
	tx,ty = m2(x0,y0)
	plt.annotate("",xy=(tx,ty),xycoords='data',
		xytext=(obs_x,obs_y),textcoords='data',arrowprops=dict(arrowstyle="->",
		connectionstyle="arc3,rad=0.16"),size=16)
	#Plot deflection angle value
	dtheta = hp.rotator.angdist([obs_lon,obs_lat],sampmean,lonlat=True)*180./np.pi
	plt.text(0.82,0.05,r"$\delta \theta=%.1f^{\circ}$" %dtheta,
		fontsize=16,transform=ax.transAxes)
	cset=m2.contour(proj_xx,proj_yy,znew,[l2,l3],colors='k',fignum=1)
	fmt={}
	strs=['95%','99.7%']
	strs.reverse()
	for l, s in zip(cset.levels,strs):
		fmt[l]=s
	plt.clabel(cset,fmt=fmt)
	title_str = r'ID=%i $A_1=%.1f$ $A_2=%.1f$' %(evtnum,
		sig_areas[0],sig_areas[1])
	plt.title(title_str,y=1.0,fontsize=16)
	plt.tight_layout(rect=[0.05,0.05,0.95,0.95])
	fig = plt.gcf()
	fig.savefig("lat_lon_contour_%i.png" %evtnum)
	plt.close('all')
	cset=plt.contour(xx,yy,znew,[l1,l2,l3],colors='k',fignum=1)
	cont = cset.collections[1].get_paths()[0].vertices
	np.save('sig_contours/contour_%i.npy' %evtnum,cont)
	# Dump sigma levels to pickle file
#	with open('./siglevels/siglevel_data_%i' %(evtnum),'w') as F:
#		pickle.dump(cset.collections,F)	
#	plt.close('all')
	with open('analysis_params.txt','a') as F:
		s = "%i " %evtnum + "%.2f "*5 %(obs_E,obs_lon,obs_lat,sampmean[0],
			sampmean[1])
		s += "%.2f "*2 %tuple(sig_areas)
		F.write(s+"\n")
	print "Event analysis complete"
