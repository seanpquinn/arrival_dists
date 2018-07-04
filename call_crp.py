#!/usr/bin/python

import numpy as np
import scipy.stats as ss
import crpropa as crp
import sys
import healpy as hp

a1,b1=(12.6-18.25)/2.75,(19.5-18.25)/2.75
a2,b2=(0.-0.2)/0.12,(1-0.2)/0.12
a3,b3=(0-10.97)/3.80,(25.-10.97)/3.8
a4,b4=(0-2.84)/1.3,(8-2.84)/1.3

E = float(sys.argv[1])
lon = float(sys.argv[2])
lat = float(sys.argv[3])
s1 = int(sys.argv[4])
s2 = int(sys.argv[5])
s3 = int(sys.argv[6])

pid = -crp.nucleusId(1,1)
sun = crp.Vector3d(-8.5,0,0) * crp.kpc

E = E * crp.EeV
nhat = hp.dir2vec(lon,lat,lonlat=True)
direc = crp.Vector3d()
direc.setXYZ(nhat[0],nhat[1],nhat[2])

ps = crp.ParticleState(pid,E,sun,direc)
cr = crp.Candidate(ps)
sim = crp.ModuleList()
sim.add( crp.Redshift() )
sim.add( crp.PhotoPionProduction(crp.CMB) )
sim.add( crp.PhotoPionProduction(crp.IRB) )
sim.add( crp.PhotoDisintegration(crp.CMB) )
sim.add( crp.PhotoDisintegration(crp.IRB) )
sim.add( crp.NuclearDecay() )
sim.add( crp.ElectronPairProduction(crp.CMB) )
sim.add( crp.ElectronPairProduction(crp.IRB) )
np.random.seed(s1)
p1 = ss.truncnorm.rvs(a1,b1,18.25,2.75,size=1)[0]
p2 = ss.truncnorm.rvs(a2,b2,0.2,0.12,size=1)[0]
p3 = ss.truncnorm.rvs(a3,b3,10.97,3.80,size=1)[0]
p4 = ss.truncnorm.rvs(a4,b4,2.84,1.30,size=1)[0]
Bfield = crp.JF12Field(s1,p1,p2,p3,p4)
Bfield.randomStriated(s2)
Bfield.randomTurbulent(s3)
sim.add(crp.PropagationCK(Bfield,1e-8,0.5*crp.parsec,15*crp.parsec))
sim.add(crp.SphericalBoundary(crp.Vector3d(0),20*crp.kpc))
sim.run(cr)
pos=cr.current.getPosition()
mom=cr.current.getMomentum()
print(cr.current.getEnergy(),pos.x,pos.y,
            pos.z,mom.x,mom.y,mom.z,cr.getRedshift())
