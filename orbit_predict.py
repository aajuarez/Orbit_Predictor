######################################################
######### Aaron J. Juarez, Sept 02--11 2015 ##########
######################################################
import numpy as np
from numpy import pi, cos, sin, tan, arccos, arcsin, arctan, arctan2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.optimize import newton
font = {'family':'serif', 'size':12}
plt.rc('font', **font)

#### constants
G = 6.67259e-8 #CGS
magicnumber = 206264.806
obliquity = 23.45 * (pi/180) #deg->rad, Earth's axial tilt
Nres=1e4

#### timeframe
#The Julian date for CE  2015 August    1 00:00:00.0 UT is JD 2457235.50
#The Julian date for CE  2015 December 31 00:00:00.0 UT is JD 2457387.50
#The Julian date for CE  2020 August    1 00:00:00.0 UT is JD 2459062.50
t = np.linspace(2457235.,2457387.5,Nres) #August 1 -- December 31, 2015
#t = np.linspace(2457235.,2459062.5,Nres) #Five years
#t = np.linspace(2457235.,2460888.5,Nres) #Ten years
t *= 86400 #time in sec

#### orbital elements
'''
years=11
t = np.linspace(0,years*3.15569e7,400)
t0 = 0 # time of periastron passage
a = 1.496e13 * 5.2 #1 AU in cm, multiplied by a factor
a = 1.496e13 * 1.0 #1 AU in cm, multiplied by a factor
e = 0.4
e = 0.01
inclin = 45 * (pi/180) #deg -> rad
ascnod = 5 * (pi/180)
argper = 7 * (pi/180)
inclin = 0.01 * (pi/180) #deg -> rad
ascnod = 0 * (pi/180)
argper = 50 * (pi/180)
m1 = 1.99e33  # Solar mass in grams
#m2 = 5.976e27 # Earth mass
m2 = 1 * 1.899e30 # Jupiter mass
V_Z = 0
'''
#### orbital elements : HD 80606b
t0 = 2454424.857 #pm0.05 JD, time of periastron passage; Moutou+ 2009
t0 *= 86400
a = 1.496e13 * 0.453 #pm0.015 AU -> cm; Moutou+ 2009
e = 0.9336  #pm0.0002
inclin = 89.285 * (pi/180) #deg -> rad, pm0.023 deg; Fossey+ 2009
ascnod = -19.02 * (pi/180) #-19.02|+160.98 pm 0.45 deg; Wiktorowicz & Laughlin 2014
#ascnod = 291.00 * (pi/180) # pm 6.7 deg; Naef+ 2001
argper = 300.4977 * (pi/180) #pm0.0045 deg; Fossey+ 2009
m1 = 0.9 * 1.99e33  # star mass in grams
m2 = 4.0 * 1.899e30 #pm0.3 Jupiter mass; Moutou+ 2009
V_Z = 3.767 * 1e5 #pm0.01 km/s --> cm/s; Naef+ 2001
mu_alp = 56.4 #mas/yr; IGSL
err_alp= 1.2
mu_del = 12.6 #mas/yr; IGSL
err_del= 1.7
RA  = 140.656571 * (pi/180) #deg->rad
DEC = 50.603736  * (pi/180) #deg->rad
plx = 17.13 #mas; Hipparcos
plx_err = 5.77
'''
#### orbital elements : upsilson andromedae c
t0 =  2449922.532
t0 *= 86400
a = 1.496e13 * 0.829 #pm0.043 AU -> cm
e = 0.245 #pm0.006
inclin = 7.868 * (pi/180) #deg -> rad
ascnod = 236.853 * (pi/180) #pm7.528
argper =  247.659 * (pi/180) #pm0.0045 deg
m1 = 1.31 * 1.99e33  # star mass in grams
m2 = 13.98 * 1.899e30 #pm0.3 Jupiter mass
V_Z = 53.480 * 1e5 #km/s --> cm/s
mu_alp = -172.77 #mas/yr; IGSL
err_alp= 1.2
mu_del = -382.45 #mas/yr
err_del= 1.7
RA  = 140.656571 * (pi/180) #deg->rad
DEC = 50.603736  * (pi/180) #deg->rad
plx = 73.45 #mas
plx_err = 5.77
#'''

##############################################################################
def find_nearest(array,values):
    nearest = np.array([])
    for i in values:
        idx = (np.abs(array-i)).argmin()
        nearest = np.append(nearest,array[idx])
    return nearest

def period(a,m1,m2):
    return np.sqrt(4 * pi**2 * a**3 / G / (m1+m2))

def Kepler_eq(E,e,M):
    return E - e * sin(E) - M

def orb_pred(m1,m2,a,e,inclin,ascnod,argper,t,t0,T):
#    T = period(a,m1,m2)
    # mean anomaly
    M = (t-t0) * 2 * pi / T
    # eccentric anomaly, solved by Newton-Raphson method
    E = newton(Kepler_eq, M, args=(e,M))
    # radius
    r = a*(1 - e*cos(E))
    # true anomaly
    f = 2*arctan(np.sqrt((1+e)/(1-e)) * tan(E*0.5)) #even sampled->?
#    f = arccos((a * (1-e**2)/r - 1)/e)
    return f,r

def sky_proj(r,f,inclin,ascnod,argper): #in-orbit frame to observer frame
    X = r * (cos(ascnod)*cos(argper+f) - sin(ascnod)*sin(argper+f)*cos(inclin))
    Y = r * (sin(ascnod)*cos(argper+f) - cos(ascnod)*sin(argper+f)*cos(inclin))
    Z = r * sin(argper+f) * sin(inclin)
    return X,Y,Z

def center_mass_frame(a,m1,m2):
    a1 = a*m2/(m1+m2) #star major axis
    a2 = a*m1/(m1+m2) #planet major axis
    return a1,a2

def radial_velocity(m1,m2,a,e,argper,inclin,f,V_Z,T):
    K = 2*pi/T * a * sin(inclin) * m2/(m1+m2) / np.sqrt(1-e**2)
    v_r = V_Z + K * (cos(argper+f) + e*cos(argper))
    return v_r

def suneclong(e,t,t0,T):
    M = (t-t0) * 2 * pi / T
    q = 2 * e * sin(M) + 5/4 * e**2 * sin(2*M)
    return q

def ellipse(parallax, beta, tta):
    x = parallax * cos(tta)
    y = parallax * sin(beta) * sin(tta)
    return x,y

##############################################################################
T = period(a,m1,m2)
#T = 111.81 * 8.64e4 #force period to be 111 days
print "T [yr] =", T/3.15569e7 # seconds -> year
print "T [day] =", T/8.64e4 # seconds -> day
a1,a2 = center_mass_frame(a,m1,m2)

X,Y,Z = [], [], []
Xa1,Ya1,Za1 = [],[],[]
Xa2,Ya2,Za2 = [],[],[]
v_rad1,v_rad2 = [],[]
r_arr=[]
proj_sep, proj_sep_a1, proj_sep_a2 = [],[],[]
PA, PA_a1, PA_a2 = [],[],[]
for i in range(len(t)):
    f,r = orb_pred(m1,m2,a,e,inclin,ascnod,argper,t[i],t0,T)
    X_,Y_,Z_ = sky_proj(r,f,inclin,ascnod,argper)
    X.append(X_/1.496e13);Y.append(Y_/1.496e13);Z.append(Z_/1.496e13)
    proj_sep.append(np.sqrt(X[i]**2 + Y[i]**2))
    PA.append(arctan2(Y[i],X[i])*(180/pi) + 180)
    
    f,r = orb_pred(m1,m2,a1,e,inclin,ascnod,argper+pi,t[i],t0,T)
    X_,Y_,Z_ = sky_proj(r,f,inclin,ascnod,argper+pi)
    Xa1.append(X_/1.496e13);Ya1.append(Y_/1.496e13);Za1.append(Z_/1.496e13)
    proj_sep_a1.append(np.sqrt(Xa1[i]**2 + Ya1[i]**2))
    PA_a1.append(arctan2(Ya1[i],Xa1[i])*(180/pi) + 180)
    
    v_rad_ = radial_velocity(m1,m2,a,e,argper,inclin,f,V_Z,T) # star
    v_rad1.append(v_rad_/1e5)
    
    f,r = orb_pred(m1,m2,a2,e,inclin,ascnod,argper,t[i],t0,T)
    r_arr.append(r/1.496e13)
    X_,Y_,Z_ = sky_proj(r,f,inclin,ascnod,argper)
    Xa2.append(X_/1.496e13);Ya2.append(Y_/1.496e13);Za2.append(Z_/1.496e13)
    proj_sep_a2.append(np.sqrt(Xa2[i]**2 + Ya2[i]**2))
    PA_a2.append(arctan2(Ya2[i],Xa2[i])*(180/pi) + 180)
    
    v_rad_ = radial_velocity(m2,m1,a,e,argper,inclin,f,V_Z,T) # planet
    v_rad2.append(v_rad_/1e5)

##############################################################################
### Projected Sep / PA
t/=8.64E4 # time in days
JDt0 = t[0]; t-=JDt0
print JDt0
fig = plt.figure(figsize=(12,6))
g=fig.add_subplot(121,axisbg='0.87')
h=fig.add_subplot(122,axisbg='0.87')
#g.plot(t, proj_sep,'k-')
#h.plot(t, PA,'k-')
g.plot(t, proj_sep_a1,'k-')
h.plot(t, PA_a1,'k-')
#g.plot(t, proj_sep_a2,'k-')
#h.plot(t, PA_a2,'k-')
g.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
g.set_ylabel(r'$\rho\ \rm[AU]$', fontsize=16)
h.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
h.set_ylabel(r'$\rm PA\ [deg]$', fontsize=16)
h.set_ylim([0,360])
h.set_yticks(np.linspace(0,360,7))
plt.tight_layout()



### Relative Astrometry
fig = plt.figure(figsize=(8,9))
g=fig.add_subplot(211,axisbg='0.87')#,aspect=1)
h=fig.add_subplot(212,axisbg='0.87')#,aspect=1)
g.plot(X,Y,'k-',X,Y,'k.')
h.plot(X,Z,'k-',X,Z,'k.')
g.set_xlabel(r'$X\ \rm[AU]$', fontsize=16)
g.set_ylabel(r'$Y\ \rm[AU]$', fontsize=16)
h.set_xlabel(r'$X\ \rm[AU]$', fontsize=16)
h.set_ylabel(r'$Z\ \rm[AU]$', fontsize=16)
plt.tight_layout()



### Absolute Astrometry
fig = plt.figure(figsize=(9,7))
cm=fig.add_subplot(111,axisbg='0.57')
cm.plot(Xa1,Ya1,'k-')#,Xa1,Ya1,'g.')
cm.plot(Xa2,Ya2,'k-')#,Xa2,Ya2,'b.')
cm.set_xlabel(r'$X\ \rm[AU]$', fontsize=16)
cm.set_ylabel(r'$Y\ \rm[AU]$', fontsize=16)
sc = cm.scatter(Xa1,Ya1, c=v_rad1, marker='o',
                vmin=min(v_rad2), vmax=max(v_rad2),
                s=40, cmap=plt.cm.RdBu_r, lw=0) # s: marker size
sc = cm.scatter(Xa2,Ya2, c=v_rad2, marker='o',
                vmin=min(v_rad2), vmax=max(v_rad2),
                s=40, cmap=plt.cm.RdBu_r, lw=0) # s: marker size
cbar = fig.colorbar(sc, ax=cm, pad=0.01, shrink=0.77) # sc is "mappable"
cbar.set_clim(min(v_rad2), max(v_rad2))
cbar.set_label(r'$v_{\rm rad}\ \rm[km\ s^{-1}]$', fontsize=16)
plt.tight_layout()

#''' find maxima & transit positions/radial velocities
#t/=8.64E4 # time in days
#t/=3.15569e7 # time in years
t_transit = 2454876.344 #pm0.01 JD; Fossey+ 2009
Ntra = 21; #Ntra = 22
t_tra = [t_transit + Ntra*T/8.64E4];t_tra-=JDt0
vr_interp1 = np.interp(t_tra, t, v_rad1)
time_close = find_nearest(t, t_tra)
Xper,Yper = [],[]
for i in range(len(time_close)):
    Xper.append(Xa2[np.where(t==time_close[i])[0]])
    Yper.append(Ya2[np.where(t==time_close[i])[0]])
cm.plot(Xper,Yper,'k*',ms=10)

t_periastron = 2454424.807 #pm0.05 JD; Moutou+ 2009
Nper = 24; #Nper = 25
t_per = [t_periastron + Nper*T/8.64E4,
         t_periastron + (Nper+0.5)*T/8.64E4,
         t_periastron + (Nper+1)*T/8.64E4];t_per-=JDt0
vr_interp2 = np.interp(t_per, t, v_rad1)
time_close = find_nearest(t, t_per)
Xper,Yper = [],[]
for i in range(len(time_close)):
    Xper.append(Xa2[np.where(t==time_close[i])[0]])
    Yper.append(Ya2[np.where(t==time_close[i])[0]])
cm.plot(Xper,Yper,'go',ms=10)
#'''



### Radial Velocity
fig = plt.figure(figsize=(8,6))
vr=fig.add_subplot(111,axisbg='0.87')
vr.plot(t,v_rad1,'k-'); vr.plot(t,v_rad1,'bo',alpha=0.3,mew=0,color='royalblue')
vr.plot(t_tra,vr_interp1,'k*',ms=11); print 't_tra:',t_tra
#JD 2457335.670000 is CE 2015 November 09 04:04:48.0 UT  Monday
#-> should be observable... transit at night, moon is down!
vr.plot(t_per,vr_interp2,'go',ms=10,clip_on=0); print 't_max:',t_per
#vr.set_xlabel(r'$t\ \rm[years]$', fontsize=16)
vr.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
vr.set_ylabel(r'$v_{\rm rad}\ \rm[km\ s^{-1}]$', fontsize=16)
plt.tight_layout()

for i in [g,h,cm]: i.plot(0,0,'kx',ms=10,mew=2)


### Radius vs time
r_arr=np.array(r_arr)
'''
fig = plt.figure()
rad=fig.add_subplot(111,axisbg='0.77')#,aspect=1)
rad.plot(t,r_arr)
#rad.plot(t,proj_sep);rad.plot(t,proj_sep_a1);rad.plot(t,proj_sep_a2)
plt.tight_layout()
#print r_arr[np.where(r_arr == r_arr[:1000].min())]
#print r_arr[np.where(r_arr == r_arr[1000:].min())]
#print t[np.where(r_arr == r_arr.min())]
#print t[np.where(r_arr == r_arr[:1000].min())]
#print t[np.where(r_arr == r_arr[1000:].min())]
#'''
cut = round(Nres/4)
ind1=np.where(r_arr == r_arr[:cut].min())[0][0] #min1
ind2=np.where(r_arr == r_arr[cut:].min())[0][0] #min2
ind3=np.where(r_arr == r_arr.max())[0][0]       #max
for i in [ind1,ind2,ind3]:
    cm.plot(Xa2[i],Ya2[i],'w^',ms=9)
    vr.plot(t[i],v_rad1[i],'w^',ms=9, clip_on=0)


### Astrometric Motion
dist = 1e3/plx #distance in pc
dist *= magicnumber #distance in AU
angdistx = arctan(np.array(Xa1)/dist)
angdisty = arctan(np.array(Ya1)/dist)
for i in [angdistx,angdisty]: #convert rad -> milli-arcsec
    i*=(180/pi)*3600 * 1e3


###pan 1: planet influence on the star
newx, newy = angdistx, angdisty
t_samp = np.random.choice(t,100); t_samp = np.sort(t_samp)
x_samp,y_meas,indz = [],[],[]
for i in range(len(t_samp)):
    indz.append(np.where(t_samp[i]==t)[0][0])
    x_samp.append(newx[indz[i]]+3e-3*np.random.randn())
    y_meas.append(newy[indz[i]]+3e-3*np.random.randn())

fig = plt.figure(figsize=(9,9))
rvt=fig.add_subplot(311,axisbg='0.87')
dvt=fig.add_subplot(312,axisbg='0.87')
pos=fig.add_subplot(313,axisbg='0.87')
rvt.plot(t, newx,'k-')
dvt.plot(t, newy,'k-')
pos.plot(newx,newy,'k-')
#pos.errorbar(x_samp,y_meas,xerr=10e-3,yerr=10e-3,color='k',fmt='.',capsize=0)
pos.errorbar(0,-1.2e-2,xerr=10e-3,yerr=10e-3,color='0.3',fmt='',capsize=0,lw=2)
rvt.plot(t_samp, x_samp,'b.',color='seagreen')
dvt.plot(t_samp, y_meas,'b.',color='seagreen')
pos.plot(x_samp,y_meas,'b.',color='seagreen')
rvt.set_ylabel(r'$\rm RA\ [mas]$', fontsize=16)
rvt.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
dvt.set_ylabel(r'$\rm DEC\ [mas]$', fontsize=16)
dvt.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
pos.set_ylabel(r'$\rm DEC\ [mas]$', fontsize=16)
pos.set_xlabel(r'$\rm RA\ [mas]$', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(hspace=0.25,bottom=0.07)


###pan 2: include parallax motion
#The Julian date for CE  2015 March 20 00:00:00.0 UT is JD 2457101.50
vernaleq = 2457101.50
beta = arcsin(cos(obliquity)*sin(DEC)+sin(obliquity)*cos(DEC)*sin(RA))
lamb = arcsin(cos(DEC)*cos(RA)/cos(beta))

#t*=8.64E4 # time in sec
#lamb_sun = suneclong(0.016711,t,t0,T)
#newx,newy = [],[]
#for i in range(len(t)):
#    del_lamb=plx*sin(lamb_sun[i]-lamb[i])/cos(beta)
#    del_beta=plx*cos(lamb_sun[i]-lamb[i])*sin(beta)
#    newx.append(angdistx[i] + del_beta * (180/pi)*3600 * 1e3)
#    newy.append(angdisty[i] + del_lamb * (180/pi)*3600 * 1e3)

tta = np.linspace(0, (t[-1]-t[0])/365.242 * 2 * pi, len(t)) #angle swept over time
tt_ = (t[0] + JDt0 - vernaleq) / 365.242 * 2 * pi #zero angle wrt vernal equinox
ellx,elly = ellipse(plx, beta, tta+tt_)
newx,newy = angdistx + ellx, angdisty + elly
x_samp,y_meas = np.array(x_samp) + ellx[indz], np.array(y_meas) + elly[indz]

#t/=8.64E4 # time in days
fig = plt.figure(figsize=(9,9))
rvt=fig.add_subplot(311,axisbg='0.87')
dvt=fig.add_subplot(312,axisbg='0.87')
pos=fig.add_subplot(313,axisbg='0.87')
rvt.plot(t, newx,'k-')
dvt.plot(t, newy,'k-')
pos.plot(newx,newy,'k-')
#pos.errorbar(x_samp,y_meas,xerr=10e-3,yerr=10e-3,color='k',fmt='.',capsize=0)
rvt.plot(t_samp,x_samp,'b.',color='seagreen')
dvt.plot(t_samp,y_meas,'b.',color='seagreen')
pos.plot(x_samp,y_meas,'b.',color='seagreen')
rvt.set_ylabel(r'$\rm RA\ [mas]$', fontsize=16)
rvt.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
dvt.set_ylabel(r'$\rm DEC\ [mas]$', fontsize=16)
dvt.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
pos.set_ylabel(r'$\rm DEC\ [mas]$', fontsize=16)
pos.set_xlabel(r'$\rm RA\ [mas]$', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(hspace=0.25,bottom=0.07)


###pan 3: include proper motion
mu_alp, mu_del = mu_alp/365.242, mu_del/365.242
delt=t[1]-t[0]
newx+=mu_alp*delt*np.arange(len(t))
newy+=mu_del*delt*np.arange(len(t))
for i in range(len(indz)):
    x_samp[i]+=mu_alp*delt*(indz[i]+1)
    y_meas[i]+=mu_del*delt*(indz[i]+1)

fig = plt.figure(figsize=(9,9))
rvt=fig.add_subplot(311,axisbg='0.87')
dvt=fig.add_subplot(312,axisbg='0.87')
pos=fig.add_subplot(313,axisbg='0.87')
rvt.plot(t, newx,'k-')
dvt.plot(t, newy,'k-')
pos.plot(newx,newy,'k-')
#pos.errorbar(x_samp,y_meas,xerr=10e-3,yerr=10e-3,color='k',fmt='.',capsize=0)
rvt.plot(t_samp,x_samp,'b.',color='seagreen')
dvt.plot(t_samp,y_meas,'b.',color='seagreen')
pos.plot(x_samp,y_meas,'b.',color='seagreen')
rvt.set_ylabel(r'$\rm RA\ [mas]$', fontsize=16)
rvt.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
dvt.set_ylabel(r'$\rm DEC\ [mas]$', fontsize=16)
dvt.set_xlabel(r'$t\ \rm[+ %s\ JD]$'%str(JDt0), fontsize=16)
pos.set_ylabel(r'$\rm DEC\ [mas]$', fontsize=16)
pos.set_xlabel(r'$\rm RA\ [mas]$', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(hspace=0.25,bottom=0.07)

''' 3D plot of orbits (CoM)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(Xa1,Ya1,Za1,'g-')
ax.plot(Xa2,Ya2,Za2,'b-')
colz=np.linspace(0,1,len(Xa2))
#for i in range(len(Xa2)):
#    ax.scatter(Xa2[i],Ya2[i],Za2[i])#c=plt.cm.RdBu_r(colz[i]))
ax.set_xlabel('$X$',size=16)
ax.set_ylabel('$Y$',size=16)
ax.set_zlabel('$Z$',size=16)
#'''
plt.show()
