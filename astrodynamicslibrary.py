# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#libraries
import numpy as np
import math
from lambertsolver import lambertizzo2015 as lambert
from scipy import optimize
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GME = 398600.4418

myatan = lambda x,y: np.pi*(1.0-0.5*(1+np.sign(x))*(1-np.sign(y**2))\
         -0.25*(2+np.sign(x))*np.sign(y))\
         -np.sign(x*y)*np.arctan((np.abs(x)-np.abs(y))/(np.abs(x)+np.abs(y)))

def period(a,m=GME):
# This function computes the period of the satellite in seconds
# input:
#    a -semi-major axis (km)
#    m- gravitational parameter of central body (km^3/s^2)
# output:
#   period - period of the satellite's orbit in seconds
    
    return math.sqrt(a**3*4*np.pi**2/m)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def coe_from_sv(R,V,mu):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the classical orbital elements (coe)
#  from the state vector (R,V).
#
#  input:
#  mu   - gravitational parameter of central body (km^3/s^2)
#  R    - position vector (km)
#  V    - velocity vector (km/s)
#
#  output:
#  coe  - classical orbital elements [a,e,i,RAAN,w,TA]
#
#  Functions required: none
# ---------------------------------------------
    rsize=math.sqrt(R[0]**2+R[1]**2+R[2]**2)
    vsq=V[0]**2+V[1]**2+V[2]**2
    rdotv=R[0]*V[0]+R[1]*V[1]+R[2]*V[2]
    Energy=1/2*vsq-mu/rsize
    a=-mu/(2*Energy)
    h=np.cross(R,V)
    hsize=math.sqrt(h[0]**2+h[1]**2+h[2]**2)
    i=math.acos(h[2]/hsize)
    p=[(vsq-mu/rsize)/GME,-rdotv/GME]
    ex=p[0]*R[0]+p[1]*V[0]
    ey=p[0]*R[1]+p[1]*V[1]
    ez=p[0]*R[2]+p[1]*V[2]
    e=math.sqrt(ex**2+ey**2+ez**2)
    nx=-h[1]
    ny=h[0]
    nz=0
    n=math.sqrt(nx**2+ny**2)
   
    if n==0:
        n=0.00001
    if (ny>0):
        RAAN=math.acos(nx/n)
    else:
        RAAN=2*math.pi-math.acos(nx/n)
    
    ndote=nx*ex+ny*ey+nz*ez
    acosargom = ndote/(e*n);
    
    if(ez>0):
        w=math.acos(acosargom)
    else:
        w=2*math.pi-math.acos(acosargom)   
        
    edotr=ex*R[0]+ey*R[1]+ez*R[2]
    acosargf = edotr/(e*rsize);
    
    if(rdotv>0):
        TA=math.acos(acosargf)
    else:
        TA=2*math.pi-math.acos(acosargf)   
            
    
    coe=[a,e,np.rad2deg(i),np.rad2deg(RAAN),np.rad2deg(w),np.rad2deg(TA)]

    return coe 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sv_from_coe(coe,mu):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the state vector (r,v) from the
#  classical orbital elements (coe).
#
#  input: 
#  coe  - classical orbital elements [a,e,i,RAAN,w,TA]
#  mu   - gravitational parameter of central body (km^3/s^2)
#
#  output:
#  R    - position vector  (km)
#  V    - velocity vector  (km/s)
#  
#  Functions required: none
# ----------------------------------------------
    a=coe[0]
    e=coe[1]
    i=np.deg2rad(coe[2])
    RAAN=np.deg2rad(coe[3])
    w=np.deg2rad(coe[4])
    TA=np.deg2rad(coe[5])
    p=a*(1-e**2)
    r=p/(1+e*math.cos(TA))
    
    rpqw=r*np.array([math.cos(TA),math.sin(TA),0]) #r in the perifocal frame
    vpqw=math.sqrt(mu/p)*np.array([-math.sin(TA),e+math.cos(TA),0]) #v in the perifocal frame
    R3=np.array([[np.cos(RAAN),np.sin(RAAN),0],[-np.sin(RAAN),np.cos(RAAN),0],[0,0,1]])
    R2=np.array([[1,0,0],[0,np.cos(i),np.sin(i)],[0,-np.sin(RAAN),np.cos(RAAN)]])
    R3v=np.array([[np.cos(w),np.sin(w),0],[-np.sin(w),np.cos(w),0],[0,0,1]])
    if(math.isnan(w)):
        R3v=np.identity(3)
    else:
        R3v=np.array([[np.cos(w),np.sin(w),0],[-np.sin(w),np.cos(w),0],[0,0,1]])
    R3t=R3.transpose()
    R2t=R2.transpose()
    R3vt=R3v.transpose()
    
    T=R3t.dot(R2t).dot(R3vt)
    
    R=np.dot(T,rpqw)
    V=np.dot(T,vpqw)
    #print(reci)
   


    return R,V
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def date2jd(year,month,day,hour,minute,second):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the julian day from an input calendar date.
#
#  input:
#  year   - year in YYYY
#  month  - month in MM
#  day    - day in DD
#  hour   - hour in HH
#  minute - minutes in MM
#  second - seconds in SS.
#
#  output:
#  jd - julian day
#
#  Functions required: none
#---------------------------------------------

    #...Equation 5.48:
    j0 = 367*year - np.floor(7*(year + np.floor((month + 9)/12))/4) + np.floor(275*month/9) + day + 1721013.5;

    ut = (hour + minute/60 + second/3600)/24
    
    #...Equation 5.47
    jd = j0 + ut

    return jd
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def M2E(e, M):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function uses Newton's method to solve Kepler's 
#  equation  E - e*sin(E) = M  for the eccentric anomaly,
#  given the eccentricity and the mean anomaly.
#
#  input:
#  E  - eccentric anomaly (radians)
#  e  - eccentricity
#
#  output:
#  M  - mean anomaly (radians) 
#
#  Functions required: kep_eq (nest definition)
# ----------------------------------------------
    if M<0:
        print("M<0")
    if M<=math.pi:
        E=M+e/2
    else:
        E=M-e/2
    h=1;
    while abs(h)>=0.0001:
        h=(E-e*math.sin(E)-M)/(1-e*math.cos(E))
        E=E-h
     
    if E<0:
        print("E<0")
    return E
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def E2M(e,E):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the the mean anomaly
# from the eccentric anomaly and the eccentricity.
#
#  input:
#  E  - eccentric anomaly (radians)
#  e  - eccentricity
# 
#  output:
#  M  - mean anomaly (radians) 
#
#  Functions required: none
# ----------------------------------------------

    M=E-e*math.sin(E)
         
    return M
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def E2f(e,E):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the the true anomaly
# from the eccentric anomaly and the eccentricity.
#
#  input:
#  E  - eccentric anomaly (radians)
#  e  - eccentricity
#
#  output:
#  f  - true anomaly (radians)
#
#  Functions required: none 
# ----------------------------------------------  
    
    f=(math.atan(math.sqrt((1+e)/(1-e))*math.tan(E/2)))*2 #atan returns (-pi/2,+pi/2)
    if f<0:
        f=f+2*math.pi
     
    
    return f
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def f2E(e,f):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the the true anomaly
# from the eccentric anomaly and the eccentricity.
#
#  input:
#  E  - eccentric anomaly (radians)
#  e  - eccentricity
#
#  output:
#  f  - true anomaly (radians)
#
#  Functions required: none 
# ----------------------------------------------
    E=(math.atan(math.sqrt((1-e)/(1+e))*math.tan(f/2)))*2
    if E<0:
        E=E+2*math.pi
    
    return E
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def kep_propagate(kep,  DT, mu):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function propages an initial Keplerian state
#  for a time interval DT.
#
#  input:
#  kep - initial keplerian state [a,e,i,RAAN,w,TA]
#  DT - time interval to propagate
#  mu   - gravitational parameter of central body (km^3/s^2)
# 
#  output:
#  r - position vector after DT (km)
#  v - velocity vector after DT (km/s)
#
#  Functions required: f2E, E2M, M2E, E2f, sv_from_coe 
# ----------------------------------------------  
    
    #initialstate=sv_from_coe(kep,mu)
    a=kep[0]
    e=kep[1]
    i=np.deg2rad(kep[2])
    RAAN=np.deg2rad(kep[3])
    w=np.deg2rad(kep[4])
    TAinit=np.deg2rad(kep[5])
    Einit=f2E(e,TAinit)
    Minit=E2M(e,Einit)
        
    n=math.sqrt(mu/a**3)
    
    Mfinal=Minit+n*DT%(2*math.pi)
    Efinal=M2E(e,Mfinal)
    TAfinal=E2f(e,Efinal)
    r,v=sv_from_coe([a,e,np.rad2deg(i),np.rad2deg(RAAN),np.rad2deg(w),np.rad2deg(TAfinal)],mu)
    #print(np.rad2deg(TAfinal))
     
    
    return r,v
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def groundtrack(kep, ug0, DT, steps, mu):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the groundtrack of satellite given
#  the initial keplerian state vector and the Greenwich
#  hour angle at t=0
#
#  input:
#  kep  - initial keplerian state [a,e,i,RAAN,w,TA]
#  ug0  - initial Greenwich hour angle
#  DT   - timestep
#  step - number of steps
#  mu   - gravitational parameter of central body (km^3/s^2)
# 
#  output:
#  phi_list    - list of latitudes at each time-step  
#  lambda_list - list of longitudes at each time-step
#
#  Functions required: kep_propagate
# ----------------------------------------------
    # Earth's rotational rate
    n_earth = 2*np.pi/86164.09053083288
    
    Lambda_list=np.zeros(steps)
    phi_list=np.zeros(steps)
    
    for i in range(steps):
        r,v=kep_propagate(kep,i*DT,mu)
        
        
        
        x=r[0]
        y=r[1]
        z=r[2]
        
        rabs=math.sqrt(x**2+y**2+z**2)
        if np.abs(rabs)<0.00001:
            if rabs<0:
                rabs=-0.00001
            else:
                rabs=0.00001
        phi_list[i]=math.asin(z/rabs)

        if np.abs(x)<0.00001:
            if x<0:
                x=-0.00001
            else:
                x=0.00001
       
        if (x>0):
            Lambda_list[i]=math.atan(y/x)
        elif(y>0):
            Lambda_list[i]=math.atan(y/x)+math.pi
        else:
             Lambda_list[i]=math.atan(y/x)-math.pi

        theta_earth=np.deg2rad(ug0)+n_earth*i*DT

        Lambda_list[i]=Lambda_list[i]-theta_earth
        Lambda_list[i] = (Lambda_list[i] + math.pi) % (2 * math.pi) - math.pi 
         
    
    
    return Lambda_list, phi_list
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def make_groundtrack_lines(lambda_data,phi_data):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function separates the groundtrack data into 
#  continuous lines for plotting purposes
#
#  input:
#  phi_list    - list of latitudes at each time-step  
#  lambda_list - list of longitudes at each time-step
#
#  output:
#  curves_phi    - continuous latitude curves
#  curves_lambda - continuous longitude curves
#
#  Functions required: none
# ----------------------------------------------
    tol = np.pi
    curves_lambda = []
    curves_phi = []
    current_line_lambda = []
    current_line_phi = []
    for i in range(len(phi_data)):
        if i == 0:
            pass
        elif abs(lambda_data[i]-lambda_data[i-1]) > tol:
            curves_lambda.append(current_line_lambda)
            curves_phi.append(current_line_phi)
            current_line_lambda = []
            current_line_phi = []
        current_line_lambda.append(lambda_data[i])
        current_line_phi.append(phi_data[i])
    curves_lambda.append(current_line_lambda)
    curves_phi.append(current_line_phi)
    return curves_lambda, curves_phi
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def interplanetary(planet1,planet2,jd_dep,jd_arr):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function computes the hyperbolic excess velovicity at 
#  the departure and arrival for a transfer between planet 1 and 2
#  with specific departure and arrival dates.
#
#  input:
#  planet1  - departure planet
#  planet2  - arrival planet
#  jd_dep   - departure day in jd
#  jd_arr   - arrival day in jd
#
#  output:
#  nvinfD   - norm of the departure v_infinity
#  nvinfA   - norm of the arrival v_infinity
#
#  Functions required: lambert, planet_elements_and_sv
# ----------------------------------------------
    mu = 1.327124e11
    
    r1,v1,coe1=planet_elements_and_sv(planet1,jd_dep) #position, velocity vectors and coe of the departure planet
    r2,v2,coe2=planet_elements_and_sv(planet2,jd_arr) #position, velocity vectors and coe of the arrival planet
    
    #...Use of the lambert solver to find the spacecraft's velocity at
    #   departure and arrival, assuming a prograde trajectory:
    
    tof=(jd_arr-jd_dep)*86400 #time of flight in seconds
    vD,vA=lambert(mu,r1,r2,tof)
   
    
    #...Equations 8.94 and 8.95:
    nvinfD=np.linalg.norm(vD-v1)
    nvinfA=np.linalg.norm(vA-v2)
    
    return nvinfD,nvinfA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def planetary_elements(planet_id):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    This function extracts a planet's J2000 orbital elements and
#    centennial rates from Table 8.1.
#
#    input:
#    planet_id - 0 through 8, for Mercury through Pluto
#
#    output:
#    J2000_coe - vector of J2000_elements corresponding to "planet_id", with au converted to km
#    rates - row vector of cent_rates corresponding to "planet_id", with au converted to km and arcseconds converted to degrees
#
#  Functions required: none
#--------------------------------------------------------------------
    J2000_elements = np.array([[ 0.38709893, 0.20563069, 7.00487, 48.33167, 77.45645, 252.25084],
                               [ 0.72333199, 0.00677323, 3.39471, 76.68069, 131.53298, 181.97973],
                               [ 1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435],
                               [ 1.52366231, 0.09341233, 1.85061, 49.57854, 336.04084, 355.45332],
                               [ 5.20336301, 0.04839266, 1.30530, 100.55615, 14.75385, 34.40438],
                               [ 9.53707032, 0.05415060, 2.48446, 113.71504, 92.43194, 49.94432],
                               [ 19.19126393, 0.04716771, 0.76986, 74.22988, 170.96424, 313.23218],
                               [ 30.06896348, 0.00858587, 1.76917, 131.72169, 44.97135, 304.88003],
                               [ 39.48168677, 0.24880766, 17.14175, 110.30347, 224.06676, 238.92881]])
    
    cent_rates = np.array([[ 0.00000066, 0.00002527, -23.51, -446.30, 573.57 ,538101628.29],
                           [ 0.00000092, -0.00004938, -2.86, -996.89, -108.80, 210664136.06],
                           [-0.00000005, -0.00003804, -46.94, -18228.25, 1198.28, 129597740.63],
                           [-0.00007221, 0.00011902 ,-25.47, -1020.19, 1560.78, 68905103.78],
                           [ 0.00060737, -0.00012880, -4.15, 1217.17, 839.93 ,10925078.35],
                           [-0.00301530, -0.00036762, 6.11 ,-1591.05, -1948.89, 4401052.95],
                           [ 0.00152025, -0.00019150, -2.09, -1681.4, 1312.56, 1542547.79],
                           [ -0.00125196, 0.00002514, -3.64 ,-151.25, -844.43, 786449.21],
                           [-0.00076912, 0.00006465, 11.07, -37.33, -132.25 ,522747.90]])
    
    J2000_coe = J2000_elements[planet_id,:]
    rates = cent_rates[planet_id,:]
    #...Convert from AU to km:
    au = 149597871
    J2000_coe[0] = J2000_coe[0]*au
    rates[0] = rates[0]*au
    #...Convert from arcseconds to fractions of a degree:
    rates[2:6] = rates[2:6]/3600
    return J2000_coe, rates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def planet_elements_and_sv(planet_id, jd):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  This function calculates the orbital elements and the state  
#  vector of a planet from the date in Julian days.
#
#  input:  
#  planet_id - planet identifier:
#               0 = Mercury
#               1 = Venus
#               2 = Earth
#               3 = Mars
#               4 = Jupiter
#               5 = Uranus
#               7 = Neptune
#               8 = Pluto
#  jd - epoch to retrieve the planet's state vector in julian days
#
#  output:
#  r,v - position [km] and velocity [km/s] vector of the planet with respect
#        to a heliocentric reference frame
#  coe       - vector of heliocentric orbital elements
#              [a  e  incl  RA  w  TA ],
#              where
#               a     = semimajor axis                      (km)
#               e     = eccentricity
#               incl  = inclination                         
#               RA    = right ascension                     
#               w     = argument of perihelion              
#               TA    = true anomaly                        
# 
#  Functions required: M2E, sv_from_coe, planetary_elements
# --------------------------------------------------------------------
    mu = 1.327124e11
    
    #...Obtain the data for the selected planet from Table 8.1:
    [J2000_coe, rates] = planetary_elements(planet_id)

    #...Equation 8.93a:
    t0     = (jd - 2451545)/36525

    #...Equation 8.93b:
    elements = J2000_coe + rates*t0

    a      = elements[0];
    e      = elements[1];
    #...Reduce the angular elements to within the range 0 - 360 degrees:
    incl   = elements[2];
    RA     = np.mod(elements[3],360);
    w_hat  = np.mod(elements[4],360);
    L      = np.mod(elements[5],360);
    w      = np.mod(w_hat - RA ,360);
    M      = np.mod(L - w_hat  ,360);

    #...Algorithm 3.1 (for which M must be in radians)
    E      = M2E(e, np.deg2rad(M))  #rad

    #...Equation 3.13 (converting the result to degrees):
    TA     = np.mod(2*np.arctan(np.sqrt((1 + e)/(1 - e))*np.tan(E/2)),2*np.pi)
    coe    = [a, e, incl, RA, w, np.rad2deg(TA)]

    #...Algorithm 4.5:
    [r, v] = sv_from_coe(coe, mu)

    return r,v,coe
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~