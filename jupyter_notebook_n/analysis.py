# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: py311-main
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Updated Code:
#
# 1. Calculate Orientation in ELVES and SAGA surveys
#
# 2. Load TNG100-1 Data (using 3 satellite lower mass limits)
#
# 3. Calculate one- and two-sample KS test for number count in both ELVES/SAGA and TNG100
#
# 4. Fit ELVES/SAGA and TNG100 with cosine function, boostrap sample, plot quench fraction
#
# 5. Compare sinusoidal function with constant (mean quench fraction) using BIC and AIC statistic
#
#

# %%
# import libraries, set font, ect.

import matplotlib.pyplot as plt 
# %matplotlib inline 

import astropy as ap
from astropy import units as u
from fractions import Fraction 
import astroquery as aq
import numpy as np
from astropy.table import Table
import array as arr
import pandas as pd
import scipy.stats
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy.table import Table, join

import scipy.interpolate as interp
import scipy
import emcee
import corner
import sys

import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], 
    "mathtext.fontset": "cm",  
    "axes.unicode_minus": False
})


# %% [markdown]
# ## True Angle versus Projected Angle

# %%
N = 100000
R = 1.0

phi = 2 * np.pi * np.random.random(N)
costheta = 2 * np.random.random(N) - 1
u = np.random.random(N)

theta = np.arccos(costheta)
r = R * u**(1/3)

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

true_offset_angle = np.arctan(np.abs(z)/np.sqrt(x**2 + y**2))
apparent_offset_angle = np.arctan(np.abs(z)/np.abs(x))

#plt.scatter(true_offset_angle,apparent_offset_angle,alpha=0.005);

plt.figure(figsize=(6, 5))
plt.hist2d(true_offset_angle,apparent_offset_angle,bins=100,norm=mpl.colors.LogNorm())
plt.xlabel('True offset angle')
plt.ylabel('Apparent offset angle')
plt.colorbar(label='Counts')
plt.show()

# %%
true_deg = np.degrees(true_offset_angle)
app_deg  = np.degrees(apparent_offset_angle)

delta_deg = np.abs(app_deg - true_deg)

true_bins = np.linspace(0, 90, 10) 
bin_centers = 0.5 * (true_bins[:-1] + true_bins[1:])

# %%


delta_max = 90    
delta_bins = np.linspace(0, delta_max, 200)

plt.figure(figsize=(7,5))

for i in range(len(true_bins)-1):
    
    mask = (true_deg >= true_bins[i]) & (true_deg < true_bins[i+1])
    if np.sum(mask) < 1:
        continue  

    hist, edges = np.histogram(delta_deg[mask],bins=delta_bins,density=True)

    cdf = np.cumsum(hist) * np.diff(edges)

    plt.plot(edges[:-1],cdf,label=f'{true_bins[i]:.0f}–{true_bins[i+1]:.0f}°')

#plt.axhline(0.9, color='k', ls='--', lw=1)
#plt.axvline(5, color='k', ls=':', lw=1)

plt.xlabel(r'$|\theta_{\rm app} - \theta_{\rm true}|$ [deg]')
plt.ylabel('Cumulative fraction')
plt.legend(title='True offset angle')
plt.xlim(0, delta_max)
plt.ylim(0, 1.01)
plt.tight_layout()
plt.show()

# %%
# Convert to degrees if not already
true_deg = np.degrees(true_offset_angle)
app_deg  = np.degrees(apparent_offset_angle)

# Absolute difference
delta_deg = np.abs(app_deg - true_deg)

# Fraction within 10 degrees
frac_within_10 = np.sum(delta_deg <= 20) / len(delta_deg)

print(f"Fraction of galaxies with apparent angle within 10° of true: {frac_within_10*100:.3f}")

# %%
# 

inclinations = np.arange(0, 91, 1)
true_deg = np.degrees(true_offset_angle)

plt.figure(figsize=(7,5))

for threshold in [5,10,15,20,25,30]:
    fractions_within_thresh = []
    
    for inc_deg in inclinations:
        inc = np.radians(inc_deg)
        
        # rotate y-z plane by inclination
        x_rot = x
        y_rot = y * np.cos(inc) - z * np.sin(inc)
        z_rot = y * np.sin(inc) + z * np.cos(inc)
        
        # apparent offset angle along projected axes
        app_rot_deg = np.degrees(np.arctan(np.abs(z_rot) / np.abs(x_rot)))
        
        # fraction within threshold degrees
        delta_deg = np.abs(app_rot_deg - true_deg)
        frac = np.sum(delta_deg <= threshold) / len(delta_deg)
        fractions_within_thresh.append(frac)
    
    # plot with label for this threshold
    plt.plot(inclinations, fractions_within_thresh[::-1], label=f'{threshold}°')

plt.xlabel('Inclination Angle of xy Plane [deg.]')
plt.ylabel('Fraction of Apparent Offset within [ ] deg. of True Offset')
plt.xlim(0, 90)
plt.ylim(0, 1)
plt.legend(title='Threshold Angle [deg.]',ncol=3,fancybox=False,edgecolor='k')
plt.show()

# %% [markdown]
# ## SAGA

# %%
# combine table C1 and C3, attach host properties to each satellite

saga_host_path = 'saga-dr3-tableC1.txt'
saga_sats_path = 'saga-dr3-tableC3.txt'
saga_hosts = Table.read(saga_host_path, format='ascii')
saga_sats = Table.read(saga_sats_path, format = 'ascii')
saga_joined = join(saga_sats, saga_hosts, 'HOSTID', 'left',uniq_col_name='{table_name}{col_name}',table_names=['', 'HOST_'])

#print(saga_joined)

# %% [markdown]
# ### Filters and Cuts on SAGA Data

# %%
# subselect the 3 different samples in the SAGA survey: gold, silver, participation

''' 1 = Gold;
    2 = Silver;
    3 = Participation'''

gold_filter = saga_sats['sample']== 1
silver_filter = saga_sats['sample'] == 2
participation_filter = saga_sats['sample'] == 3

# %%
# g-r color selection (use 0.7 as the division)
# where did this choice come from?

gr = np.array(saga_hosts['gr'].data)
blue_filter = gr<0.7
red_filter = gr >0.7


# %%
# plot color distribution 

plt.hist(gr,color='k',histtype='step',linewidth=2)
plt.xlabel('g - r')
plt.ylabel('N')
plt.show()


# %%
# r200 calculation, from Jingyao

def calc_r200(mhalo, do_print=False):

    import astropy.constants as const
    from astropy.cosmology import Planck18 as cosmo
    import astropy.units as u
    import numpy as np
    import sys


    mhalo = mhalo*u.Msun
    delta = 200 ## rho/rho_crit=200

    # calculate r200 with respect to critical density
    rho_c = cosmo.critical_density0
    r200_c = ((3*mhalo/(4.*np.pi*delta*rho_c))**(1./3.)).to(u.kpc)


    # calculate r200 with respect to matter density
    rho_m = cosmo.critical_density0 * cosmo.Om0
    r200_m = ((3*mhalo/(4.*np.pi*delta*rho_m))**(1./3.)).to(u.kpc)
    
    return r200_c.value, r200_m.value

# load Mhalo and calculate R200
hosts_log_mhalo_array = saga_hosts['log(Mhalo)'].data
r200c,r200m = calc_r200((10**hosts_log_mhalo_array))

# plot halo mass distribution

plt.hist(hosts_log_mhalo_array,color='k',histtype='step',linewidth=2)
plt.xlabel(r'$\mathrm{logM_{h,central}/M_\odot}$')
plt.ylabel('N')
plt.show()

# %% [markdown]
# ### Calculating Azimuthal Angle from Major Axis

# %%
# print data 

saga_sats

# %%
inclinations_hosts = np.cos(saga_hosts['ba'])

# %%
# calculate angle

saga_PA = np.zeros(len(saga_sats))
saga_quenched = saga_sats['quenched'].data
for m in range(len(saga_hosts)):
    h_c = SkyCoord(saga_hosts['RAdeg'][m],saga_hosts['DEdeg'][m], frame = "icrs", unit = "deg")
    
    for n in range(len(saga_sats)):

        if saga_hosts['HOSTID'][m] == saga_sats['HOSTID'][n]:

            PA = saga_hosts['PA'][m]
            s_c = SkyCoord(saga_sats['RAdeg'][n],saga_sats['DEdeg'][n], frame = "icrs", unit = "deg")
            relative_angle = h_c.position_angle(s_c).degree
            #print(relative_angle)
            relative_angle_corrected = 90 + relative_angle
            #PA_central = customSimbad.query_object(saga_hosts['HOSTID'][m])['GALDIM_ANGLE'].value
            #print((90 - PA) + relative_angle)
            angle_final = ((90 - PA) + relative_angle_corrected)%360
            saga_PA[n] = angle_final

# %%
# plot angle distribution

plt.hist(saga_PA,color='k',histtype='step',linewidth=2)
plt.xlabel(r'Projected Offset Angle [deg.]')
plt.ylabel('N')
plt.show()


# %%
# convert to 0-90

def map_to_0_90(angles):

    angles = np.mod(angles, 360)
    return np.where(angles <= 90, angles,  
           np.where(angles <= 180, 180 - angles,  
           np.where(angles <= 270, angles - 180,  
                    360 - angles))) 

# plot angle distribution

plt.hist(map_to_0_90(saga_PA),color='k',histtype='step',linewidth=2)
plt.xlabel(r'Projected Offset Angle [deg.]')
plt.ylabel('N')
plt.show()

# %%
# apply radius limit 

#lmc_limit = saga_sats_gmag < 19

r200c_hosts = np.zeros(len(saga_sats))
r200m_hosts = np.zeros(len(saga_sats))

sats_rhost_array = saga_sats['Rhost'].data

host_data = {saga_hosts['HOSTID'][i]: hosts_log_mhalo_array[i] for i in range(len(saga_hosts))}

# Iterate over the satellites in saga_sats, and match each with its host using HOSTID
for i in range(len(saga_sats)):
    host_id = saga_sats['HOSTID'][i]  # Get the host ID of the current satellite
    # Check if the host exists in saga_hosts
    if host_id in host_data:
        host_mhalo = host_data[host_id]  # Get the host distance 
        mhalo = 10**host_mhalo #since the mhalo is given as the logarithm
        r200c,r200m = calc_r200(mhalo)
        r200c_hosts[i] = r200c
         
radius_limit = sats_rhost_array < r200c_hosts

# %%
# make a few different subselections:

sample = np.array(saga_sats['sample'])

index1 = (sats_rhost_array < r200c_hosts) & (sample == 1.)
index2 = (sats_rhost_array < r200c_hosts) & (sample == 2.)
index3 = (sats_rhost_array < r200c_hosts) & (sample == 3.)

# %%
# compare entire SAGA sample with different subselections 

plt.hist(map_to_0_90(saga_PA),color='k',histtype='step',linewidth=2, density = True, bins = 18, label = r'Full SAGA Sample (N = %i)'%len(saga_PA))
plt.hist(map_to_0_90(saga_PA[radius_limit]),color='darkred',histtype='step',linewidth=2, density = True, bins = 18, label = r'$\mathrm{<\,R_{200c}}$ (N = %i)'%len(saga_PA[radius_limit]))

plt.xlabel(r'Projected Offset Angle [deg.]')
plt.ylabel('N (Norm.)')
plt.legend(loc='upper right', fancybox = False)

plt.ylim(0,0.020)

plt.show()

# %%
# compare entire SAGA sample with different subselections 

#plt.hist(map_to_0_90(saga_PA),color='k',histtype='step',linewidth=2, density = True, bins = 18, label = r'Full SAGA Sample (N = %i)'%len(saga_PA))
plt.hist(map_to_0_90(saga_PA[index1]),color='goldenrod',histtype='step',linewidth=2, density = True, bins = 18, label = r'$\mathrm{<\,R_{200c}}$ + Gold Sample (N = %i)'%len(saga_PA[index1]))
plt.hist(map_to_0_90(saga_PA[index2]),color='silver',histtype='step',linewidth=2, density = True, bins = 18, label = r'$\mathrm{<\,R_{200c}}$ + Silver Sample (N = %i)'%len(saga_PA[index2]))
plt.hist(map_to_0_90(saga_PA[index3]),color='k',histtype='step',linewidth=2, density = True, bins = 18, label = r'$\mathrm{<\,R_{200c}}$ + Participation Sample (N = %i)'%len(saga_PA[index3]))

plt.xlabel(r'Projected Offset Angle [deg.]')
plt.ylabel('N (Norm.)')
plt.legend(loc='upper right', fancybox = False)

plt.ylim(0,0.050)

plt.show()

# %%
# compare the different SAGA samples (based on satellite stellar mass)

saga_mstar = saga_sats['log(M*)']

plt.hist(saga_mstar[index1],color='goldenrod',histtype='step',linewidth=2, density =False, label = r'$\mathrm{<\,R_{200c}}$ + Gold Sample (N = %i)'%len(saga_mstar[index1]))
plt.hist(saga_mstar[index2],color='silver',histtype='step',linewidth=2, density =False, label = r'$\mathrm{<\,R_{200c}}$ + Silver Sample (N = %i)'%len(saga_mstar[index2]))
plt.hist(saga_mstar[index3],color='k',histtype='step',linewidth=2, density =False, label = r'$\mathrm{<\,R_{200c}}$ + Participation Sample (N = %i)'%len(saga_mstar[index3]))

plt.xlabel(r'$\mathrm{log\, M_{\star,sat.}/M_\odot}$')
plt.ylabel('N')
plt.legend(loc='upper right', fancybox = False)

#plt.ylim(0,0.050)

plt.show()

# %%
quenched = np.array(saga_sats['quenched'])

q_frac, bin_edges, binnumber = scipy.stats.binned_statistic(saga_mstar,quenched,statistic='mean',bins=7,range=[6,10])
plt.scatter((bin_edges[:-1]+bin_edges[1:])/2,q_frac,color='black', label = r'Full SAGA Sample (N = %i)'%len(saga_PA))

q_frac, bin_edges, binnumber = scipy.stats.binned_statistic(saga_mstar[radius_limit],quenched[radius_limit],statistic='mean',bins=7,range=[6,10])
plt.scatter((bin_edges[:-1]+bin_edges[1:])/2,q_frac,color='darkred', label = r'$\mathrm{<\,R_{200c}}$ (N = %i)'%len(saga_PA[radius_limit]))

plt.axvline(7.5,ls='--',color='goldenrod',label='Gold Sample')
plt.axvline(6.75,ls='--',color='silver',label='Silver Sample')
 
plt.xlabel(r'$\mathrm{log\, M_{\star,sat.}/M_\odot}$')
plt.ylabel('N')
plt.legend(loc='upper right', fancybox = False)
plt.ylim(0,1)
plt.show()

# %%
# there are 4 hosts with no satellites
# nsat-GSPc = Number of confirmed satellites in all three samples: Gold, Silver, Participation

plt.hist(saga_hosts['nsat-GSPc'],bins=20,color='k')
plt.xlabel(r'Number of confirmed satellites in all three samples')
plt.ylabel('N')

plt.show()

print(np.array(saga_hosts['nsat-GSPc'])[np.array(saga_hosts['nsat-GSPc']) == 0])

# %%
# there are 16 hosts in the Gold sample with no Gold sample mass (logM > 7.5) satellites 

len(saga_hosts['nsat-Gc'][saga_hosts['nsat-Gc'] == 0])

# %%
# Gold sample completeness

plt.hist(saga_hosts['nsat-Gc']/saga_hosts['nsat-G'],bins=50,color='k')
plt.xlabel(r'Gold Sample Completeness')
plt.ylabel('N')

plt.show()

# %%
import astropy.units as u

# %%
hostid, idx = np.unique(saga_joined['HOSTID'],return_index=True)
host_ba = saga_joined['HOST_ba'][idx]
host_inclination = (np.arccos(host_ba)*u.rad).to('deg')

plt.hist(host_inclination,color='k',histtype='step', bins = 18, linewidth=2, density =False)

plt.xlabel(r'Host Inclination, $i$ [deg.]')
plt.ylabel('N')
#plt.legend(loc='upper right', fancybox = False)
#plt.ylim(0,1)
plt.show()

# %%
host_ba = saga_joined['HOST_ba']
host_inclination = np.array((np.arccos(host_ba)*u.rad).to('deg'))

index_inclination = (host_inclination >= 35)

index_inclination_ids = np.array(saga_joined['HOSTID'])[index_inclination]

# %%
# make a few different subselections:

index_inclination_ids = np.array(saga_joined['HOSTID'])[index_inclination]

HOSTID = np.array(saga_sats['HOSTID'])

index1 = (sats_rhost_array < r200c_hosts) & ((sample == 1.) | (sample == 2.))  & (np.isin(HOSTID,index_inclination_ids))
#index2 = (sats_rhost_array < r200c_hosts) & (sample == 2.)
#index3 = (sats_rhost_array < r200c_hosts) & (sample == 3.)

# %%
print('number of SAGA hosts =',len(np.unique(saga_sats['HOSTID'][index1])))
print('number of SAGA satellites =',len(np.unique(saga_PA[index1])))

# %%
# compare entire SAGA sample with different subselections 

plt.hist(map_to_0_90(saga_PA[index1]),color='goldenrod',histtype='step',linewidth=2, density = False, bins = 18, label = r'$\mathrm{<\,R_{200c}}$ + Gold Sample + $i > 0^\circ$(N = %i)'%len(saga_PA[index1]))
#plt.hist(map_to_0_90(saga_PA[index2]),color='silver',histtype='step',linewidth=2, density = True, bins = 18, label = r'$\mathrm{<\,R_{200c}}$ + Silver Sample (N = %i)'%len(saga_PA[index2]))
#plt.hist(map_to_0_90(saga_PA[index3]),color='k',histtype='step',linewidth=2, density = True, bins = 18, label = r'$\mathrm{<\,R_{200c}}$ + Participation Sample (N = %i)'%len(saga_PA[index3]))

plt.xlabel(r'Projected Offset Angle [deg.]')
plt.ylabel('N')
plt.legend(loc='upper right', fancybox = False)

plt.ylim(0,20)

plt.show()

# %%
# final sample, where we only select satellites within r200c of the host galaxy

#saga_PA = saga_PA[radius_limit]
#saga_quenched = saga_quenched[radius_limit]

saga_PA = saga_PA[index1]
saga_90 = map_to_0_90(saga_PA)
saga_quenched = saga_quenched[index1]

# %% [markdown]
# ## Importing and Processing ELVES

# %%
# import ELVES data 

sats = pd.read_csv('Carlsten22_ELVES_confirmed_sats - Carlsten22_ELVES_confirmed_sats_compre.csv')
hosts_unfiltered = pd.read_csv("Carlsten22_ELVES_host - Carlsten22_ELVES_host.csv", skipfooter= 1, engine='python')
s_host = pd.read_csv("saga_host_compre.csv")

# %%
hosts_unfiltered

# %%
sats

# %%
from astroquery.simbad import Simbad

customSimbad = Simbad()
customSimbad.reset_votable_fields()
# Add degree-valued RA/DEC + host PA
customSimbad.add_votable_fields('ra', 'dec', 'galdim_angle','galdim_majaxis','galdim_minaxis')


# %%
hosts_unfiltered['Host']

# %%
# get position angle from ELVES hosts 

hosts_missing_pa = []
pa_val = []
elves_inclination = []
elves_PA = []
pa_indices = []

for i, host in enumerate(hosts_unfiltered['Host']):
    tab = customSimbad.query_object(host)

    val = tab['galdim_angle']
    major_axis = tab['galdim_majaxis']
    minor_axis = tab['galdim_minaxis']
    
    #print(val[0])
    pa_val.append(val.value)
    if np.ma.is_masked(val):
        hosts_missing_pa.append(host)
    else:
        pa_indices.append(i)
        print(hosts_unfiltered['Host'][i])
        print("major_axis:", major_axis)
        print("minor_axis:", minor_axis)
        #print(' d')
        inclination_angle = np.arccos((minor_axis)/(major_axis))*180/np.pi
        print("inclination_angle:", inclination_angle)
        elves_inclination.append(inclination_angle)
        elves_PA.append(val)

hosts  = hosts_unfiltered.iloc[pa_indices].copy()
#print("Hosts missing GALDIM_ANGLE:")
#for h in hosts_missing_pa:
    #print(h)
    #print(customSimbad.query_object(h)['galdim_angle'])


# %% [markdown]
# ### We update sats table to reflect the valid PA hosts

# %%
sats_corrected = sats[sats['Host'].isin(hosts['Host'])].copy()

# %%
"""NGC1023, 90  # https://ned.ipac.caltech.edu/byname?objname=NGC1023&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
NGC3379, spherical: 90 or 0  # https://ned.ipac.caltech.edu/byname?objname=NGC3379&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
NGC4258, 150 # https://ned.ipac.caltech.edu/byname?objname=NGC4258&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
NGC5236, face-on: 0 # https://ned.ipac.caltech.edu/byname?objname=NGC5236&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1
NGC5457, also face-on # https://ned.ipac.caltech.edu/byname?objname=NGC5457&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1"""

# %%
plt.hist(np.array(elves_inclination), linewidth=2, density =False)
plt.show()

# %%
# plot ELVES host inclination

elves_inclination_reformat = np.concatenate([c.filled(np.nan) for c in elves_inclination])
elves_PA_reformat = np.concatenate([c.astype(float).filled(np.nan) for c in elves_PA])

plt.hist(elves_inclination_reformat,color='k',bins=18)
plt.xlabel("ELVES Host Inclination [deg.]")
plt.ylabel("N")
plt.show()

# %%
plt.scatter(elves_inclination_reformat,elves_PA_reformat,color='k')
plt.xlabel("ELVES Host Inclination [deg.]")
plt.ylabel("N")
plt.show()

# %%
import numpy as np
from astropy.coordinates import SkyCoord

pa_list, dproj_list, mass_list = [], [], []
host_list = []
kept_indices = []  # optional: lets you map back to rows in `sats`

for r in range(len(sats_corrected)):
    host = sats_corrected['Host'].get(r)
    # Query host once each place we need
    h_table = Simbad.query_object(host)
    pa_tab  = customSimbad.query_object(host)


    # Skip if PA is missing/masked
    pa_col = pa_tab['galdim_angle']
    if np.ma.is_masked(pa_col):
        continue

    # Build coords (skip if satellite coords missing)
    try:
        h_coords = SkyCoord(ra=h_table['ra'], dec=h_table['dec'], frame='icrs')
        s_coords = SkyCoord(ra=sats['RA(deg)'].get(r),
                            dec=sats['DEC(deg)'].get(r),
                            frame='icrs', unit='deg')
    except Exception:
        # If any coord is malformed, skip this row
        continue

    position_angle = h_coords.position_angle(s_coords).degree

    # Compute the angle using GALDIM_ANGLE value
    angle = ((90 - pa_col.value) + position_angle + 90) % 360

    # Append valid values
    pa_list.append(angle)
    dproj_list.append(sats['Rproj(kpc)'].get(r))
    mass_list.append(sats['lgM_star'].get(r))
    kept_indices.append(r)

# Final arrays contain ONLY rows with valid PA
elves_PA    = np.array(pa_list, dtype=float)
elves_dproj = np.array(dproj_list, dtype=float)
elves_mass  = np.array(mass_list, dtype=float)

# %%
print("len(hosts) =", len(hosts))
print("len(elves_inclination_reformat) =", len(elves_inclination_reformat))
print("same indices:", hosts.index.equals(hosts_unfiltered.iloc[pa_indices].index))


# %%
print(len(set(host_list)))

# %%
print(hosts['Host'])

# %%
hosts["Inclination"] = np.asarray(elves_inclination_reformat, float)


# %%
#elves position angles

n_theta = np.zeros(len(elves_PA))  
    
for n in range(len(elves_PA)):
    angle_val = float(elves_PA[n])  # Convert to scalar to avoid deprecation warning
    if angle_val > 180:
        n_theta[n] = (180 - (angle_val % 180))
    else: 
        n_theta[n] = angle_val
        
#sats_filter =  (sats['lgM_star']<(10**10))&(sats['Rproj_in_Rvir?']==1)


# %%
#elves quench values
elves_quenched = sats['ETG?'].iloc[kept_indices]
elves_PA = elves_PA

# %%
print('number of ELVES hosts =',len(hosts))
print('number of ELVES satellites =',len(elves_PA))

# %%
# Apply inclination >= 35 deg filter to ELVES satellites (by host inclination)
# `hosts` contains one row per host with a measured PA; `hosts['Inclination']` is in degrees.

host_incl_map = dict(zip(np.asarray(hosts['Host']), np.asarray(hosts['Inclination'], float)))
elves_host_name_per_sat = np.asarray(sats_corrected['Host'].iloc[kept_indices])
elves_host_incl_per_sat = np.array([host_incl_map.get(h, np.nan) for h in elves_host_name_per_sat], dtype=float)

elves_keep = np.isfinite(elves_host_incl_per_sat) & (elves_host_incl_per_sat >= 35)

# Filter the satellite-level arrays + kept_indices used downstream
elves_PA = np.asarray(elves_PA, dtype=float)[elves_keep]
elves_quenched = np.asarray(elves_quenched)[elves_keep]
kept_indices = np.asarray(kept_indices, dtype=int)[elves_keep]

print('ELVES satellites after i>=35 cut =', elves_keep.sum())


# %%

# %%
saga_index = index1


# %%

# %% [markdown]
# # Working with Both ELVES and SAGA

# %%

# %% [markdown]
# ### Quench Fraction Calculation

# %%

def bootstrap_error(data, n_bootstrap=10000):
    if len(data) == 0:
        return 0.0
    bootstraps = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    means = np.nanmean(bootstraps, axis=1)
    return np.std(means)
    
def quenched_fraction_with_bootstrap(angles, quenching, bin_size, window_size, angle_range=(0, 90), n_bootstrap=10000):
    
    angles = np.asarray(angles).flatten()
    quenching = np.asarray(quenching, dtype=float).flatten()
    
    centers = [] 
    means = []
    errors = []

    center_value = angle_range[0] + window_size
    
    while center_value <= angle_range[1] - window_size:

        # Define window: angles within [center - window_size, center + window_size)
        in_window = (angles >= center_value - window_size) & (angles < center_value + window_size)
        values_in_window = quenching[in_window]

        # Store center
        centers.append(center_value)

        # Compute moving mean and bootstrap error
        means.append(np.nanmean(values_in_window))
        errors.append(bootstrap_error(values_in_window, n_bootstrap=n_bootstrap))

        # Move to next window center
        center_value += bin_size
        
    return np.array(centers), np.array(means), np.array(errors)


# %%

def bootstrap_error(data, n_bootstrap=10000):
    if len(data) == 0:
        return 0.0
    bootstraps = np.random.choice(data, size=(n_bootstrap, len(data)), replace=True)
    means = np.nanmean(bootstraps, axis=1)
    return np.std(means)
    
def quenched_fraction_with_bootstrap(angles, quenching, bin_size, window_size, angle_range=(0, 90), n_bootstrap=10000):
    
    angles = np.asarray(angles).flatten()
    quenching = np.asarray(quenching, dtype=float).flatten()
    
    centers = [] 
    means = []
    errors = []

    center_value = angle_range[0] + 2.5
    
    while center_value < 90:

        # Define window: angles within [center - window_size, center + window_size)
        in_window = (angles >= center_value - 2.5) & (angles < center_value + 2.5)
        values_in_window = quenching[in_window]

        # Store center
        centers.append(center_value)

        # Compute moving mean and bootstrap error
        means.append(np.nanmean(values_in_window))
        #print(len(values_in_window))
        errors.append(bootstrap_error(values_in_window, n_bootstrap=n_bootstrap))

        # Move to next window center
        center_value += 5
        
    return np.array(centers), np.array(means), np.array(errors)


# %%
#saga data
saga_bins, saga_quench, saga_error = quenched_fraction_with_bootstrap(saga_90, saga_quenched, 5, 5)

# %%
# elves data
elves_90 = map_to_0_90(elves_PA)
elves_bins, elves_quench, elves_error = quenched_fraction_with_bootstrap(elves_90, elves_quenched,5,5)
elves_90_correct = []
for i in range(len(elves_90)):
    elves_90_correct.append(elves_90[i][0])

# %%
elves_mean = np.mean(elves_quenched)

# %%
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']  # Or 'Times'
plt.rcParams['mathtext.fontset'] = 'stix'  # For math symbols to match Times
plt.rcParams['mathtext.rm'] = 'Times New Roman'

# %% [markdown]
# ## Normalized Count per Azimuthal Angle Bin

# %%
# Load angle bin centers and TNG100 number counts (used in later plots and fits)

angle_array = np.loadtxt('angle_array.txt')

tng100_number_count = np.loadtxt('tng100_number_count_1e8.txt')
tng100_number_count_1e7 = np.loadtxt('tng100_number_count_1e7.txt')
tng100_number_count_1e6 = np.loadtxt('tng100_number_count_1e6.txt')

tng_hist  = tng100_number_count / np.sum(tng100_number_count) / 5

tng_hist_1e7  = tng100_number_count_1e7 / np.sum(tng100_number_count_1e7) / 5
tng_hist_1e6  = tng100_number_count_1e6 / np.sum(tng100_number_count_1e6) / 5

bin_width = np.diff(angle_array).mean()
edges = np.concatenate([angle_array - bin_width/2, [angle_array[-1] + bin_width/2]])


# %% [markdown]
# ### Mean Angle 

# %%
# import TNG100 data

df_tng100_1e8 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_mstar_1e8.csv")
df_tng100_host_mh_1e8 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_host_mh_1e8.csv")
df_tng100_alpha_1e8 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_alpha_1e8.csv")

index_1e8 = (df_tng100_host_mh_1e8['host_mh'] > 12) & (df_tng100_host_mh_1e8['host_mh'] < 12.5)

df_tng100_1e7 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_mstar_1e7.csv")
df_tng100_host_mh_1e7 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_host_mh_1e7.csv")
df_tng100_alpha_1e7 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_alpha_1e7.csv")

index_1e7 = (df_tng100_host_mh_1e7['host_mh'] > 12) & (df_tng100_host_mh_1e7['host_mh'] < 12.5)

df_tng100_1e6 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_mstar_1e6.csv")
df_tng100_host_mh_1e6 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_host_mh_1e6.csv")
df_tng100_alpha_1e6 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_alpha_1e6.csv")

index_1e6 = (df_tng100_host_mh_1e6['host_mh'] > 12) & (df_tng100_host_mh_1e6['host_mh'] < 12.5)

# %%
print(np.mean(df_tng100_alpha_1e8[index_1e8]),np.std(df_tng100_alpha_1e8[index_1e8]))
print(np.mean(df_tng100_alpha_1e7[index_1e7]),np.std(df_tng100_alpha_1e7[index_1e7]))
print(np.mean(df_tng100_alpha_1e6[index_1e6]),np.std(df_tng100_alpha_1e6[index_1e6]))

# %% [markdown]
# ### One-Sample KS-Test

# %%
sort_saga = np.argsort(saga_90)
sort_elves = np.argsort(np.array(elves_90_correct))

cdf_saga = np.arange(1, len(saga_90)+1) / len(saga_90)
cdf_elves = np.arange(1, len(elves_90_correct)+1) / len(elves_90_correct)

plt.plot(saga_90[sort_saga], cdf_saga, color=saga_line, label='SAGA')
plt.plot(np.array(elves_90_correct)[sort_elves], cdf_elves, color=elves_line, label='ELVES')

angles_1d_1e8 = df_tng100_alpha_1e8.iloc[:,0].to_numpy()
sort_tng100_1e8 = np.argsort(angles_1d_1e8)
cdf_tng100_1e8 = np.arange(1, len(angles_1d_1e8)+1) / len(angles_1d_1e8)
plt.plot(angles_1d_1e8[sort_tng100_1e8], cdf_tng100_1e8, color='black', ls = '-', label='TNG 1e8')

angles_1d_1e7 = df_tng100_alpha_1e7.iloc[:,0].to_numpy()
sort_tng100_1e7 = np.argsort(angles_1d_1e7)
cdf_tng100_1e7 = np.arange(1, len(angles_1d_1e7)+1) / len(angles_1d_1e7)
plt.plot(angles_1d_1e7[sort_tng100_1e7], cdf_tng100_1e7, color='black', ls = '--', label='TNG 1e7')

angles_1d_1e6 = df_tng100_alpha_1e6.iloc[:,0].to_numpy()
sort_tng100_1e6 = np.argsort(angles_1d_1e6)
cdf_tng100_1e6 = np.arange(1, len(angles_1d_1e6)+1) / len(angles_1d_1e6)
plt.plot(angles_1d_1e6[sort_tng100_1e6], cdf_tng100_1e6, color='black', ls = ':', label='TNG 1e6')

# Plot

plt.xlabel(r'$\theta$ [deg.]')
plt.ylabel('CDF')
plt.legend(fancybox=False,edgecolor='k',ncol=1)
plt.show()

# %%
# implement one-sample K-S test
# see https://www.astroml.org/astroML-notebooks/chapter4/astroml_chapter4_Comparison_of_distributions.html

import scipy.stats as stats

# example from website
#np.random.seed(0)
#vals = np.random.normal(loc=0, scale=1, size= 1000)
#print(f'Normal: {stats.kstest(vals, "norm")}')
#print(f'Uniform: {stats.kstest(vals, "uniform")}')

print(f'Uniform: {stats.kstest(saga_90[sort_saga], "uniform",args=(0, 90))}')
print(f'Uniform: {stats.kstest(np.array(elves_90_correct)[sort_elves], "uniform",args=(0, 90))}')


print(f'Uniform: {stats.kstest(angles_1d_1e8[sort_tng100_1e8], "uniform",args=(0, 90))}')
print(f'Uniform: {stats.kstest(angles_1d_1e7[sort_tng100_1e7], "uniform",args=(0, 90))}')
print(f'Uniform: {stats.kstest(angles_1d_1e6[sort_tng100_1e6], "uniform",args=(0, 90))}')

# %%
from scipy import stats

# %%
# calulate how many > 1e6 satellites are needed to find a significant difference from a uniform distribution

rng = np.random.default_rng(42) # set random seed

angles = df_tng100_alpha_1e6.iloc[:, 0].to_numpy() # > 1e6 satellite population

Ns = np.arange(10, 2000, 100)
n_boot = 10000

pval_median_1e6 = []

pval_lo1_1e6 = []
pval_hi1_1e6 = []

pval_lo2_1e6 = []
pval_hi2_1e6 = []

pval_lo3_1e6 = []
pval_hi3_1e6 = []

one_sigma_low = (1-0.682689492137086)/2
one_sigma_high = 1-(1-0.682689492137086)/2

two_sigma_low = (1-0.954499736103642)/2
two_sigma_high = 1-(1-0.954499736103642)/2

three_sigma_low = (1-0.997300203936740)/2
three_sigma_high = 1-(1-0.997300203936740)/2

for N in Ns:
    
    pvals_N = []

    for _ in range(n_boot):

        subsample = rng.choice(angles, size=N, replace=True)

        p = stats.kstest(subsample, "uniform", args=(0, 90)).pvalue
        pvals_N.append(p)

    pvals_N = np.array(pvals_N)

    pval_median_1e6.append(np.median(pvals_N))
    
    pval_lo1_1e6.append(np.percentile(pvals_N, one_sigma_low*100))
    pval_hi1_1e6.append(np.percentile(pvals_N, one_sigma_high*100))
    pval_lo2_1e6.append(np.percentile(pvals_N, two_sigma_low*100))
    pval_hi2_1e6.append(np.percentile(pvals_N, two_sigma_high*100))
    pval_lo3_1e6.append(np.percentile(pvals_N, three_sigma_low*100))
    pval_hi3_1e6.append(np.percentile(pvals_N, three_sigma_high*100))

# %%
#

rng = np.random.default_rng(42) # set random seed

angles = df_tng100_alpha_1e7.iloc[:, 0].to_numpy() # > 1e6 satellite population

Ns = np.arange(10, 2000, 100)
n_boot = 10000

pval_median_1e7 = []

pval_lo1_1e7 = []
pval_hi1_1e7 = []

pval_lo2_1e7 = []
pval_hi2_1e7 = []

pval_lo3_1e7 = []
pval_hi3_1e7 = []

for N in Ns:
    
    pvals_N = []

    for _ in range(n_boot):

        subsample = rng.choice(angles, size=N, replace=True)

        p = stats.kstest(subsample, "uniform", args=(0, 90)).pvalue
        pvals_N.append(p)

    pvals_N = np.array(pvals_N)

    pval_median_1e7.append(np.median(pvals_N))
    
    pval_lo1_1e7.append(np.percentile(pvals_N, one_sigma_low*100))
    pval_hi1_1e7.append(np.percentile(pvals_N, one_sigma_high*100))
    pval_lo2_1e7.append(np.percentile(pvals_N, two_sigma_low*100))
    pval_hi2_1e7.append(np.percentile(pvals_N, two_sigma_high*100))
    pval_lo3_1e7.append(np.percentile(pvals_N, three_sigma_low*100))
    pval_hi3_1e7.append(np.percentile(pvals_N, three_sigma_high*100))

# %%
#

rng = np.random.default_rng(42) # set random seed

angles = df_tng100_alpha_1e8.iloc[:, 0].to_numpy() # > 1e6 satellite population

Ns = np.arange(10, 2000, 100)
n_boot = 10000

pval_median_1e8 = []

pval_lo1_1e8 = []
pval_hi1_1e8 = []

pval_lo2_1e8 = []
pval_hi2_1e8 = []

pval_lo3_1e8 = []
pval_hi3_1e8 = []

for N in Ns:
    
    pvals_N = []

    for _ in range(n_boot):

        subsample = rng.choice(angles, size=N, replace=True)

        p = stats.kstest(subsample, "uniform", args=(0, 90)).pvalue
        pvals_N.append(p)

    pvals_N = np.array(pvals_N)

    pval_median_1e8.append(np.median(pvals_N))
    
    pval_lo1_1e8.append(np.percentile(pvals_N, one_sigma_low*100))
    pval_hi1_1e8.append(np.percentile(pvals_N, one_sigma_high*100))
    pval_lo2_1e8.append(np.percentile(pvals_N, two_sigma_low*100))
    pval_hi2_1e8.append(np.percentile(pvals_N, two_sigma_high*100))
    pval_lo3_1e8.append(np.percentile(pvals_N, three_sigma_low*100))
    pval_hi3_1e8.append(np.percentile(pvals_N, three_sigma_high*100))

# %%


Ns = np.arange(10, 2000, 100)
pval_median = np.array(pval_median_1e6)
pval_lo = np.array(pval_lo1_1e6)
pval_hi = np.array(pval_hi1_1e6)

fig, ax = plt.subplots(1, 3, figsize=(20, 5), sharex=True, sharey=True)

ax[0].tick_params(which='both',direction='in',top=True,right=True)
ax[1].tick_params(which='both',direction='in',top=True,right=True)
ax[2].tick_params(which='both',direction='in',top=True,right=True)

ax[0].fill_between(Ns,pval_lo3_1e6,pval_hi3_1e6,color='gray',alpha=0.4)#,label=r'$\pm1\sigma$')
ax[0].fill_between(Ns,pval_lo2_1e6,pval_hi2_1e6,color='gray',alpha=0.4)#,label=r'$\pm2\sigma$')
ax[0].fill_between(Ns,pval_lo1_1e6,pval_hi1_1e6,color='gray',alpha=0.4)#,label=r'$\pm3\sigma$')
ax[0].plot(Ns,pval_median_1e6,color='black',lw=2)#,label='Median KS p-value')

ax[1].fill_between(Ns,pval_lo3_1e7,pval_hi3_1e7,color='gray',alpha=0.4)#,label=r'$\pm1\sigma$')
ax[1].fill_between(Ns,pval_lo2_1e7,pval_hi2_1e7,color='gray',alpha=0.4)#,label=r'$\pm2\sigma$')
ax[1].fill_between(Ns,pval_lo1_1e7,pval_hi1_1e7,color='gray',alpha=0.4)#,label=r'$\pm3\sigma$')
ax[1].plot(Ns,pval_median_1e7,color='black',lw=2)#,label='Median KS p-value')

ax[2].fill_between(Ns,pval_lo3_1e8,pval_hi3_1e8,color='gray',alpha=0.4)#,label=r'$\pm1\sigma$')
ax[2].fill_between(Ns,pval_lo2_1e8,pval_hi2_1e8,color='gray',alpha=0.4)#,label=r'$\pm2\sigma$')
ax[2].fill_between(Ns,pval_lo1_1e8,pval_hi1_1e8,color='gray',alpha=0.4)#,label=r'$\pm3\sigma$')
ax[2].plot(Ns,pval_median_1e8,color='black',lw=2)#,label='Median KS p-value')

ax[0].axhline(0.10, color='r', ls='--', lw=1,label=r'p = 0.10')
ax[0].axhline(0.05, color='r', ls='--', lw=1,label=r'p = 0.05')
ax[1].axhline(0.10, color='r', ls='--', lw=1,label=r'p = 0.10')
ax[1].axhline(0.05, color='r', ls='--', lw=1,label=r'p = 0.05')
ax[2].axhline(0.10, color='r', ls='--', lw=1,label=r'p = 0.10')
ax[2].axhline(0.05, color='r', ls='--', lw=1,label=r'p = 0.05')

ax[0].set_xlim(10,1500)
ax[0].set_ylim(1e-12,1e1)

ax[0].set_yscale('log')

ax[0].set_xlabel(r'$\mathrm{N_{sat.}}$',fontsize=12)
ax[0].set_ylabel('p-value (from one-sample KS test)',fontsize=12)
ax[0].legend(loc='lower left', fancybox=False,edgecolor='k',ncol=3,fontsize=12)
ax[0].grid(True, which='both', alpha=0.3,ls='--')
ax[1].grid(True, which='both', alpha=0.3,ls='--')
ax[2].grid(True, which='both', alpha=0.3,ls='--')

ax[0].set_title(r'$\mathrm{TNG100:\,M_{\star,sat.}/M_\odot>6.0}$')
ax[1].set_title(r'$\mathrm{TNG100:\,M_{\star,sat.}/M_\odot>7.0}$')
ax[2].set_title(r'$\mathrm{TNG100:\,M_{\star,sat.}/M_\odot>8.0}$')

plt.subplots_adjust(wspace=0.05)
plt.show()


# %% [markdown]
# ### Two-Sample KS-Test (+ One-Sample KS-Test for Q and SF subsample)

# %%
#elves_quenched = elves_quenched.to_numpy()
elves_90_correct = np.array(elves_90_correct)

# %%
saga_q = (saga_quenched == 1)
saga_sf = (saga_quenched == 0)
sort_saga_q = np.argsort(saga_90[saga_q])
sort_saga_sf = np.argsort(saga_90[saga_sf])

#sort_elves = np.argsort(np.array(elves_90_correct))

elves_q = (elves_quenched == 1)
elves_sf = (elves_quenched == 0)
sort_elves_q = np.argsort(elves_90_correct[elves_q])
sort_elves_sf = np.argsort(elves_90_correct[elves_sf])

cdf_saga_q = np.arange(1, len(saga_90[saga_q])+1) / len(saga_90[saga_q])
cdf_saga_sf = np.arange(1, len(saga_90[saga_sf])+1) / len(saga_90[saga_sf])

cdf_elves_q = np.arange(1, len(elves_90_correct[elves_q])+1) / len(elves_90_correct[elves_q])
cdf_elves_sf = np.arange(1, len(elves_90_correct[elves_sf])+1) / len(elves_90_correct[elves_sf])

plt.plot(saga_90[saga_q][sort_saga_q], cdf_saga_q, color=saga_line, ls = '-', label='SAGA Quenched')
plt.plot(saga_90[saga_sf][sort_saga_sf], cdf_saga_sf, color=saga_line, ls = '--', label='SAGA Star-Forming')

plt.plot(elves_90_correct[elves_q][sort_elves_q], cdf_elves_q, color=elves_line, ls = '-', label='ELVES Quenched')
plt.plot(elves_90_correct[elves_sf][sort_elves_sf], cdf_elves_sf, color=elves_line, ls = '--', label='ELVES Star-Forming')

# plot isotropic expectation
#theta_ref = np.linspace(0, 90, 200)
#cdf_iso = theta_ref / 90.0
#plt.plot(theta_ref, cdf_iso, color='black', ls='-', lw=2)

plt.xlabel(r'$\theta$ [deg.]')
plt.ylabel('CDF')
plt.legend(fancybox=False,edgecolor='k',ncol=2)
plt.show()

# %%
# two-sample KS test for SAGA/ELVES

print('SAGA:')
print(f'Q against Uniform Dist.: {stats.kstest(saga_90[saga_q][sort_saga_q], "uniform",args=(0, 90))}')
print(f'SF against Uniform Dist.: {stats.kstest(saga_90[saga_sf][sort_saga_sf], "uniform",args=(0, 90))}')
print(' ')
print('ELVES:')
print(f'Q against Uniform Dist.: {stats.kstest(elves_90_correct[elves_q][sort_elves_q], "uniform",args=(0, 90))}')
print(f'SF against Uniform Dist.: {stats.kstest(elves_90_correct[elves_sf][sort_elves_sf], "uniform",args=(0, 90))}')

#print(f'ELVES Q vs. SF: {stats.ks_2samp(elves_90_correct[elves_q][sort_elves_q], elves_90_correct[elves_sf][sort_elves_sf])}')

# %%
# two-sample KS test for SAGA/ELVES

print(f'SAGA Q vs. SF: {stats.ks_2samp(saga_90[saga_q][sort_saga_q], saga_90[saga_sf][sort_saga_sf])}')
print(' ')
print(f'ELVES Q vs. SF: {stats.ks_2samp(elves_90_correct[elves_q][sort_elves_q], elves_90_correct[elves_sf][sort_elves_sf])}')

# %%
# TNG100-1 plot

# 1e8

df_tng100_1e8 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_mstar_1e8.csv")
df_tng100_host_mh_1e8 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_host_mh_1e8.csv")
df_tng100_alpha_1e8 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_alpha_1e8.csv")
tng100_sfr_info_1e8 = pd.read_csv("satellite_1e8/tng100_sfr_info_1e8.csv")

index = (df_tng100_host_mh_1e8['host_mh'] > 12) & (df_tng100_host_mh_1e8['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info_1e8['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info_1e8['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info_1e8['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info_1e8['host_mh'][index]))
q_sf_array[index_sf] = 0.

angles_all = df_tng100_alpha_1e8.iloc[:, 0]
angles_q  = angles_all[index][index_q]
angles_sf = angles_all[index][index_sf]
angles_q_sorted  = np.sort(angles_q)
angles_sf_sorted = np.sort(angles_sf)
cdf_q  = np.arange(1, len(angles_q_sorted)  + 1) / len(angles_q_sorted)
cdf_sf = np.arange(1, len(angles_sf_sorted) + 1) / len(angles_sf_sorted)
plt.plot(angles_q_sorted,  cdf_q,  color='black', ls='-',  label='TNG 1e8 Q')
plt.plot(angles_sf_sorted, cdf_sf, color='black', ls='--', label='TNG 1e8 SF')

# two-sample KS test
print(f'TNG100-1, Msat > 1e8: {stats.ks_2samp(angles_q_sorted, angles_sf_sorted)}')
# one-sample KS test
print(f'Q against Uniform Dist.: {stats.kstest(angles_q_sorted, "uniform",args=(0, 90))}')
print(f'SF against Uniform Dist.: {stats.kstest(angles_sf_sorted, "uniform",args=(0, 90))}')
print(' ')

# 1e7

df_tng100_1e7 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_mstar_1e7.csv")
df_tng100_host_mh_1e7 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_host_mh_1e7.csv")
df_tng100_alpha_1e7 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_alpha_1e7.csv")
tng100_sfr_info_1e7 = pd.read_csv("satellite_1e7/tng100_sfr_info_1e7.csv")

index = (df_tng100_host_mh_1e7['host_mh'] > 12) & (df_tng100_host_mh_1e7['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info_1e7['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info_1e7['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info_1e7['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info_1e7['host_mh'][index]))
q_sf_array[index_sf] = 0.

angles_all = df_tng100_alpha_1e7.iloc[:, 0]
angles_q  = angles_all[index][index_q]
angles_sf = angles_all[index][index_sf]
angles_q_sorted  = np.sort(angles_q)
angles_sf_sorted = np.sort(angles_sf)
cdf_q  = np.arange(1, len(angles_q_sorted)  + 1) / len(angles_q_sorted)
cdf_sf = np.arange(1, len(angles_sf_sorted) + 1) / len(angles_sf_sorted)
plt.plot(angles_q_sorted,  cdf_q,  color='black', ls='-',  label='TNG 1e7 Q')
plt.plot(angles_sf_sorted, cdf_sf, color='black', ls='--', label='TNG 1e7 SF')

# two-sample KS test
print(f'TNG100-1, Msat > 1e7: {stats.ks_2samp(angles_q_sorted, angles_sf_sorted)}')
# one-sample KS test
print(f'Q against Uniform Dist.: {stats.kstest(angles_q_sorted, "uniform",args=(0, 90))}')
print(f'SF against Uniform Dist.: {stats.kstest(angles_sf_sorted, "uniform",args=(0, 90))}')
print(' ')

# 1e6

df_tng100_1e6 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_mstar_1e6.csv")
df_tng100_host_mh_1e6 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_host_mh_1e6.csv")
df_tng100_alpha_1e6 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_alpha_1e6.csv")
tng100_sfr_info_1e6 = pd.read_csv("satellite_1e6/tng100_sfr_info_1e6.csv")

index = (df_tng100_host_mh_1e6['host_mh'] > 12) & (df_tng100_host_mh_1e6['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info_1e6['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info_1e6['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info_1e6['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info_1e6['host_mh'][index]))
q_sf_array[index_sf] = 0.

angles_all = df_tng100_alpha_1e6.iloc[:, 0]
angles_q  = angles_all[index][index_q]
angles_sf = angles_all[index][index_sf]
angles_q_sorted  = np.sort(angles_q)
angles_sf_sorted = np.sort(angles_sf)
cdf_q  = np.arange(1, len(angles_q_sorted)  + 1) / len(angles_q_sorted)
cdf_sf = np.arange(1, len(angles_sf_sorted) + 1) / len(angles_sf_sorted)
plt.plot(angles_q_sorted,  cdf_q,  color='black', ls='-',  label='TNG 1e6 Q')
plt.plot(angles_sf_sorted, cdf_sf, color='black', ls='--', label='TNG 1e6 SF')

# two-sample KS test
print(f'TNG100-1, Msat > 1e6: {stats.ks_2samp(angles_q_sorted, angles_sf_sorted)}')
# one-sample KS test
print(f'Q against Uniform Dist.: {stats.kstest(angles_q_sorted, "uniform",args=(0, 90))}')
print(f'SF against Uniform Dist.: {stats.kstest(angles_sf_sorted, "uniform",args=(0, 90))}')
print(' ')

plt.xlabel(r'$\theta$ [deg.]')
plt.ylabel('CDF')
plt.legend(fancybox=False,edgecolor='k',ncol=3)
plt.show()


# %% [markdown]
# ## MCMC fit: 

# %%
def boostrap_90(data,sf_index,q_index,N):
    
    bins = 18 # same as boostrap function
    dphi = 90/bins/2
    n_bootstrap = N  # number of bootstrap resamples

    # arrays to hold bootstrapped quenched fractions
    boot_fq = np.zeros((n_bootstrap, bins))

    # loop over bootstrap resamples
    for i in range(n_bootstrap):

        sf_sample = data.loc[sf_index].sample(frac=1, replace=True)
        q_sample  = data.loc[q_index].sample(frac=1, replace=True)
        
        g_N_sf, _ = np.histogram(sf_sample['alpha'],density=False, range=[0,90], bins=bins)
        g_N_q, _  = np.histogram(q_sample['alpha'],density=False, range=[0,90], bins=bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            fq = g_N_q / (g_N_q + g_N_sf)
        fq[np.isnan(fq)] = 0.0  # handle empty bins
        
        boot_fq[i] = fq
    
    fq_mean = np.nanmean(boot_fq, axis=0)
    fq_std = np.nanstd(boot_fq, axis=0)

    return fq_mean, fq_std



# %%
# load TNG100 data

df_tng100 = pd.read_csv("satellite_1e8/centrals_satellites_tng100_mstar_1e8.csv")
df_tng100_host_mh = pd.read_csv("satellite_1e8/centrals_satellites_tng100_host_mh_1e8.csv")
df_tng100_alpha = pd.read_csv("satellite_1e8/centrals_satellites_tng100_alpha_1e8.csv")
tng100_sfr_info = pd.read_csv("satellite_1e8/tng100_sfr_info_1e8.csv")

# %%
# boostrap TNG100 data 

index = (df_tng100_host_mh['host_mh'] > 12) & (df_tng100_host_mh['host_mh'] < 12.5)

sfr_interp = np.log10((10**(0.75*tng100_sfr_info['mstar'][index]-7.5))/10)

index_sf = (10**sfr_interp <= tng100_sfr_info['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info['sfr'][index])

q_sf_array = np.ones(len(tng100_sfr_info['host_mh'][index]))
q_sf_array[index_sf] = 0.

fq_mean_1e8, fq_std_1e8 = boostrap_90(tng100_sfr_info[index],index_sf,index_q,10000)

# load 1e6 and 1e7 data 

df_tng100 = pd.read_csv("satellite_1e7/centrals_satellites_tng100_mstar_1e7.csv")
df_tng100_host_mh = pd.read_csv("satellite_1e7/centrals_satellites_tng100_host_mh_1e7.csv")
df_tng100_alpha = pd.read_csv("satellite_1e7/centrals_satellites_tng100_alpha_1e7.csv")
tng100_sfr_info = pd.read_csv("satellite_1e7/tng100_sfr_info_1e7.csv")

index = (df_tng100_host_mh['host_mh'] > 12) & (df_tng100_host_mh['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info['host_mh'][index]))
q_sf_array[index_sf] = 0.
fq_mean_1e7, fq_std_1e7 = boostrap_90(tng100_sfr_info[index],index_sf,index_q,10000)

df_tng100 = pd.read_csv("satellite_1e6/centrals_satellites_tng100_mstar_1e6.csv")
df_tng100_host_mh = pd.read_csv("satellite_1e6/centrals_satellites_tng100_host_mh_1e6.csv")
df_tng100_alpha = pd.read_csv("satellite_1e6/centrals_satellites_tng100_alpha_1e6.csv")
tng100_sfr_info = pd.read_csv("satellite_1e6/tng100_sfr_info_1e6.csv")

index = (df_tng100_host_mh['host_mh'] > 12) & (df_tng100_host_mh['host_mh'] < 12.5)
sfr_interp = np.log10((10**(0.75*tng100_sfr_info['mstar'][index]-7.5))/10)
index_sf = (10**sfr_interp <= tng100_sfr_info['sfr'][index])
index_q = (10**sfr_interp > tng100_sfr_info['sfr'][index])
q_sf_array = np.ones(len(tng100_sfr_info['host_mh'][index]))
q_sf_array[index_sf] = 0.
fq_mean_1e6, fq_std_1e6 = boostrap_90(tng100_sfr_info[index],index_sf,index_q,10000)


# %%
def calculate_log_likelihood(theta, bin_centers, f_q, sigma_i):

    a, b, f = theta

    # Compute total variance
    s_i = sigma_i**2 + (np.exp(f))**2

    # Model prediction
    f_model = a + b * np.cos(2 * np.radians(bin_centers))

    # Residuals and log-likelihood
    residuals = (f_q - f_model)**2 / s_i
    log_likelihood = -0.5 * np.sum(residuals + np.log(2 * np.pi * s_i))
    
    return log_likelihood

def log_prior(theta):
    a, b, f = theta
    if 0 < a < 1 and -1 < b < 1 and -10 < f < 2:
        return 0.0
    return -np.inf

def calculate_log_probability(theta, bin_centers, f_q, sigma_i):
    
    log_prior_val = log_prior(theta)
    if not np.isfinite(log_prior_val):
        return -np.inf
    log_likelihood = calculate_log_likelihood(theta, bin_centers, f_q, sigma_i)
    return log_prior_val + log_likelihood


# %%
# 1e6

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e6, fq_std_1e6))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_1e6, b_1e6, f_1e6 = mean_params_tng100_all
a_1e6_std, b_1e6_std, f_1e6_std = std_params_tng100_all

# 1e7

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e7, fq_std_1e7))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_1e7, b_1e7, f_1e7 = mean_params_tng100_all
a_1e7_std, b_1e7_std, f_1e7_std = std_params_tng100_all

# 1e8

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_1e8, fq_std_1e8))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_1e8, b_1e8, f_1e8 = mean_params_tng100_all
a_1e8_std, b_1e8_std, f_1e8_std = std_params_tng100_all


# %%
# repeat for ELVES and SAGA surveys 

def bootstrap_90_observational_binned(x, y, N, bins=18, angle_range=(0, 90)):

    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    bin_edges = np.linspace(angle_range[0], angle_range[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    boot_fq = np.full((N, bins), np.nan)

    # loop over bins
    for j in range(bins):
        in_bin = (x >= bin_edges[j]) & (x < bin_edges[j + 1])
        y_bin = y[in_bin]
        n_bin = len(y_bin)

        if n_bin < 2:
            continue  # leave NaN

        # bootstrap galaxies *within this bin*
        for i in range(N):
            idx = np.random.choice(n_bin, size=n_bin, replace=True)
            boot_fq[i, j] = np.mean(y_bin[idx])

    fq_mean = np.nanmean(boot_fq, axis=0)
    fq_std  = np.nanstd(boot_fq, axis=0)

    return bin_centers, fq_mean, fq_std

def bootstrap_90_observational_binned(x, y, N=10000, bins=18, angle_range=(0, 90)):
    """
    Parameters
    ----------
    x : array-like
        Galaxy angles (0-90 deg)
    y : array-like
        Quantity to average per galaxy (e.g., 1 for quenched, 0 for SF)
    N : int
        Number of bootstrap resamples
    bins : int
        Number of bins in angle
    angle_range : tuple
        Min and max of angle (default: 0-90 deg)
    
    Returns
    -------
    bin_centers : array
        Centers of bins
    fq_mean : array
        Mean fraction per bin over all bootstraps
    fq_std : array
        1-sigma uncertainty per bin from bootstrap
    """

    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    bin_edges = np.linspace(angle_range[0], angle_range[1], bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    boot_fq = np.full((N, bins), np.nan)

    n_gal = len(x)

    for i in range(N):
        # bootstrap galaxies *across the full sample*
        idx = np.random.choice(n_gal, size=n_gal, replace=True)
        x_boot = x[idx]
        y_boot = y[idx]

        # histogram per bin
        for j in range(bins):
            in_bin = (x_boot >= bin_edges[j]) & (x_boot < bin_edges[j + 1])
            y_bin = y_boot[in_bin]
            if len(y_bin) > 0:
                boot_fq[i, j] = np.mean(y_bin)
            else:
                boot_fq[i, j] = np.nan  # empty bin

    fq_mean = np.nanmean(boot_fq, axis=0)
    fq_std = np.nanstd(boot_fq, axis=0)

    return bin_centers, fq_mean, fq_std

    
x = saga_90.copy()
y = saga_quenched.copy()

bin_centers, fq_mean_saga, fq_std_saga = bootstrap_90_observational_binned(x, y, N=10000)

elves_90_correct = []
for i in range(len(elves_90)):
    elves_90_correct.append(elves_90[i][0])
x = np.array(elves_90_correct)
y = np.array(elves_quenched)

bin_centers, fq_mean_elves, fq_std_elves = bootstrap_90_observational_binned(x, y, N=10000)

# %%
# re-fit SAGA and ELVES data with updated boostrapping 

# SAGA

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_saga, fq_std_saga))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_saga, b_saga, f_saga = mean_params_tng100_all
a_saga_std, b_saga_std, f_saga_std = std_params_tng100_all

# ELVES 

# MCMC Configuration

n_walkers_full = 20
n_dim_full = 3 
initial_guess_full = [0.7, 0.025, -3]
pos_full = np.array(initial_guess_full) + np.random.randn(n_walkers_full, n_dim_full) * 1e-2

# Setting up the sampler
sampler_full = emcee.EnsembleSampler(n_walkers_full, n_dim_full, calculate_log_probability,args=(angle_array, fq_mean_elves, fq_std_elves))

# Run MCMC
n_steps = 10_000
sampler_full.run_mcmc(pos_full, n_steps, progress=True)

# Post-Processing
samples_full = sampler_full.get_chain(discard=1000, flat=True)
log_prob_full = sampler_full.get_log_prob(discard=1000, flat=True)

mean_params_tng100_all = np.mean(samples_full, axis=0)
std_params_tng100_all = np.std(samples_full, axis=0)
a_elves, b_elves, f_elves = mean_params_tng100_all
a_elves_std, b_elves_std, f_elves_std = std_params_tng100_all

# %%
hosts['Inclination'] = elves_inclination_reformat




# %%
saga_hosts.columns

# %%
# Combined figure: angle distributions (top row) + quenched fraction + fit (bottom row)

# New high-contrast colors
saga_line = '#1f77b4'
saga_fill = '#aec7e8'
elves_line = '#d62728'
elves_fill = '#ff9896'

fig, axs = plt.subplots(2, 2, figsize=(13, 9))
(ax_dist_tng, ax_dist_obs), (ax_fq_tng, ax_fq_obs) = axs

# ===============================
# TOP ROW — ANGLE DISTRIBUTIONS
# ===============================

# --- TNG100 distributions (from Michael counts) ---
ax_dist_tng.step(edges, np.r_[tng_hist_1e6,  tng_hist_1e6[-1]], where='post',
                 label=r"$\mathrm{M_{\star,sat.}\, >\, 10^6\, M_\odot}$", lw=2, color='black', ls=':')
ax_dist_tng.step(edges, np.r_[tng_hist_1e7,  tng_hist_1e7[-1]], where='post',
                 label=r"$\mathrm{M_{\star,sat.}\, >\, 10^7\, M_\odot}$", lw=2, color='black', ls='--')
ax_dist_tng.step(edges, np.r_[tng_hist,      tng_hist[-1]],      where='post',
                 label=r"$\mathrm{M_{\star,sat.}\, >\, 10^8\, M_\odot}$", lw=2, color='black', ls='-')

ax_dist_tng.set_ylabel(r'P ($	heta$)', fontsize=14, labelpad=20)
ax_dist_tng.set_xlabel(r'$	heta$ [deg.]', fontsize=14, labelpad=20)
ax_dist_tng.set_xlim(0, 90)
ax_dist_tng.set_ylim(0, 0.030)
ax_dist_tng.legend(fontsize=12, loc='upper right', fancybox=False, edgecolor='black',
                   ncol=1, title='TNG100:', title_fontsize=12)
ax_dist_tng.tick_params(axis='both', which='major', direction='in', labelsize=12, length=7, width=1)
ax_dist_tng.tick_params(axis='both', which='minor', direction='in', labelsize=12, length=2, width=1)

# --- SAGA / ELVES observed distributions ---
# SAGA already filtered earlier via `index1`: (Rhost < R200c) & (Gold|Silver) & (inclination >= 35)
# ELVES filtered earlier in this notebook using host inclination >= 35.

saga_90 = map_to_0_90(saga_PA)
elves_90 = map_to_0_90(elves_PA)

bins = np.linspace(0, 90, 19)
counts_saga, _ = np.histogram(saga_90, bins=bins, density=True)
counts_elves, _ = np.histogram(elves_90, bins=bins, density=True)

ax_dist_obs.step(bins, np.append(counts_saga, counts_saga[-1]), where='post',
                 color=saga_line, linewidth=2, label='SAGA')
ax_dist_obs.step(bins, np.append(counts_elves, counts_elves[-1]), where='post',
                 color=elves_line, linewidth=2, label='ELVES')

ax_dist_obs.set_xlabel(r'$	heta$ [deg.]', fontsize=14, labelpad=20)
ax_dist_obs.set_xlim(0, 90)
ax_dist_obs.set_ylim(0, 0.030)
ax_dist_obs.set_yticks([])
ax_dist_obs.legend(fontsize=12, loc='upper right', fancybox=False, edgecolor='black', ncol=2)
ax_dist_obs.tick_params(axis='both', which='major', direction='in', labelsize=12, length=7, width=1)
ax_dist_obs.tick_params(axis='both', which='minor', direction='in', labelsize=12, length=2, width=1)

# ===============================
# BOTTOM ROW — QUENCHED FRACTIONS + FITS
# ===============================

# --- TNG100 quenched fractions + fits ---
ax_fq_tng.errorbar(angle_array, fq_mean_1e8, yerr=fq_std_1e8, fmt='o', color='black', mfc='black', mec='black', mew=1, capsize=3)
ax_fq_tng.errorbar(angle_array, fq_mean_1e7, yerr=fq_std_1e7, fmt='o', color='black', mfc='black', mec='black', mew=1, capsize=3)
ax_fq_tng.errorbar(angle_array, fq_mean_1e6, yerr=fq_std_1e6, fmt='o', color='black', mfc='black', mec='black', mew=1, capsize=3)

x = np.linspace(0, np.pi/2, 1000)
ax_fq_tng.plot((x*u.rad).to('degree'), a_1e6 + b_1e6 * np.cos(2 * x), color='black', lw=2, ls=':')
ax_fq_tng.plot((x*u.rad).to('degree'), a_1e7 + b_1e7 * np.cos(2 * x), color='black', lw=2, ls='--')
ax_fq_tng.plot((x*u.rad).to('degree'), a_1e8 + b_1e8 * np.cos(2 * x), color='black', lw=2, ls='-')

# Uncertainty bands (same as before)
for (a_mu, a_sig, b_mu, b_sig) in [(a_1e6, a_1e6_std, b_1e6, b_1e6_std), (a_1e7, a_1e7_std, b_1e7, b_1e7_std), (a_1e8, a_1e8_std, b_1e8, b_1e8_std)]:
    n_mc = 10000
    a_samp = np.random.normal(a_mu, a_sig, n_mc)
    b_samp = np.random.normal(b_mu, b_sig, n_mc)
    cos2x = np.cos(2 * x)
    y_mc = a_samp[:, None] + b_samp[:, None] * cos2x[None, :]
    y_low = np.percentile(y_mc, 16, axis=0)
    y_high = np.percentile(y_mc, 84, axis=0)
    theta_deg = (x * u.rad).to('degree').value
    ax_fq_tng.fill_between(theta_deg, y_low, y_high, color='k', alpha=0.1, edgecolor=None)

ax_fq_tng.set_ylabel(r'$\mathrm{f_q}$', fontsize=14, labelpad=20)
ax_fq_tng.set_xlabel(r'$	heta$ [deg.]', fontsize=14, labelpad=20)
ax_fq_tng.set_xlim(0, 90)
ax_fq_tng.set_ylim(0, 1)
ax_fq_tng.tick_params(axis='both', which='major', direction='in', labelsize=12, length=7, width=1)
ax_fq_tng.tick_params(axis='both', which='minor', direction='in', labelsize=12, length=2, width=1)

# --- SAGA / ELVES quenched fractions + fits (from the bootstrapping + MCMC cells above) ---
ax_fq_obs.errorbar(angle_array, fq_mean_saga, yerr=fq_std_saga, fmt='o', color=saga_line, mfc=saga_line, mec=saga_line, mew=1, capsize=3, label='SAGA')
ax_fq_obs.errorbar(angle_array, fq_mean_elves, yerr=fq_std_elves, fmt='o', color=elves_line, mfc=elves_line, mec=elves_line, mew=1, capsize=3, label='ELVES')

x = np.linspace(0, np.pi/2, 1000)
ax_fq_obs.plot((x*u.rad).to('degree'), a_saga + b_saga * np.cos(2 * x), color=saga_line, lw=2)
ax_fq_obs.plot((x*u.rad).to('degree'), a_elves + b_elves * np.cos(2 * x), color=elves_line, lw=2)

for (a_mu, a_sig, b_mu, b_sig, col) in [(a_saga, a_saga_std, b_saga, b_saga_std, saga_line), (a_elves, a_elves_std, b_elves, b_elves_std, elves_line)]:
    n_mc = 10000
    a_samp = np.random.normal(a_mu, a_sig, n_mc)
    b_samp = np.random.normal(b_mu, b_sig, n_mc)
    cos2x = np.cos(2 * x)
    y_mc = a_samp[:, None] + b_samp[:, None] * cos2x[None, :]
    y_low = np.percentile(y_mc, 16, axis=0)
    y_high = np.percentile(y_mc, 84, axis=0)
    theta_deg = (x * u.rad).to('degree').value
    ax_fq_obs.fill_between(theta_deg, y_low, y_high, color=col, alpha=0.1, edgecolor=None)

ax_fq_obs.set_xlabel(r'$	heta$ [deg.]', fontsize=14, labelpad=20)
ax_fq_obs.set_xlim(0, 90)
ax_fq_obs.set_ylim(0, 1)
ax_fq_obs.set_yticks([])
ax_fq_obs.legend(fontsize=12, loc='upper right', fancybox=False, edgecolor='black', ncol=2)
ax_fq_obs.tick_params(axis='both', which='major', direction='in', labelsize=12, length=7, width=1)
ax_fq_obs.tick_params(axis='both', which='minor', direction='in', labelsize=12, length=2, width=1)

plt.subplots_adjust(wspace=0.05, hspace=0.22)
plt.show()


# %% [markdown]
# ## BIC/AIC Analysis:

# %%
b_samp = np.random.normal(b_1e6, b_1e6_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

# %%
np.abs(b_1e6/b_1e6_std)

# %% [markdown]
# ### Interpretation: If the amplitude b = 0, then the sinsoidal function is merely a constant, or the quench fraction. We can therefore ask, how many std away from 0 is the parameter b mean? We find that there is significant overlap in the SAGA and ELVES b parameter. However, there is less overlap in the TNG100-1 subsamples.

# %%
# calulate how many std the mean is from 0 (for the parameter b, such that b = 0 is consistent with a st)

n_mc = 100000
b_samp = np.random.normal(b_saga, b_saga_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_elves, b_elves_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_1e8, b_1e8_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_1e7, b_1e7_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)

#######

b_samp = np.random.normal(b_1e6, b_1e6_std, n_mc)

#plt.hist(b_samp,bins=100,histtype='step',color='k');

b_mean = np.mean(b_samp)
b_std  = np.std(b_samp)
sigma_significance = np.abs(b_mean / b_std)
print(sigma_significance)


# %%
# calculate the BIC (and AIC) for the sinusoidal fit versus constant
# be careful of f value -- is this an extra parmameter?

def sinusoid(x, a, b, f):
    
    return a + b * np.cos(2 * x)

def BIC(x_data, y_data, y_err, a_fit, b_fit, f_fit):

    resid = y_data - sinusoid(x_data, a_fit, b_fit, f_fit)
    print(resid)                       
    n = len(y_data)
    k = 2
    logL = -0.5 * np.sum((resid / y_err)**2 + np.log(2*np.pi*y_err**2))
    bic_sin = k * np.log(n) - 2 * logL
    print("BIC Sinudoid Fit =", bic_sin) 
    aic_sin = 2*k - 2*logL
    print("AIC Sinudoid Fit =", aic_sin) 

    resid_const = y_data - np.mean(y_data)
    k_const = 1
    logL_const = -0.5 * np.sum((resid_const / y_err)**2 + np.log(2*np.pi*y_err**2))
    bic_const = k_const * np.log(n) - 2 * logL_const
    print("BIC Constant Fit =", bic_const)
    aic_const = 2*k - 2*logL_const
    print("AIC Constant Fit =", aic_const) 

    print('Delta BIC =',np.abs(bic_const-bic_sin))
    print('Delta AIC =',np.abs(aic_const-aic_sin))
    if bic_const < bic_sin:
        print("From BIC, Constant model preferred")
    else:
        print("From BIC, Sinusoid model preferred")

    if aic_const < aic_sin:
        print("From AIC, Constant model preferred")
    else:
        print("From AIC, Sinusoid model preferred")

print('ELVES:')
BIC(angle_array, fq_mean_elves, fq_std_elves, a_elves, b_elves, f_elves)
print(' ')
print('SAGA:')
BIC(angle_array, fq_mean_saga, fq_std_saga, a_saga, b_saga, f_saga)
print(' ')
print('TNG100-1, > 1e8:')
BIC(angle_array, fq_mean_1e8, fq_std_1e8, a_1e8, b_1e8, f_1e8)
print(' ')
print('TNG100-1, > 1e7:')
BIC(angle_array, fq_mean_1e7, fq_std_1e7, a_1e7, b_1e7, f_1e7)
print(' ')
print('TNG100-1, > 1e6:')
BIC(angle_array, fq_mean_1e6, fq_std_1e6, a_1e6, b_1e6, f_1e6)

# %% [markdown]
# ### Interpretation: Instead of looking at the amplitude b, we can instead ask: How much better of a fit is the sinusoidal function compared to a constant (the mean quench fraction across all angle bins). This can be calculated using both the BIC (penalizes number of paramaeters more) and AIC. We find that all data are better fit with a constant. Supposedly the AIC for the SAGA data is slightly lower for the sinusoid compared to the constant model, but the $\Delta$BIC is 1.15. As a good rule of thumb, > 10 suggests that there is strong evidence.

# %% [markdown]
# ## Old Code from Nick:

# %% [markdown]
# ## Alignment Test

# %%

#SAGA
theta_deg = np.array(saga_90)  # your folded angles
theta_rad = np.deg2rad(theta_deg)

# Effect size
A_obs = np.mean(np.cos(2 * theta_rad))

# Bootstrap CI
B = 5000
rng = np.random.default_rng(0)
A_boot = []
n = len(theta_rad)

for _ in range(B):
    sample = rng.choice(theta_rad, size=n, replace=True)
    A_boot.append(np.mean(np.cos(2 * sample)))

lo, hi = np.percentile(A_boot, [2.5, 97.5])
print("SAGA")
print(f"<cos(2θ)> = {A_obs:.4f}")
print(f"95% CI: [{lo:.4f}, {hi:.4f}]")


# %%

#ELVES
theta_deg = np.array(elves_90)  # your folded angles
theta_rad = np.deg2rad(theta_deg)

# Effect size
A_obs = np.mean(np.cos(2 * theta_rad))

# Bootstrap CI
B = 5000
rng = np.random.default_rng(0)
A_boot = []
n = len(theta_rad)

for _ in range(B):
    sample = rng.choice(theta_rad, size=n, replace=True)
    A_boot.append(np.mean(np.cos(2 * sample)))

lo, hi = np.percentile(A_boot, [2.5, 97.5])
print("ELVES")
print(f"<cos(2θ)> = {A_obs:.4f}")
print(f"95% CI: [{lo:.4f}, {hi:.4f}]")

# %%
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import kstest

print("SAGA: Testing Whether Quench and Star forming follow different distribution")
# Inputs:
# theta_deg: folded angles in [0, 90], shape (N,)
# quenched:  1 for quenched, 0 for star-forming, shape (N,)
# host_id:   host identifier per satellite, shape (N,)
theta_deg = np.asarray(saga_90)
quenched  = np.asarray(saga_quenched).astype(int)
host_id   = np.asarray(saga_sats['HOSTID'][radius_limit])


# Split groups
th_q  = theta_deg[quenched == 1]
th_sf = theta_deg[quenched == 0]


print("saga:", kstest(theta_deg, 'uniform' , args=(0, 90)))


# --- (1) Two-sample KS ---
D, p_ks = ks_2samp(th_q, th_sf, alternative='two-sided')
# Extract scalar values to avoid formatting errors
D = D.item() if hasattr(D, 'item') else (float(D[0]) if hasattr(D, '__len__') else D)
p_ks = p_ks.item() if hasattr(p_ks, 'item') else (float(p_ks[0]) if hasattr(p_ks, '__len__') else p_ks)



# --- (2) ΔA effect size ---
c2 = np.cos(2 * np.deg2rad(theta_deg))
A_q  = c2[quenched == 1].mean()
A_sf = c2[quenched == 0].mean()
dA   = A_q - A_sf

print(f"KS: D={D:.4f}, p={p_ks:.3g}")
print(f"A_q={A_q:.4f}, A_sf={A_sf:.4f}, ΔA={dA:.4f}")

# --- (3) 95% CI for ΔA via cluster (host) bootstrap ---
rng = np.random.default_rng(0)
hosts = np.unique(host_id)
B = 5000
dA_boot = []

for _ in range(B):
    boot_hosts = rng.choice(hosts, size=len(hosts), replace=True)
    # keep all satellites belonging to sampled hosts
    mask = np.isin(host_id, boot_hosts)
    c2_b = c2[mask]
    q_b  = quenched[mask]
    if (q_b == 1).any() and (q_b == 0).any():
        dA_boot.append(c2_b[q_b==1].mean() - c2_b[q_b==0].mean())

lo, hi = np.percentile(dA_boot, [2.5, 97.5])
print(f"ΔA 95% CI: [{lo:.4f}, {hi:.4f}]")

# --- (4) Cluster-aware permutation p-values ---
# Permute labels WITHIN each host; preserves per-host clustering and class sizes.
def permute_labels_within_host(q, h, rng):
    q_perm = q.copy()
    for hh in np.unique(h):
        idx = np.where(h == hh)[0]
        q_perm[idx] = rng.permutation(q_perm[idx])
    return q_perm

Bperm = 5000
dA_perm = []
D_perm  = []

for _ in range(Bperm):
    q_p = permute_labels_within_host(quenched, host_id, rng)
    # ΔA under permuted labels
    A_q_p  = c2[q_p==1].mean() if (q_p==1).any() else np.nan
    A_sf_p = c2[q_p==0].mean() if (q_p==0).any() else np.nan
    dA_perm.append(A_q_p - A_sf_p)

    # KS under permuted labels
    th_q_p  = theta_deg[q_p == 1]
    th_sf_p = theta_deg[q_p == 0]
    if len(th_q_p) > 0 and len(th_sf_p) > 0:
        Dp, _ = ks_2samp(th_q_p, th_sf_p, alternative='two-sided')
        D_perm.append(Dp.item() if hasattr(Dp, 'item') else (float(Dp[0]) if hasattr(Dp, '__len__') else Dp))

dA_perm = np.array(dA_perm, dtype=float)
D_perm  = np.array(D_perm, dtype=float)

p_perm_dA = (np.sum(np.abs(dA_perm) >= abs(dA)) + 1) / (np.sum(~np.isnan(dA_perm)) + 1)
p_perm_KS = (np.sum(D_perm >= D) + 1) / (len(D_perm) + 1)

print(f"Permutation p (ΔA): {p_perm_dA:.4g}")
print(f"Permutation p (KS): {p_perm_KS:.4g}")


# %%
import numpy as np
from scipy.stats import ks_2samp
from scipy.stats import kstest

print("ELVES: Testing Whether Quench and Star forming follow different distribution")
# Inputs:
# theta_deg: folded angles in [0, 90], shape (N,)
# quenched:  1 for quenched, 0 for star-forming, shape (N,)
# host_id:   host identifier per satellite, shape (N,)
theta_deg = np.asarray(elves_90)
quenched  = np.asarray(elves_quenched).astype(int)
# Use the kept_indices from the ELVES processing to get the correct host IDs
host_id   = np.asarray(sats['Host'].iloc[kept_indices])


# Split groups
th_q  = theta_deg[quenched == 1]
th_sf = theta_deg[quenched == 0]


print("ELVES:", kstest(theta_deg, 'uniform' , args=(0, 90)))


# --- (1) Two-sample KS ---
D, p_ks = ks_2samp(th_q, th_sf, alternative='two-sided')
# Extract scalar values to avoid formatting errors
D = D.item() if hasattr(D, 'item') else (float(D[0]) if hasattr(D, '__len__') else D)
p_ks = p_ks.item() if hasattr(p_ks, 'item') else (float(p_ks[0]) if hasattr(p_ks, '__len__') else p_ks)



# --- (2) ΔA effect size ---
c2 = np.cos(2 * np.deg2rad(theta_deg))
A_q  = c2[quenched == 1].mean()
A_sf = c2[quenched == 0].mean()
dA   = A_q - A_sf

print(f"KS: D={D:.4f}, p={p_ks:.3g}")
print(f"A_q={A_q:.4f}, A_sf={A_sf:.4f}, ΔA={dA:.4f}")

# --- (3) 95% CI for ΔA via cluster (host) bootstrap ---
rng = np.random.default_rng(0)
hosts = np.unique(host_id)
B = 5000
dA_boot = []

for _ in range(B):
    boot_hosts = rng.choice(hosts, size=len(hosts), replace=True)
    # keep all satellites belonging to sampled hosts
    mask = np.isin(host_id, boot_hosts)
    c2_b = c2[mask]
    q_b  = quenched[mask]
    if (q_b == 1).any() and (q_b == 0).any():
        dA_boot.append(c2_b[q_b==1].mean() - c2_b[q_b==0].mean())

lo, hi = np.percentile(dA_boot, [2.5, 97.5])
print(f"ΔA 95% CI: [{lo:.4f}, {hi:.4f}]")

# --- (4) Cluster-aware permutation p-values ---
# Permute labels WITHIN each host; preserves per-host clustering and class sizes.
def permute_labels_within_host(q, h, rng):
    q_perm = q.copy()
    for hh in np.unique(h):
        idx = np.where(h == hh)[0]
        q_perm[idx] = rng.permutation(q_perm[idx])
    return q_perm

Bperm = 5000
dA_perm = []
D_perm  = []

for _ in range(Bperm):
    q_p = permute_labels_within_host(quenched, host_id, rng)
    # ΔA under permuted labels
    A_q_p  = c2[q_p==1].mean() if (q_p==1).any() else np.nan
    A_sf_p = c2[q_p==0].mean() if (q_p==0).any() else np.nan
    dA_perm.append(A_q_p - A_sf_p)

    # KS under permuted labels
    th_q_p  = theta_deg[q_p == 1]
    th_sf_p = theta_deg[q_p == 0]
    if len(th_q_p) > 0 and len(th_sf_p) > 0:
        Dp, _ = ks_2samp(th_q_p, th_sf_p, alternative='two-sided')
        D_perm.append(Dp.item() if hasattr(Dp, 'item') else (float(Dp[0]) if hasattr(Dp, '__len__') else Dp))

dA_perm = np.array(dA_perm, dtype=float)
D_perm  = np.array(D_perm, dtype=float)

p_perm_dA = (np.sum(np.abs(dA_perm) >= abs(dA)) + 1) / (np.sum(~np.isnan(dA_perm)) + 1)
p_perm_KS = (np.sum(D_perm >= D) + 1) / (len(D_perm) + 1)

print(f"Permutation p (ΔA): {p_perm_dA:.4g}")
print(f"Permutation p (KS): {p_perm_KS:.4g}")


# %%

# %%

# %%
