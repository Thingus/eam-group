# This code compare GEOS-Chem model and DEFRA sites active ammonia
# Please contact Alok Pandey ap744@leicester.ac.uk for any further clarifications or details

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import StandardScaler
import datetime
import xarray as xr
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.cm as cm
import glob
from scipy import stats
#from bootstrap import rma
from scipy.stats import gaussian_kde

#Use todays date to plot files - good practice to save files name with date
Today_date=datetime.datetime.now().strftime("%Y%m%d")

###Different cmap options
# cmap = matplotlib.cm.get_cmap('brewer_RdBu_11')
# cmap = cm.jet
cmap = cm.rainbow
#cmap = cm.YlOrRd

#read UKEAP ammonia datasets here scratch_alok -> /scratch/uptrop/ap744
path='/scratch/uptrop/ap744/UKEAP_data/UKEAP_NH3_particulate_ammonium/gaseousAmmonia_Active/'
ammonia_files=glob.glob(path + '27-UKA0*-2016_active_*.csv')
print (ammonia_files)

# read csv file having DEFRA sites details
sites = pd.read_csv('/home/a/ap744/scratch_alok/UKEAP_data/DEFRA_UKEAP_sites_details/UKEAP_NH3_sites_details.csv', encoding= 'unicode_escape')
#print (sites.head(10))
ID = sites["UK-AIR_ID"]
#print (ID)



# site wise annual mean computation  
x = []
for f in ammonia_files:
	df = pd.read_csv(f)  
	#print (df.head(5))
	print (len(ammonia_files))
	sitesA = sites.copy()
	#df['Measurement'].values[df['Measurement'] <=0.1] = np.nan

	mean_A= df["Measurement"].mean()
	print (mean_A, f[89:97])
	sitesA["ammonia_annual_mean"] = mean_A
	#print (sitesA.head(10))
	
	x.append(
	{
		'UK-AIR_ID':f[89:97],
		'ammonia_annual_mean':mean_A
		}
		)
	#print (x)
	
id_mean = pd.DataFrame(x)
#print (id_mean.head(3))

df_merge_col = pd.merge(sites, id_mean, on='UK-AIR_ID', how ='right')
print (df_merge_col.head(5))

#####export csv file having site wise annual mean information if needed 
#df_merge_col.to_csv(r'/home/a/ap744/scratch_alok/python_work/ammonia_annual_mean.csv')

#drop extra information from pandas dataframe
df_merge_colA = df_merge_col.drop(['S No'], axis=1)
print (df_merge_colA.head(5))

# change datatype to float to remove any further problems
df_merge_colA['Long'] = df_merge_colA['Long'].astype(float)
df_merge_colA['Lat'] = df_merge_colA['Lat'].astype(float)

#get sites information
sites_lon = df_merge_colA['Long']
sites_lat = df_merge_colA['Lat']
sites_ammonia_AM = df_merge_colA['ammonia_annual_mean']
sites_name = df_merge_colA['Site_Name']
print (sites_ammonia_AM, sites_name, sites_lat, sites_lon)


#####Reading GEOS-Chem files 
#os.chdir("/data/uptrop/Projects/DEFRA-NH3/GC/geosfp_eu_naei_iccw/AerosolMass/2016")
os.chdir("/scratch/uptrop/ap744/GEOS-Chem_outputs/")

Species  = sorted(glob.glob("GEOSChem.SpeciesConc*.nc4"))
Aerosols = sorted(glob.glob("GEOSChem.AerosolMass*nc4"))
StateMet = sorted(glob.glob("GEOSChem.StateMet*B.nc4"))

Species = Species[:] 
Aerosols = Aerosols[:]
StateMet = StateMet[:]
print(Aerosols, Species, StateMet, sep = "\n")

Species  = [xr.open_dataset(file) for file in Species]
Aerosols = [xr.open_dataset(file) for file in Aerosols]
StateMet = [xr.open_dataset(file) for file in StateMet]

#ds = xr.open_mfdataset(StateMet)
#monthly_data = ds.resample(time='m').mean()
#print(monthly_data)
#monthly_data_StateMet = StateMet.resample(freq = 'm', dim = 'time', how = 'mean')
#print(monthly_data_StateMet)

#ammonia sufrace layer
GC_surface_ammonia = [data['SpeciesConc_NH3'].isel(time=0,lev=0) for data in Species]
print (GC_surface_ammonia)

# convert unit for NH3 (dry mol/mol to ug/m3)
surface_AIRDEN = [data['Met_AIRDEN'].isel(time=0,lev=0) for data in StateMet]

surface_NO2_mass  = [x*y*46/(6.022*1e11) for (x,y) in zip(surface_NO2,surface_AIRNUMDEN)]

#Geos-Chem Annual Mean
GC_surface_ammonia = sum(GC_surface_ammonia)/len(GC_surface_ammonia)
print (GC_surface_ammonia.shape)

"""#GEOS-Chem lat long information --Not working properly
#gc_lon = Aerosols[0]['lon']
#gc_lat = Aerosols[0]['lat']
#gc_lon,gc_lat = np.meshgrid(gc_lon,gc_lat)

# get GEOS-Chem lon and lat
gc_lon = GC_surface_ammonia['lon']
gc_lat = GC_surface_ammonia['lat']
print (len(gc_lon))
print (len(gc_lat))
print ((gc_lon))
print ((gc_lat))

# get number of sites from size of long and lat:
nsites=len(sites_lon)

# Define GEOS-Chem data obtained at same location as monitoring sites:
gc_data_ammonia=np.zeros(nsites)

# Count the number of valid measurements at each site:
counter_site = np.zeros(nsites)

for w in range(len(sites_lat)):
	#print ((sites_lat[w],gc_lat))
	# lat and lon indices:
	lon_index = np.argmin(np.abs(np.subtract(sites_lon[w],gc_lon)))
	lat_index = np.argmin(np.abs(np.subtract(sites_lat[w],gc_lat)))

	#print (lon_index)
	#print (lat_index)
	gc_data_ammonia[w] = GC_surface_ammonia[lon_index, lat_index]

print (gc_data_ammonia.shape)
print (sites_ammonia_AM.shape)

# quick scatter plot
#plt.plot(sites_ammonia_AM,gc_data_ammonia,'o')
#plt.show()

# Compare DERFA and GEOS-Chem:

#Normalized mean bias
nmb=100.*((sites_ammonia_AM)- np.mean(gc_data_ammonia))/(gc_data_ammonia)
print(' DEFRA NMB= ', nmb)

#correlation
correlate=stats.pearsonr(gc_data_ammonia,sites_ammonia_AM)
print('Correlation = ',correlate)

#Regression ~ bootstrap is not working
#regres=rma(gc_data_ammonia,sites_ammonia_AM,1000)
#print('slope: ',regres[0])   
#print('Intercept: ',regres[1])
#print('slope error: ',regres[2])   
#print('Intercept error: ',regres[3])

# plotting spatial map model and network 
os.chdir('/home/a/ap744/scratch_alok/shapefiles/GBP_shapefile')
Europe_shape = r'GBR_adm1.shp'
Europe_map = ShapelyFeature(Reader(Europe_shape).geometries(),
                               ccrs.PlateCarree(), edgecolor='black',facecolor='none')
print ('Shapefile_read')
title_list = 'DEFRA and GEOS-Chem Particulate ammonia'
title_list1 = 'Spatial Map DEFRA and GEOS-Chem Particulate ammonia'

#fig,ax = plt.subplots(2,1, figsize=(11,11))
fig1 = plt.figure(facecolor='White',figsize=[11,11]);pad= 1.1;
# plt.suptitle(title_list, fontsize = 35, y=0.96)

ax = plt.subplot(1,1,1);
#plt.title(title_list1, fontsize = 30, y=1)
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(Europe_map)
ax.set_extent([-9, 3, 49, 61], crs=ccrs.PlateCarree()) # [lonW,lonE,latS,latN]

GC_surface_ammonia.plot(ax=ax,cmap=cmap,vmin = 0,vmax =3,
								cbar_kwargs={'shrink': 0.5, 
											'pad' : 0.01,
											'label': 'GEOS-Chem ammonia ($\mu$g m$^{-3}$)',
											'orientation':'horizontal'})

ax.scatter(x=sites_lon, y=sites_lat,c=sites_ammonia_AM,
		facecolors='none',edgecolors='black',linewidths=5,s = 100)
ax.scatter(x=sites_lon, y=sites_lat,c=sites_ammonia_AM,
		cmap=cmap,s = 100,vmin = 0,vmax = 3)

PCM=ax.get_children()[2] #get the mappable, the 1st and the 2nd are the x and y axes

colorbar = plt.colorbar(PCM, ax=ax,label='DEFRA_ammonia ($\mu$g m$^{-3}$)',
                        orientation='vertical',shrink=0.5,pad=0.01)

#ax.annotate('ammonia', xy=(10.1, 42.5), xytext=(10, 42.5),fontsize = 30)

ax.set_title('DEFRA and GEOS-Chem Particulate ammonia')



colorbar.ax.tick_params(labelsize=10) 
colorbar.ax.xaxis.label.set_size(10)
plt.savefig('/home/a/ap744/scratch_alok/python_work/'+Today_date+'ammonia_GEOS-Chem_DEFRAspatial.png',bbox_inches='tight')


#scatter plot 

#ax = plt.subplot(1,2,2);
#ax.scatter(x=sites_ammonia_AM, y=gc_data_ammonia, c=gc_data_ammonia, 
#	cmap=cmap, s=100,vmin=0, vmax=2)
#ax.colorbar(extend = 'both')


# Stat Calculation
x1 = gc_data_ammonia
y1 = sites_ammonia_AM
x1y1 = np.vstack([x1,y1])
z1 = gaussian_kde(x1y1)(x1y1)

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1,y1)  
line1 = slope1*x1+intercept1


title_list2 = 'DERFA & GEOS-Chem ammonia'
fig2 = plt.figure(facecolor='White',figsize=[16,16]);pad= 1.1;
ax = plt.subplot(1,1,1);
plt.title(title_list2, fontsize = 30, y=1)
ax.scatter(x1, y1, c=z1, s=101, edgecolor='',cmap=cm.jet)
plt.plot(x1, line1, 'r-', linewidth=3)
lineStart = 0 
lineEnd = 2
plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'c',linewidth=2)
plt.xlim(lineStart, lineEnd)
plt.ylim(lineStart, lineEnd)
# plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
plt.draw()

plt.xlabel("GEOS-Chem ammonia ($\mu$g m$^{-3}$)", fontsize = 30)
plt.ylabel("DEFRA_ammonia ($\mu$g m$^{-3}$)", fontsize = 30)
ax.yaxis.get_offset_text().set_size(25)
plt.yticks(fontsize = 25)
ax.xaxis.get_offset_text().set_size(25)
plt.xticks(fontsize = 25)
# plt.legend(( 'RegressionLine R$^2$={0:.2f}'.format(r_valueA),'1:1 Line','Data'))
plt.legend(('RegressionLine' ,'1:1 Line','Data'),fontsize = 25)
# print slope1,round(slope1,2),str(round(intercept1,1))
ax.annotate('R$^2$ = {0:.2f}'.format(r_value1*r_value1),xy=(0.8,0.7), xytext=(0, pad),
		xycoords='axes fraction', textcoords='offset points',
		ha='center', va='bottom',rotation='horizontal',fontsize=30)
ax.annotate('y = {0:.2f}'.format(slope1)+'x + {0:.2f}'.format(intercept1),xy=(0.7,0.63), xytext=(0, pad),
		xycoords='axes fraction', textcoords='offset points',
		ha='center', va='bottom',rotation='horizontal',fontsize=30)
for axis in ['top','bottom','left','right']:
  ax.spines[axis].set_linewidth(3)

#plt.show()
plt.subplots_adjust(left=0.05, bottom=0.07, right=0.98, top=0.97, wspace=0.20, hspace=0.05);
plt.savefig('/home/a/ap744/scratch_alok/python_work/'+Today_date+'ammonia_GEOS-Chem_DEFRAscatter.png',bbox_inches='tight')
plt.show()"""
