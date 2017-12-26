import pandas as pd
import csv
import numpy as np
import os
from functools import reduce
from pandas import DataFrame, Series 
import glob

#data ingestion and wrangling for crop production
path =r'csv_data/'
filenames = glob.glob(path + "/*cropyear_production.csv")

dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename))
# Concatenate all data into one DataFrame
big_frame = pd.concat(dfs, ignore_index=True)



with open('CROP_PRODUCTION_ALL.csv', 'w') as f:
    big_frame.to_csv(f, header=False,index=0)


colnames=['Year','Commodity_Code','Crop_Name','County_Code','County','Harvested','Yield','Production','Price','Unit','Value'] 
crop_production=pd.read_csv('CROP_PRODUCTION_ALL.csv', names=colnames, header=None,dtype=object)


crop_production_Fresno=crop_production[crop_production['County'].dropna().str.contains("Fresno") & crop_production['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
crop_production_Fresno['County']='Fresno'
#crop_production_Fresno.loc[:,'County']='Fresno'
crop_production_Kern=crop_production[crop_production['County'].dropna().str.contains("Kern") & crop_production['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
#crop_production_Kern.loc[:,'County']='Kern'
crop_production_Kern['County']='Kern'
with open('1-CROP_PRODUCTION-Fresno.csv', 'w') as f:
    
    crop_production_Fresno.to_csv(f, header=True,index=0)
f.close()

########################################################33


with open('1-CROP_PRODUCTION-Kern.csv', 'w') as f:
    
    crop_production_Kern.to_csv(f, header=True,index=0)
f.close()

#################################################################
df_crop_f=pd.read_csv('1-CROP_PRODUCTION-Fresno.csv',usecols=['Year','Commodity_Code','County_Code','County','Harvested','Yield','Production','Price','Value'],dtype={'Year':int,'Harvested':int,'Production':int,'Price':float})
df_crop_k=pd.read_csv('1-CROP_PRODUCTION-Kern.csv',usecols=['Year','Commodity_Code','County_Code','County','Harvested','Yield','Production','Price','Value'],dtype={'Year':int,'Harvested':int,'Production':int,'Price':float})
#print(df_crop1)

##data ingestion and wrangling for Precipitation 
df_prec_f=pd.read_csv("f_temp_prec/Fresno_Rainfall_Yearly.csv")
df_prec_f.rename(columns={
                 'January': 'January_p',
                 'February': 'February_p',
                 'March': 'March_p',
                 'April': 'April_p',
                 'May': 'May_p',
                 'June': 'June_p',
                 'July': 'July_p',
                 'August': 'August_p',
                 'September': 'September_p',
                 'October': 'October_p',
                 'November': 'November_p',
                 'December': 'December_p',
                 'Total': 'Total_p'}, inplace=True)

df_prec_f.replace(['T'],np.random.uniform(+0.001,+0.005), inplace=True)
df_prec1_f=df_prec_f[(df_prec_f['Year']>='1980') & (df_prec_f['Year']<'2016')]
df_prec1_f = df_prec1_f.astype(float)

################################

with open('2-RAINFALL_FRESNO.csv', 'w') as f:
    df_prec1_f.to_csv(f, header=True,index=0,float_format='%.3f')
f.close()


####################################################

df_prec_k=pd.read_csv("k_temp_prec/Kern_Rainfall_Yearly.csv")
df_prec_k.columns=['Year','January_p','February_p','March_p','April_p','May_p','June_p','July_p','August_p','September_p','October_p','November_p','December_p','Total_p']

df_prec_k.replace(['T'],np.random.uniform(+0.001,+0.005), inplace=True)
df_prec_k = df_prec_k.astype(float)
df_prec_k.replace(r'[\-]+', r'', regex=True, inplace=True)
df_prec1_k=df_prec_k[(df_prec_k['Year']>=1980) & (df_prec_k['Year']<2016)]
df_prec1_k = df_prec_k.astype(float)
#############################################################


with open('2-RAINFALL_KERN.csv', 'w') as f:
    #df_prec1_k = df_prec1_k.astype(float)
    df_prec1_k.to_csv(f, header=True,index=0,float_format='%.3f')
f.close()

#################################################################
##data ingestion and wrangling for Temperatures

df_temp_f=pd.read_csv("f_temp_prec/Fresno_Average_Monthly_Temperatures.csv")
df_temp_f.rename(columns={
                 'January': 'January_t',
                 'February': 'February_t',
                 'March': 'March_t',
                 'April': 'April_t',
                 'May': 'May_t',
                 'June': 'June_t',
                 'July': 'July_t',
                 'August': 'August_t',
                 'September': 'September_t',
                 'October': 'October_t',
                 'November': 'November_t',
                 'December': 'December_t',
                 'Annual': 'Annual_t'}, inplace=True)

df_temp1_f=df_temp_f[(df_temp_f['Year']>=1980) & (df_temp_f['Year']<2016)]
df_temp1_f = df_temp1_f.astype(float)


df_temp_k=pd.read_csv("k_temp_prec/Kern_Average_Temperatures.csv")
df_temp_k.rename(columns={
                 'January': 'January_t',
                 'February': 'February_t',
                 'March': 'March_t',
                 'April': 'April_t',
                 'May': 'May_t',
                 'June': 'June_t',
                 'July': 'July_t',
                 'August': 'August_t',
                 'September': 'September_t',
                 'October': 'October_t',
                 'November': 'November_t',
                 'December': 'December_t',
                 'Annual': 'Annual_t'}, inplace=True)
df_temp_k.columns=['Year','January_t','February_t','March_t','April_t','May_t','June_t','July_t','August_t','September_t','October_t','November_t','December_t','Annual_t']
df_temp_k.replace(r'[\-]+', r'', regex=True, inplace=True)
df_temp1_k=df_temp_k[(df_temp_k['Year']>=1980) & (df_temp_k['Year']<2016)]







######Fresno Census Data####################

f_census1=pd.read_csv("census_data/fresno_percapita_personal_income.csv")
f_census2=pd.read_csv("census_data/fresno_personal_income.csv")
f_census3=pd.read_csv("census_data/fresno-resident_population.csv")
f_census4=pd.read_csv("census_data/fresno_house_price_index.csv")

f_census_df=[f_census1,f_census2,f_census3,f_census4]
fresno_census_concat = reduce(lambda  left,right: pd.merge(left,right,on='DATE', how='left'), f_census_df)
fresno_census_concat['DATE'] = fresno_census_concat['DATE'].map(lambda x: str(x)[:-6])
with open('3-FRESNO_CENSUS_DATA.csv', 'w') as f:
    fresno_census_concat.to_csv(f, header=True,index=0)

col_census=['Year','Percapita_Personal_Income','Personal_Income','Resident_Population','House_Price_Index']
fresno_census=pd.read_csv('FRESNO_CENSUS_DATA.csv', names=col_census)
fresno_census=fresno_census[(fresno_census['Year']>='1980') & (fresno_census['Year']<'2016')]
fresno_census = fresno_census.astype(float)





######Kern Census Data####################

k_census1=pd.read_csv("census_data/kern_percapita_personal_income.csv")
k_census2=pd.read_csv("census_data/kern_personal_income.csv")
k_census3=pd.read_csv("census_data/kern_resident_population.csv")
k_census4=pd.read_csv("census_data/kern_house_price_index.csv")

k_census_df=[k_census1,k_census2,k_census3,k_census4]
kern_census_concat = reduce(lambda  left,right: pd.merge(left,right,on='DATE', how='left'), k_census_df)
kern_census_concat['DATE'] = kern_census_concat['DATE'].map(lambda x: str(x)[:-6])

kern_census_concat.rename(columns={
                 'DATE': 'Year',
                 'PCPI06029': 'Percapita_Personal_Income',
                 'PI06029': 'Personal_Income',
                 'CAKERN0POP': 'Resident_Population',
                 'ATNHPIUS06029A': 'House_Price_Index'}, inplace=True)


with open('3-KERN_CENSUS_DATA.csv', 'w') as f:
    kern_census_concat.to_csv(f, header=True,index=0)

#col_census1=['Year','Percapita_Personal_Income','Personal_Income','Resident_Population']

kern_census=pd.read_csv('KERN_CENSUS_DATA.csv')

#kern_census.columns=['Year','Percapita_Personal_Income','Personal_Income','Resident_Population']

kern_census1=kern_census[(kern_census['Year']>=1980) & (kern_census['Year']<2016)]

print(kern_census1)
####################################################





with open('4-TEMPERATURES_FRESNO.csv', 'w') as f:
    df_temp1_k = df_temp1_k.astype(float)

    df_temp1_f.to_csv(f, header=True,index=0)
f.close()

with open('4-TEMPERATURES_KERN.csv', 'w') as f:
    df_temp1_k.to_csv(f, header=True,index=0)
f.close()

############################################
fresno_data_frames=[df_crop_f,df_prec1_f,df_temp1_f,fresno_census]

df_concat_fresno = reduce(lambda  left,right: pd.merge(left,right,on='Year', how='left'), fresno_data_frames)
####################################################






kern_data_frames=[df_crop_k,df_prec1_k,df_temp1_k,kern_census]

df_concat_kern = reduce(lambda  left,right: pd.merge(left,right,on='Year', how='left'), kern_data_frames)

'''
with open('4-Fresno_Data.csv', 'w') as f:
    df_concat_fresno.to_csv(f, header=True,index=0)
f.close()
######################################################3

with open('4-Kern_Data.csv', 'w') as f:
    df_concat_kern.to_csv(f, header=True,index=0)
f.close()

'''



with open('5-MERGE.csv', 'w') as f:
    df_concat_fresno.to_csv(f, header=True,index=0)
    df_concat_kern.to_csv(f, header=False,index=0)
f.close()


merge_almond_production=pd.read_csv("5-MERGE.csv")
merge_almond_production.insert(9, 'Grow_total_p', (merge_almond_production['February_p']+merge_almond_production['March_p']+merge_almond_production['April_p']+merge_almond_production['May_p']+merge_almond_production['June_p']))
merge_almond_production.insert(10, 'Grow_avg_t', (merge_almond_production['February_t']+merge_almond_production['March_t']+merge_almond_production['April_t']+merge_almond_production['May_t']+merge_almond_production['June_t'])/5)

with open('5-MERGE.csv', 'w') as f:
    merge_almond_production.to_csv(f, header=True,index=0)

f.close()

print(merge_almond_production.shape)
###########################################

'''
df0=pd.read_csv("1980cropyear_production.csv"); df1=pd.read_csv("1981cropyear_production.csv")
df2=pd.read_csv("1982cropyear_production.csv");df3=pd.read_csv("1983cropyear_production.csv")
df4=pd.read_csv("1984cropyear_production.csv");df5=pd.read_csv("1985cropyear_production.csv")
df6=pd.read_csv("1986cropyear_production.csv");df7=pd.read_csv("1987cropyear_production.csv")
df8=pd.read_csv("1988cropyear_production.csv");df9=pd.read_csv("1989cropyear_production.csv")
df10=pd.read_csv("1990cropyear_production.csv");df11=pd.read_csv("1991cropyear_production.csv")
df12=pd.read_csv("1992cropyear_production.csv");df13=pd.read_csv("1993cropyear_production.csv")
df14=pd.read_csv("1994cropyear_production.csv");df15=pd.read_csv("1995cropyear_production.csv")
df16=pd.read_csv("1996cropyear_production.csv");df17=pd.read_csv("1997cropyear_production.csv")
df18=pd.read_csv("1998cropyear_production.csv");df19=pd.read_csv("1999cropyear_production.csv")
df20=pd.read_csv("2000cropyear_production.csv");df21=pd.read_csv("2001cropyear_production.csv")
df22=pd.read_csv("2002cropyear_production.csv");df23=pd.read_csv("2003cropyear_production.csv")
df24=pd.read_csv("2004cropyear_production.csv");df25=pd.read_csv("2005cropyear_production.csv")
df26=pd.read_csv("2006cropyear_production.csv");df27=pd.read_csv("2007cropyear_production.csv")
df28=pd.read_csv("2008cropyear_production.csv");df29=pd.read_csv("2009cropyear_production.csv")
df30=pd.read_csv("2010cropyear_production.csv");df31=pd.read_csv("2011cropyear_production.csv")
df32=pd.read_csv("2012cropyear_production.csv");df33=pd.read_csv("2013cropyear_production.csv")
df34=pd.read_csv("2014cropyear_production.csv");df35=pd.read_csv("2015cropyear_production.csv")




dflist=['df0','df1','df2','df3','df4','df5','df6','df7','df8','df9','df10','df11','df12','df13','df14','df15','df16','df17','df18','df19','df20',
'df21','df22','df23','df24','df25','df26','df27','df28','df29','df30','df31','df32','df33','df34','df35']
for key in dflist:
    df = eval(key)
    df.rename(columns={
                 'Year': 'Year',
                 'Commodity Code': 'Commodity_Code',
                 'Crop Name': 'Crop_Name',
                 'County Code': 'County_Code',
                 'County': 'County',
                 'Harvested Acres': 'Harvested',
                 'Yield': 'Yield',
                 'Production': 'Production',
                 'Price P/U': 'Price',
                 'Value': 'Value'}, inplace=True)


almprod80_f=df0[df0['County'].dropna().str.contains("Fresno") & df0['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod80_k=df0[df0['County'].dropna().str.contains("Kern") & df0['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod81_f=df1[df1['County'].dropna().str.contains("Fresno") & df1['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod81_k=df1[df1['County'].dropna().str.contains("Kern") & df1['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod82_f=df2[df2['County'].dropna().str.contains("Fresno") & df2['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod82_k=df2[df2['County'].dropna().str.contains("Kern") & df2['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod83_f=df3[df3['County'].dropna().str.contains("Fresno") & df3['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod83_k=df3[df3['County'].dropna().str.contains("Kern") & df3['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod84_f=df4[df4['County'].dropna().str.contains("Fresno") & df4['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod84_k=df4[df4['County'].dropna().str.contains("Kern") & df4['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod85_f=df5[df5['County'].dropna().str.contains("Fresno") & df5['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod85_k=df5[df5['County'].dropna().str.contains("Kern") & df5['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod86_f=df6[df6['County'].dropna().str.contains("Fresno") & df6['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod86_k=df6[df6['County'].dropna().str.contains("Kern") & df6['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod87_f=df7[df7['County'].dropna().str.contains("Fresno") & df7['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod87_k=df7[df7['County'].dropna().str.contains("Kern") & df7['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod88_f=df8[df8['County'].dropna().str.contains("Fresno") & df8['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod88_k=df8[df8['County'].dropna().str.contains("Kern") & df8['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod89_f=df9[df9['County'].dropna().str.contains("Fresno") & df9['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod89_k=df9[df9['County'].dropna().str.contains("Kern") & df9['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod90_f=df10[df10['County'].dropna().str.contains("Fresno") & df10['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod90_k=df10[df10['County'].dropna().str.contains("Kern") & df10['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod91_f=df11[df11['County'].dropna().str.contains("Fresno") & df11['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod91_k=df11[df11['County'].dropna().str.contains("Kern") & df11['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod92_f=df12[df12['County'].dropna().str.contains("Fresno") & df12['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod92_k=df12[df12['County'].dropna().str.contains("Kern") & df12['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod93_f=df13[df13['County'].dropna().str.contains("Fresno") & df13['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod93_k=df13[df13['County'].dropna().str.contains("Kern") & df13['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod94_f=df14[df14['County'].dropna().str.contains("Fresno") & df14['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod94_k=df14[df14['County'].dropna().str.contains("Kern") & df14['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod95_f=df15[df15['County'].dropna().str.contains("Fresno") & df15['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod95_k=df15[df15['County'].dropna().str.contains("Kern") & df15['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod96_f=df16[df16['County'].dropna().str.contains("Fresno") & df16['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod96_k=df16[df16['County'].dropna().str.contains("Kern") & df16['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod97_f=df17[df17['County'].dropna().str.contains("Fresno") & df17['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod97_k=df17[df17['County'].dropna().str.contains("Kern") & df17['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod98_f=df18[df18['County'].dropna().str.contains("Fresno") & df18['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod98_k=df18[df18['County'].dropna().str.contains("Kern") & df18['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod99_f=df19[df19['County'].dropna().str.contains("Fresno") & df19['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod99_k=df19[df19['County'].dropna().str.contains("Kern") & df19['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod00_f=df20[df20['County'].dropna().str.contains("Fresno") & df20['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod00_k=df20[df20['County'].dropna().str.contains("Kern") & df20['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod01_f=df21[df21['County'].dropna().str.contains("Fresno") & df21['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod01_k=df21[df21['County'].dropna().str.contains("Kern") & df21['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod02_f=df22[df22['County'].dropna().str.contains("Fresno") & df22['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod02_k=df22[df22['County'].dropna().str.contains("Kern") & df22['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod03_f=df23[df23['County'].dropna().str.contains("Fresno") & df23['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod03_k=df23[df23['County'].dropna().str.contains("Kern") & df23['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod04_f=df24[df24['County'].dropna().str.contains("Fresno") & df24['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod04_k=df24[df24['County'].dropna().str.contains("Kern") & df24['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod05_f=df25[df25['County'].dropna().str.contains("Fresno") & df25['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod05_k=df25[df25['County'].dropna().str.contains("Kern") & df25['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod06_f=df26[df26['County'].dropna().str.contains("Fresno") & df26['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod06_k=df26[df26['County'].dropna().str.contains("Kern") & df26['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod07_f=df27[df27['County'].dropna().str.contains("Fresno") & df27['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod07_k=df27[df27['County'].dropna().str.contains("Kern") & df27['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod08_f=df28[df28['County'].dropna().str.contains("Fresno") & df28['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod08_k=df28[df28['County'].dropna().str.contains("Kern") & df28['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod09_f=df29[df29['County'].dropna().str.contains("Fresno") & df29['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod09_k=df29[df29['County'].dropna().str.contains("Kern") & df29['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod10_f=df30[df30['County'].dropna().str.contains("Fresno") & df30['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod10_k=df30[df30['County'].dropna().str.contains("Kern") & df30['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod11_f=df31[df31['County'].dropna().str.contains("Fresno") & df31['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod11_k=df31[df31['County'].dropna().str.contains("Kern") & df31['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod12_f=df32[df32['County'].dropna().str.contains("Fresno") & df32['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod12_k=df32[df32['County'].dropna().str.contains("Kern") & df32['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod13_f=df33[df33['County'].dropna().str.contains("Fresno") & df33['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod13_k=df33[df33['County'].dropna().str.contains("Kern") & df33['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod14_f=df34[df34['County'].dropna().str.contains("Fresno") & df34['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod14_k=df34[df34['County'].dropna().str.contains("Kern") & df34['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
almprod15_f=df35[df35['County'].dropna().str.contains("Fresno") & df35['Crop_Name'].dropna().str.contains("ALMONDS ALL")];almprod15_k=df35[df35['County'].dropna().str.contains("Kern") & df35['Crop_Name'].dropna().str.contains("ALMONDS ALL")]
#df1.to_csv('example.csv', sep='\t', encoding='utf-8')
with open('1-CROP_PRODUCTION-Fresno.csv', 'w') as f:
    
    almprod80_f.to_csv(f, header=True,index=0)
    almprod81_f.to_csv(f, header=False,index=0)
    almprod82_f.to_csv(f, header=False,index=0)
    almprod83_f.to_csv(f, header=False,index=0)
    almprod84_f.to_csv(f, header=False,index=0)
    almprod85_f.to_csv(f, header=False,index=0)
    almprod86_f.to_csv(f, header=False,index=0)
    almprod87_f.to_csv(f, header=False,index=0)
    almprod88_f.to_csv(f, header=False,index=0)
    almprod89_f.to_csv(f, header=False,index=0)
    almprod90_f.to_csv(f, header=False,index=0)
    almprod91_f.to_csv(f, header=False,index=0)
    almprod92_f.to_csv(f, header=False,index=0)
    almprod93_f.to_csv(f, header=False,index=0)
    almprod94_f.to_csv(f, header=False,index=0)
    almprod95_f.to_csv(f, header=False,index=0)
    almprod96_f.to_csv(f, header=False,index=0)
    almprod97_f.to_csv(f, header=False,index=0)
    almprod98_f.to_csv(f, header=False,index=0)
    almprod99_f.to_csv(f, header=False,index=0)
    almprod00_f.to_csv(f, header=False,index=0)
    almprod01_f.to_csv(f, header=False,index=0)
    almprod02_f.to_csv(f, header=False,index=0)
    almprod03_f.to_csv(f, header=False,index=0)
    almprod04_f.to_csv(f, header=False,index=0)
    almprod05_f.to_csv(f, header=False,index=0)
    almprod06_f.to_csv(f, header=False,index=0)
    almprod07_f.to_csv(f, header=False,index=0)
    almprod08_f.to_csv(f, header=False,index=0)
    almprod09_f.to_csv(f, header=False,index=0)
    almprod10_f.to_csv(f, header=False,index=0)
    almprod11_f.to_csv(f, header=False,index=0)
    almprod12_f.to_csv(f, header=False,index=0)
    almprod13_f.to_csv(f, header=False,index=0)
    almprod14_f.to_csv(f, header=False,index=0)
    almprod15_f.to_csv(f, header=False,index=0)

f.close()

with open('1-CROP_PRODUCTION-Kern.csv', 'w') as f:
    almprod80_k.to_csv(f, header=True,index=0)
    almprod81_k.to_csv(f, header=False,index=0)
    almprod82_k.to_csv(f, header=False,index=0)
    almprod83_k.to_csv(f, header=False,index=0)
    almprod84_k.to_csv(f, header=False,index=0)
    almprod85_k.to_csv(f, header=False,index=0)
    almprod86_k.to_csv(f, header=False,index=0)
    almprod87_k.to_csv(f, header=False,index=0)
    almprod88_k.to_csv(f, header=False,index=0)
    almprod89_k.to_csv(f, header=False,index=0)
    almprod90_k.to_csv(f, header=False,index=0)
    almprod91_k.to_csv(f, header=False,index=0)
    almprod92_k.to_csv(f, header=False,index=0)
    almprod93_k.to_csv(f, header=False,index=0)
    almprod94_k.to_csv(f, header=False,index=0)
    almprod95_k.to_csv(f, header=False,index=0)
    almprod96_k.to_csv(f, header=False,index=0)
    almprod97_k.to_csv(f, header=False,index=0)
    almprod98_k.to_csv(f, header=False,index=0)
    almprod99_k.to_csv(f, header=False,index=0)
    almprod00_k.to_csv(f, header=False,index=0)
    almprod01_k.to_csv(f, header=False,index=0)
    almprod02_k.to_csv(f, header=False,index=0)
    almprod03_k.to_csv(f, header=False,index=0)
    almprod04_k.to_csv(f, header=False,index=0)
    almprod05_k.to_csv(f, header=False,index=0)
    almprod06_k.to_csv(f, header=False,index=0)
    almprod07_k.to_csv(f, header=False,index=0)
    almprod08_k.to_csv(f, header=False,index=0)
    almprod09_k.to_csv(f, header=False,index=0)
    almprod10_k.to_csv(f, header=False,index=0)
    almprod11_k.to_csv(f, header=False,index=0)
    almprod12_k.to_csv(f, header=False,index=0)
    almprod13_k.to_csv(f, header=False,index=0)
    almprod14_k.to_csv(f, header=False,index=0)
    almprod15_k.to_csv(f, header=False,index=0)

f.close()
'''