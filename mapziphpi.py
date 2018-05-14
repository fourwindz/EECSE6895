import pathlib
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('figure', figsize=(20, 10))

zipcode_filename = "tl_2017_us_zcta510.zip"
zipcode_file = pathlib.Path(zipcode_filename)

zipcode_gdf = gpd.read_file(f"zip://{zipcode_file}")

exclude = ['000','001','002','003','004','005','006','007','008','009',  \
           '090','091','092','093','094','095','096','097','098','099',  \
           '962','963','964','965','966','967','968','969',\
           '995','996','997','998','999']

zipcode_gdf['zip3'] = zipcode_gdf['GEOID10'].str[:3]
print(zipcode_gdf.head())

mask = zipcode_gdf['zip3'].isin(exclude)
zipcode_gdf = zipcode_gdf[~mask]           

col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
        'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
        'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
        'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd'];

col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
          'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
          'LastInstallDate','ForeclosureDate','DispositionDate','PPRC','AssetRecCost','MHRC',
          'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
          'FPWA','ServicingIndicator'];

df_acq = pd.read_csv('/Users/jcorrea/Documents/AdvBigData/fannie/Acquisition_2007Q4.txt', sep='|', names=col_acq, index_col=False)
df_per = pd.read_csv('/Users/jcorrea/Documents/AdvBigData/fannie/Performance_2007Q4.txt', sep='|', names=col_per, usecols=[0, 15], index_col=False)

df_per.drop_duplicates(subset='LoanID', keep='last', inplace=True)
df = pd.merge(df_acq, df_per, on='LoanID', how='inner')

df.rename(index=str, columns={"ForeclosureDate": 'Default'}, inplace=True)

df['Default'].fillna(0, inplace=True)
df.loc[df['Default'] != 0, 'Default'] = 1

df['Default'] = df['Default'].astype(int)

#print(df.apply(lambda x: x.isnull().sum(), axis=0))
print(df.groupby('Default').count())

import sys
sys.exit(0)

df.drop(['MortInsPerc','MortInsType','CoCreditScore'], axis=1, inplace=True)
df.dropna(inplace=True)

color_values=[]
for index in range(10):
    color_values.append('#%02x%02x%02x' % (204, 204-int(204/(index+1)), 204-int(204/(index+1))))
   
zip_default = df.groupby("Zip")['Default'].sum()    

#################################################################
maxval = zip_default.max()
base = zipcode_gdf.plot(color='#CCCCCC')
for row in zip_default.iteritems():
    color_index = int(row[1]*9/maxval)
    if (str(row[0]) not in exclude) & (row[1] > 0) & (color_index > 0):
        df = zipcode_gdf.loc[str(row[0]) == zipcode_gdf['zip3']]
        if not df.empty:
            df.plot(ax=base, color = color_values[color_index])

plt.savefig('zip.png')


