from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as mp

from imblearn.combine import SMOTEENN

col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
        'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
        'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
        'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd'];

col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
          'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
          'LastInstallDate','ForeclosureDate','DispositionDate','PPRC','AssetRecCost','MHRC',
          'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
          'FPWA','ServicingIndicator'];

print ("read")    
qtr = "2007Q4"       
df_acq = pd.read_csv('C:/tmp/Acquisition_'+qtr+'.txt', sep='|', names=col_acq, index_col=False)
print ("read1")
df_per = pd.read_csv('C:/tmp/Performance_'+qtr+'.txt', sep='|', names=col_per, usecols=[0, 15], index_col=False)
print ("read2")

df_per.drop_duplicates(subset='LoanID', keep='last', inplace=True)
df = pd.merge(df_acq, df_per, on='LoanID', how='inner')

df.rename(index=str, columns={"ForeclosureDate": 'Default'}, inplace=True)

df['Default'].fillna(0, inplace=True)
df.loc[df['Default'] != 0, 'Default'] = 1

df['Default'] = df['Default'].astype(int)

print (df)

print(df.apply(lambda x: x.isnull().sum(), axis=0))

####################################
mp.figure()
sns.countplot(df['Default'])
mp.savefig('default'+qtr+'.png')

####################################
mp.figure()
columns = ['OrCLTV','DTIRat','CreditScore','OrInterestRate']

fig, axes = mp.subplots(nrows=2, ncols=2, figsize=(6,7))
mp.tight_layout(w_pad=2.0, h_pad=3.0)

for i, column in zip(range(1,5), columns):
    mp.subplot(2,2,i)
    sns.boxplot(x="Default", y=column, data=df, linewidth=0.5)
    mp.xlabel('Default')
mp.savefig('defaults'+qtr+'.png')

####################################
mp.figure()    
data = df.loc[df['Zip'].isin(df['Zip'].value_counts().index.tolist()[:10])]

ptab = pd.pivot_table(data, index='Zip', columns='Default', aggfunc='size')
xtab = ptab.div(ptab.sum(axis=1), axis=0)
xtab.plot.barh(stacked=True, figsize=(6,4))
mp.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
mp.xlabel('Fraction of Borrowers')
mp.ylabel('ZIP Code')
mp.savefig('zip'+qtr+'.png')

####################################

df['OrDateMonth'] = df['OrDate'].apply(lambda x: x.split('/')[0].strip()).astype(int)
df['OrDateYear'] = df['OrDate'].apply(lambda x: x.split('/')[1].strip()).astype(int)

df['FirstMonth'] = df['FirstPayment'].apply(lambda x: x.split('/')[0].strip()).astype(int)
df['FirstYear'] = df['FirstPayment'].apply(lambda x: x.split('/')[1].strip()).astype(int)

df.drop(['OrDate','FirstPayment'], axis=1, inplace=True)
    
print ("here")
df.drop(['MortInsPerc','MortInsType','CoCreditScore','ProductType','LoanID',"DTIRat"], axis=1, inplace=True)
df = df[df.OrCLTV.notnull()]
df = df[df.NumBorrow.notnull()]
df = df[df.OrInterestRate.notnull()]
df = df[df.CreditScore.notnull()]

print ("herea")
def getdummies(df):
    columns = df.columns[df.isnull().any()]
    nan_cols = df[columns]

    df.drop(nan_cols.columns, axis=1, inplace=True)

    cat = df.select_dtypes(include=['object'])
    num = df.drop(cat.columns, axis=1)

    data = pd.DataFrame()
    for i in cat.columns:
        tmp = pd.get_dummies(cat[i], drop_first=True)
        data = pd.concat([data, tmp], axis=1)

    df = pd.concat([num,data,nan_cols], axis=1).reset_index(drop=True)
    return df

def fillnan(df):
    columns = df.columns[df.isnull().any()]
    for name in columns:
        y = df.loc[df[name].notnull(), name].values
        X = df.loc[df[name].notnull()].drop(columns, axis=1).values
        X_test = df.loc[df[name].isnull()].drop(columns, axis=1).values
        if df[name].dtypes == 'object':
            print (name, "object")
            model = RandomForestClassifier(n_estimators=400, max_depth=3)
            model.fit(X, y)
            df.loc[df[name].isnull(), name] = model.predict(X_test)
        else:
            print (name, "number")
            model = RandomForestRegressor(n_estimators=400, max_depth=3)
            model.fit(X, y)
            df.loc[df[name].isnull(), name] = model.predict(X_test)
    return df

df = getdummies(df)
print ("hereb")
df = fillnan(df)
print ("here1")

sm = SMOTEENN()

y = df['Default'].values
X = df.drop(['Default'], axis=1).values

X_resampled, y_resampled = sm.fit_sample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.25, random_state=0)

print ("here2")
model = RandomForestClassifier(n_estimators=200)
print ("here2a")
model = model.fit(X_train, y_train)
print ("here2b")
predict = model.predict(X_test)

####################################
mp.figure()
print(classification_report(y_test, predict))

cm = confusion_matrix(y_test, predict).T
cm = cm.astype('float')/cm.sum(axis=0)

fig, ax = mp.subplots()
sns.heatmap(cm, annot=True, cmap='Blues');
ax.set_xlabel('True Label')
ax.set_ylabel('Predicted Label')
ax.xaxis.set_label_position('top')
mp.savefig('confusion'+qtr+'.png')

####################################
mp.figure()
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
roc_auc = roc_auc_score(y_test, predict)

mp.plot(fpr, tpr, lw=1, label='AUC = %0.2f'%(roc_auc))
mp.plot([0, 1], [0, 1], '--k', lw=1)
mp.xlabel('False Positive Rate')
mp.ylabel('True Positive Rate')
mp.title('Random Forest ROC')
mp.legend(loc="lower right", frameon = True).get_frame().set_edgecolor('black')
mp.savefig('ROC'+qtr+'.png')

####################################
mp.figure()

feat_labels = df.drop('Default', axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

ncomp = 20
sns.barplot(x=feat_labels[indices[:ncomp]], y=importances[indices[:ncomp]], color=sns.xkcd_rgb["pale red"])
mp.title('Top 10 Feature Importances')
mp.ylabel('Relative Feature Importance')
mp.xticks(rotation=90)
mp.savefig('features'+qtr+'.png')
