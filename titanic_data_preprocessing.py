# DATA PRE-PROCESSING & FEATURE ENGINEERING

import pandas as pd
import numpy as np
import data_prep
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
df = pd.read_csv(r"...\titanic.csv")

df.columns = [col.upper() for col in df.columns]

# 1. Feature Engineering
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df['NEW_TITLE'] = df.NAME.str.extract(r'([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) &
                                              (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) &
                                                (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and
               dataframe[col].dtypes != "O"]

cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > 20 and
               dataframe[col].dtypes == "O"]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

num_cols = [col for col in num_cols if col not in num_but_cat]


num_cols = [col for col in num_cols if "PASSENGERID" not in col]

# 2. Outliers
outlier_true_cols = []
for col in num_cols:
    if data_prep.check_outlier(df, col):
        outlier_true_cols.append(col)

if len(outlier_true_cols) > 0:
    for col in outlier_true_cols:
        data_prep.replace_with_thresholds(df, col)
    print(
        f'The outliar containing columns were:{outlier_true_cols} and the outliar values were replaced by thresholds.')

# 3. Missing Values
data_prep.missing_values_table(df)
df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) &
                                              (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) &
                                                (df['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

# 4. Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = data_prep.label_encoder(df, col)

# 5. Rare Encoding
data_prep.rare_analyser(df, "SURVIVED", cat_cols)
df = data_prep.rare_encoder(df, 0.01, cat_cols)

# 6. One-Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = data_prep.one_hot_encoder(df, ohe_cols)
cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and
               dataframe[col].dtypes != "O"]

cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > 20 and
               dataframe[col].dtypes == "O"]

cat_cols = cat_cols + num_but_cat

cat_cols = [col for col in cat_cols if col not in cat_but_car]

num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

num_cols = [col for col in num_cols if col not in num_but_cat]
num_cols = [col for col in num_cols if "PASSENGERID" not in col]
data_prep.rare_analyser(df, "SURVIVED", cat_cols)

# 7. MinMax Scaler
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])


df_prep.head()
