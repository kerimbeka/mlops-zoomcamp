import pandas as pd

from scipy import stats
from sklearn.model_selection import train_test_split

def data_cleaning():
    #read the raw data
    df = pd.read_excel('../data/Bank_Personal_Loan_Modelling.xlsx', sheet_name='Data')
    
    #drop the id column
    df.drop('ID', axis=1, inplace=True)

    # drop the noisy ZIP Code
    df.drop(df[df['ZIP Code']<20000].index, inplace=True)
    df.reset_index(drop=True, inplace =True)

    # There were negative Experience. Probably incorrect record
    df['Experience'] = df['Experience'].apply(abs)

    # Outlier Treatment
    outlier_indexes = df[stats.zscore(df['Mortgage'])>3].index
    df.drop(outlier_indexes, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # convert average monthly credit card spending to annual
    df['CCAvg'] = df['CCAvg']*12

    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Personal Loan'])
    df_test, df_val = train_test_split(df_test, test_size=0.5, stratify=df_test['Personal Loan'])

    df_val.to_csv("../data/val.csv")
    df_train.to_csv("../data/train.csv")
    df_test.to_csv("../data/test.csv")

if __name__ == "__main__":
    data_cleaning()

