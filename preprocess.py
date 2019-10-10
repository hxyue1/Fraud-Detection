import time
import pandas as pd
import numpy as np

def fill_nans(df):
    t0 = time.time()
    
    na_count = df.isna().sum().sum()
    while na_count>0:
        df = df.sample(frac=1)
        df = df.fillna(method='ffill',limit=1)
        na_count = df.isna().sum().sum()

    
    df = df.sort_index()
    t1 = time.time()

    return(df)
    print(t1-t0)
    
def feature_creation(categorical, numerical, method, df):
    
    #Creating some features by default because they will probably be needed anyway
    means_temp = df.groupby(categorical)[numerical].agg(['mean']).to_dict()
    means = df[categorical].map(means_temp['mean'])
    
    stds_temp = df.groupby(categorical)[numerical].agg(['std']).to_dict()
    stds = df[categorical].map(stds_temp['std'])
    
    
    if method == 'counts':
        counts_temp = df[categorical].value_counts().to_dict()
        counts = df[categorical].map(counts_temp)
        return(counts)
    
    if method == 'means':
        return(means)
    
    if method == 'stds':
        return(stds)
    
    if method == "devs":
        devs = df[numerical] - means
        return(devs)
    
    if method == "std_devs":
        devs = df[numerical] - means
        std_devs = devs/stds
        return(std_devs)
    
def feature_aggregation_creation(combination_list, df):
    import pandas as pd
    out_df = pd.DataFrame(
        {'temp':np.zeros(len(df))}
    )
    
    for i in np.arange(0,len(combination_list)):
        combination = combination_list[i]
        
        print(combination)
        feature = feature_creation(
            categorical = combination[0],
            numerical = combination[1],
            method = combination[2],
            df=df)
        
        name = combination[0] + '.' + combination[1] + '.' + combination[2]
        out_df[name] = feature
        
    out_df.drop('temp',axis=1,inplace=True)
    return(out_df)    