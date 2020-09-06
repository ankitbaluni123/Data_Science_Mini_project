import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
df=pd.read_csv("E:\IC-2XX(lab)\MINI PROJECT\Batch07\group7.csv")
#print(df.quantile(0.25)
def replace_outliers(col,df):
    df1=df.copy()
    for i in range(1,len(col)):
        q1=df1.quantile(0.25)[i-1]
        q3=df1.quantile(0.75)[i-1]
        iqr=q3-q1
        count=0
        count1=0
        
        for j in range(len(df1)):
            if(df1[col[i]][j]<(q1-1.5*iqr) or df1[col[i]][j]>(q3+1.5*iqr)):
                count=count+1
#        print("outliers in",col[i]," = ",count)
        df1[col[i]].loc[df1[col[i]]>1.5*iqr+q3]=df1[col[i]].median()
        df1[col[i]].loc[df1[col[i]]<-1.5*iqr+q1]=df1[col[i]].median()
        for j in range(len(df1)):
            if(df1[col[i]][j]<(q1-1.5*iqr) or df1[col[i]][j]>(q3+1.5*iqr)):
                count1=count1+1
#        print("outliers in",col[i]," after correction  = ",count1)
    df1.to_csv("outliers_removed.csv",index=None)
    return df1;
##################################################
def min_max_normalization(col,df):
    df2=df.copy()
    for i in range(1,len(col)):
        df2[col[i]] = (df2[col[i]]-df2[col[i]].min())/(df2[col[i]].max()-df2[col[i]].min())
    df2.to_csv("normalized.csv",index=None)    
#    print(df1)
    return df2;  
######################################################  
def standardization(col,df):
    df3=df.copy()
    for i in range(1,len(col)):
        df3[col[i]]=(df3[col[i]]-df3[col[i]].mean())/(df3[col[i]].std())
    df3.to_csv("standardised.csv",index=None)
#    print(df2)    
    return df3;  
######################################################  
def dim_reduction(df,dim):
    df4=df.copy()
    f1=df4["InBandwidth"]
    g1=df4["CreationTime"]
    h1=df4.drop(columns=["InBandwidth","CreationTime"])
    pca = PCA(n_components=dim)
    principalComponents = pca.fit_transform(h1)
    principalDf = pd.DataFrame(data = principalComponents,columns=["principal component {}".format(i) for i in  range(1,dim+1)])
    finaldf = pd.concat([g1,principalDf,f1],axis = 1)
    finaldf.to_csv("dim_reduced_to_"+str(dim)+".csv")
    return finaldf;    
    
########################################################33    
p=replace_outliers(df.columns,df)  ### p is datframe that has outliers removed values 
q=min_max_normalization(p.columns,p) ### q is dataframe that has (outliers removed + normalized) values 
r=standardization(p.columns,p)  #### r is dataframe that has (outliers removed + standardized) values 
s1=dim_reduction(p,1)   ### s1 is dataframe in which dimensions reduced to 1
s2=dim_reduction(p,2)   ### s2 is dataframe in which dimensions reduced to 2
s3=dim_reduction(p,3)  ### s3 is dataframe in which dimensions reduced to 3
s4=dim_reduction(p,4)  ### s4 is dataframe in which dimensions reduced to 4
s5=dim_reduction(p,5)  ### s5 is dataframe in which dimensions reduced to 5
s6=dim_reduction(p,6)  ### s6 is dataframe in which dimensions reduced to 6

#print(q)


