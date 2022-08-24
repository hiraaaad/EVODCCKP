import pandas as pd
import re
import numpy as np
from numba import jit

df=pd.DataFrame(columns=['weight','profit'])
def readttp(ProblemName):
    t = 0
    # os.chdir('..')
    f = open('EIL101/' + ProblemName, 'r')
    lines=f.readlines()
    f.close()
    InitialCapacity=int(re.findall(r'\d+',lines[4])[0]) # extract the number from line 4 which denotes the capacity of instance
    for i in range(112,len(lines)):
        df.loc[t]=[int(re.findall(r'\d+', lines[i])[2])+100,int(re.findall(r'\d+', lines[i])[1])] # + shifted capacity
        t+=1
    return (df,InitialCapacity)


@jit
def dynp(w,p,m,TotalW):
    n = len(w)
    # m=np.zeros([n+1,TotalW+1])
    for i in range(1,n+1):
        for j in range(1,TotalW+1):
            if j-1<w[i-1]:
                m[i,j]=m[i-1,j]
            else:
                m[i,j]=max(m[i-1,j],p[i-1]+m[i-1,j-w[i-1]])
    return m



def findk(woriginal,InitialCapacity):
    woriginal=np.sort(woriginal, kind='mergesort')
    k=sum(np.cumsum(woriginal)<InitialCapacity)
    return InitialCapacity+k*100





