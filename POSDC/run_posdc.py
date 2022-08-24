import sys
[ptype,num,freq,delta,rho,cond,r,k,maxeval]=sys.argv[1:]
from phoenix_posdc import phoenix_posdc
print('{},{},{},{},{},{},{} \n'.format(str(ptype),str(freq),str(delta),str(rho),str(cond),str(r),str(k)))
k = int(k)
k = k-1
phoenix_posdc(ptype,num,freq,delta,rho,cond,r,k,maxeval)