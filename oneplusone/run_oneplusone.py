import sys
[ptype,num,freq,delta,rho,cond,r,k,maxeval]=sys.argv[1:]
from phoenix_oneplusone import phoenix_oneplusone
k = int(k)
k = k - 1
phoenix_oneplusone(ptype,num,freq,delta,rho,cond,r,k,maxeval)