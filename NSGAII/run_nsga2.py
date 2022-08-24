import sys
[ptype,num,freq,delta,rho,cond,r,k,maxeval,npop,noff,crosstype,probcr]=sys.argv[1:]
from phoenix_nsga2 import phoenix_nsga2
k = int(k)
k = k - 1
phoenix_nsga2(ptype,num,freq,delta,rho,cond,r,k,maxeval,npop,noff,crosstype,probcr)