from simsopt.mhd.vmec import Vmec
from agTargets_34 import *

# Define VMEC (magnetic equilibrium) object
vmec = Vmec('input.test1')
vmec.run()

# The objective function penalty
def obj(vmec):
    out = 0.0
    out += 1.0e2 * MirrorRatioPen(vmec, t=0.16)**2
    out += 1.0e2 * AspectRatioPen(vmec, t=10.0)**2
    out += 1.0e2 * MaxElongationPen(vmec, t=6.0)**2

    sarr = [1/51, 25/51, 51/51]
    for s in sarr:
        out += np.sum( BoozerBounceResidual1(vmec, s, type='P',mpol=30,ntor=30)**2 )
    
    return out

# Get objective function output
obj(vmec)
