abs_step = 1e-3
rel_step = 1e-4
diff_method = 'centered'

from simsopt.mhd.vmec import Vmec
from simsopt import make_optimizable
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.util.mpi import MpiPartition
from simsopt.solve.mpi import least_squares_mpi_solve
from agTargets_34 import *

vmec = Vmec('input.test1')
vmec.run()
surf = vmec.boundary
mpi = MpiPartition() # I'm confident there's a way to do this without MPI, I just don't have the code written
mpi.write()

prob = LeastSquaresProblem.from_tuples([
            (make_optimizable(MirrorRatioPen, vmec, t=0.20).J, 0.0, 1.0e2),
            (make_optimizable(AspectRatioPen, vmec, t=8.0).J, 0.0, 1.0e2),
            (make_optimizable(MaxElongationPen, vmec, t=6.0).J, 0.0, 1.0e2),

            (make_optimizable(BoozerBounceResidual1,vmec,1/51,type='P',mpol=30,ntor=30).J, 0.0, 1.0),
            (make_optimizable(BoozerBounceResidual1,vmec,250/51,type='P',mpol=30,ntor=30).J, 0.0, 1.0),
            (make_optimizable(BoozerBounceResidual1,vmec,51/51,type='P',mpol=30,ntor=30).J, 0.0, 1.0)
            ])

# If true, this script optimizes only the most "important" modes.
# If false, this script optimizes many more modes, some less important.
smaller_opt = True

if smaller_opt == True:
    surf.unfix('rc(0,2)')
    surf.unfix('zs(0,2)')

    surf.unfix('rc(1,0)')
    surf.unfix('zs(1,0)')

    surf.unfix('rc(1,1)')
    surf.unfix('zs(1,1)')

    surf.unfix('rc(1,-1)')
    surf.unfix('zs(1,-1)')

    surf.unfix('rc(1,2)')
    surf.unfix('zs(1,2)')
else:
    surf.fixed_range(mmin= 0, mmax=2, 
                    nmin=-2, nmax=2, fixed=False)

    surf.fix("rc(0,0)")  # Major radius
    surf.fix("zs(0,1)")  # Do not allow rotation about the x-axis

least_squares_mpi_solve(prob, mpi, grad=True,
            abs_step=abs_step,
            rel_step=rel_step,
            diff_method=diff_method)