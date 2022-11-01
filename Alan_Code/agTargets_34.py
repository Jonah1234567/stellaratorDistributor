#!/usr/bin/env python3
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fsolve
from scipy.special import ellipe
from Helpers3 import *
from simsopt.mhd.boozer import Boozer

# QI penalty target for surface "snorm" using Boozer coordinates.
#   This is the target function that yielded the results I've been presenting,
#   so we know that it works. But there's reason to believe that it has some
#   problems that hinder convergence time, so use with caution.
def BoozerBounceResidual1(vmec,snorm,
                nphi=601,nalpha=50,nBj=401,
                mpol=60,ntor=60,
                nphi_out=1000,
                type='P',
                maxmirr=None):
    vmec.run()

    # The output penalty array
    out = np.zeros(nphi_out*nalpha)
    outind = 0

    # Construct Boozer object
    boozer = Boozer(vmec,mpol,ntor)
    boozer.register(snorm)
    boozer.run()

    # Extract relevant quantities from Boozer object
    # Mode numbers
    xm_nyq = boozer.bx.xm_b
    xn_nyq = boozer.bx.xn_b

    # Fourier coefficients
    bmnc = boozer.bx.bmnc_b
    bmns = boozer.bx.bmns_b

    # Stellarator symmetry
    lasym = vmec.wout.lasym

    # Number of field periods
    nfp = boozer.bx.nfp
    # Number of flux surfaces
    ns = vmec.wout.ns

    # Rotational transform 
    iota = UnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:], k=1, s=0)(snorm)

    # The array that will hold the |B| values along these fieldlines
    B = np.zeros((nphi, nalpha))

    # Optimize for an omnigenous stellarator with poloidal contours
    if type == 'P':
        # 2D toroidal and poloidal arrays that correspond to fieldline coordinates
        phis2D = np.tile( np.linspace(0, 2*np.pi/nfp,nphi), (nalpha,1)).T
        thetas2D = np.tile( np.linspace(0, 2*np.pi, nalpha), (nphi,1)) + iota*phis2D

        phimin = 0
        phimax = 2*np.pi/nfp

        # Loop through fourier modes, construct |B| array
        for jmn in range(len(xm_nyq)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * thetas2D - n * phis2D
            cosangle = np.cos(angle)
            if lasym == True:
                sinangle = np.sin(angle)
            B[:,:] += bmnc[jmn] * cosangle[:,:]
            if lasym == True:
                B[:,:] += bmns[jmn] * sinangle[:,:]
    # Optimize for an omnigenous stellarator with toroidal contours
    elif type == 'T':
        # 2D toroidal and poloidal arrays that correspond to fieldline coordinates
        # This is actually an array of thetas, so this should go from 0 to 2pi
        phis2D = np.tile( np.linspace(np.pi, 3*np.pi,nphi), (nalpha,1)).T
        # This is actually an array of phis
        thetas2D = np.tile( np.linspace(0, 2*np.pi/nfp, nalpha), (nphi,1)) + iota*phis2D

        phimin = np.pi
        phimax = 3*np.pi

        # Loop through fourier modes, construct |B| array
        for jmn in range(len(xm_nyq)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * phis2D - n * thetas2D
            if lasym == True:
                sinangle = np.sin(angle)
            B[:,:] += bmnc[jmn] * cosangle[:,:]
            if lasym == True:
                B[:,:] += bmns[jmn] * sinangle[:,:]
    # Optimize for an omnigenous stellarator with helical contours
    elif type == 'H':
        d_phi = 2*np.pi / (iota + nfp)
        d_theta = 2*iota*np.pi / (iota + nfp)

        phimin = 0
        phimax = d_phi
        
        # 2D toroidal and poloidal arrays that correspond to fieldline coordinates
        phis2D = np.tile( np.linspace(0, d_phi, nphi), (nalpha,1) ).T + np.tile( np.linspace(0, 2*np.pi/nfp, nalpha), (nphi,1) )
        thetas2D = np.tile( np.linspace(np.pi, np.pi + d_theta, nphi), (nalpha,1) ).T + np.tile(np.linspace(0, -2*np.pi, nalpha), (nphi,1))

        # Loop through fourier modes, construct |B| array
        for jmn in range(len(xm_nyq)):
            m = xm_nyq[jmn]
            n = xn_nyq[jmn]
            angle = m * thetas2D - n * phis2D
            cosangle = np.cos(angle)
            if lasym == True:
                sinangle = np.sin(angle)
            B[:,:] += bmnc[jmn] * cosangle[:,:]
            if lasym == True:
                B[:,:] += bmns[jmn] * sinangle[:,:]
        
        phis2D = phis2D - np.tile( np.linspace(0, 2*np.pi/nfp, nalpha), (nphi,1) )

    ########################################
    ############# PENALTY HERE #############
    ########################################
    # Find maximum and minimum field strength on surface
    Bmax = np.max(B)
    Bmin = np.min(B)

    # Adjust mirror appropriately
    if maxmirr is not None:
        mirr = (Bmax - Bmin)/(Bmax + Bmin)
        if mirr > maxmirr:
            delta = (Bmax*(1-maxmirr) - Bmin*(1+maxmirr))/2
            Bmax = Bmax + delta
            Bmin = Bmin - delta

    # B-bounces that we will be checking
    Bjs = np.linspace(Bmin,Bmax,nBj+2)
    # Ignore exact Bmax and Bmin, since they will almost certainly only exist at one point
    # TODO: Maybe we can also ignore all values of B within, say, 5% of Bmax? This gives us
    #   a sort of "throwaway region" in which we lose some particles, but maybe we'll get
    #   better wells around bmin as a result?

    #Bjs = Bjs[1:-1]
    nBj = len(Bjs)
    phips = np.linspace(phimin,phimax,2*nphi)
    Bp_arr = np.zeros((nalpha,len(phips)))

    phi12s = np.zeros((nalpha,2*nBj + 2))
    mean_phips = np.zeros((2*nBj-1))
    Bps = np.zeros(2*nBj-1)

    wts = np.zeros(nalpha)
    for ialpha in range(nalpha):
        # Fieldline information
        Ba = 1*B[:,ialpha]
        phiBs = 1*phis2D[:,ialpha]

        # Index of the minimum of B on the fieldline
        indmin = np.argmin(Ba)
        if indmin == 0:
            indmin = 2
        elif indmin == len(Ba)-1:
            indmin = len(Ba)-3

        # Define the left-hand-side of the well as everything to the left of the minimum
        Bl = 1*Ba[:indmin]
        # Make B is continuously decreasing towards Bmin
        for i in range(len(Bl)-1):
            if Bl[i] <= Bl[i+1]:
                jf = len(Bl)-1
                for j in range(i+1,len(Bl)):
                    if Bl[j] < Bl[i]:
                        jf = j
                        break
                dB_di = (Bl[jf] - Bl[i])/(j - i)
                for j in range(i+1,jf):
                    Bl[j] = Bl[i] + dB_di*(j-i)

        # Same process for right-hand-side
        Br = 1*Ba[indmin:]
        for j in range(len(Br)-1,1,-1):
            if Br[j-1] >= Br[j]:
                kf = 0
                for k in range(j-1,1,-1):
                    if Br[k] < Br[j]:
                        kf = k
                        break
                dB_di = (Br[kf] - Br[j])/(kf - j)
                for k in range(j-1,kf,-1):
                    Br[k] = Br[j] + dB_di*(k-j)

        Bl = Bmin + (Bl - Bl[-1]) * (Bmax - Bmin) / (Bl[0] - Bl[-1])
        Br = Bmin + (Br - Br[0]) * (Bmax - Bmin) / (Br[-1] - Br[0])

        # The new (phi,B) fieldline
        Blr = np.concatenate((Bl,Br))

        wts[ialpha] = np.sum(((Blr - Ba)/(Bmax - Bmin))**2)
        interp = UnivariateSpline(phiBs,Blr,k=1,s=0)

        Bp = interp(phips)
        # Populate Bp array with (phi,B)
        Bp_arr[ialpha,:] = 1*Bp

    wts = wts/np.sum(wts)

    for ialpha in range(nalpha):
        Bp = 1*Bp_arr[ialpha,:]
        # Populate bounce distance array
        for j in range(nBj):
            Bj = Bjs[j]
            phi1,phi2,m1,m2 = GetBranches(phips,Bp,Bj,Bmax,Bmin)
            if j == 0:
                mean_phips[nBj - 1] += phi1*wts[ialpha] / 2
                mean_phips[nBj - 1] += (phimax - phi2)*wts[ialpha] / 2
                Bps[nBj - 1] = Bj
            else:
                dl = phi1
                dr = phimax - phi2
                meand = (dl+dr)/2
                mean_phips[nBj - j - 1] += meand*wts[ialpha]
                mean_phips[nBj + j - 1] += (phimax - meand)*wts[ialpha]
                Bps[nBj - j - 1] = Bj
                Bps[nBj + j - 1] = Bj
            phi12s[ialpha,2*j] = phi1
            phi12s[ialpha,2*j+1] = phi2
        
    darr = np.zeros((nalpha,len(Bps)))
    for ialpha in range(nalpha):
        for j in range(nBj-1,-1,-1):
            phi1 = phi12s[ialpha,2*j]
            phi2 = phi12s[ialpha,2*j+1]

            d = ((phi1 - mean_phips[nBj - j - 1]) + (phi2 - mean_phips[nBj + j - 1]))/2
            if j != nBj-1:
                if mean_phips[nBj - j - 1] + d <= mean_phips[nBj - j - 2] + darr[ialpha,nBj - j - 2]:
                    d = mean_phips[nBj - j - 2] + darr[ialpha,nBj - j - 2] - mean_phips[nBj - j - 1] + 0.000001
                if mean_phips[nBj + j - 1] + d >= mean_phips[nBj + j] + darr[ialpha,nBj + j]:
                    d = mean_phips[nBj + j] + darr[ialpha,nBj + j] - mean_phips[nBj + j - 1] - 0.000001
            darr[ialpha,nBj - j - 1] = d
            darr[ialpha,nBj + j - 1] = d

    for ialpha in range(nalpha):
        Bpp = (Bps - Bmin)/(Bmax - Bmin)
        Bi = (B[:,ialpha] - Bmin)/(Bmax - Bmin)

        Bppf = UnivariateSpline(mean_phips + darr[ialpha,:],Bpp,k=1,s=0)
        Bf = UnivariateSpline(phis2D[:,ialpha],Bi,k=1,s=0)

        phis = np.linspace(phimin,phimax,nphi_out)
        
        out[outind:nphi_out+outind] = (Bf(phis) - Bppf(phis))
        outind += nphi_out

    out = out/np.sqrt(len(out))
    return out
    
# Penalize the configuration's mirror ratio
#   If mirror_ratio > t, then the penalty is triggered, else penalty is zero.
# For this reason, if you're using this penalty, I suggest choosing a very
#   large weight, as this will essentially act as a "wall" and prevent the 
#   mirror ratio from exceeding whatever you set as "t"
def MirrorRatioPen(vmec,t=0.21):
    vmec.run()
    xm_nyq = vmec.wout.xm_nyq
    xn_nyq = vmec.wout.xn_nyq
    bmnc = vmec.wout.bmnc.T
    bmns = 0*bmnc
    nfp = vmec.wout.nfp
    
    Ntheta = 100
    Nphi = 100
    thetas = np.linspace(0,2*np.pi,Ntheta)
    phis = np.linspace(0,2*np.pi/nfp,Nphi)
    phis2D,thetas2D=np.meshgrid(phis,thetas)
    b = np.zeros([Ntheta,Nphi])
    for imode in range(len(xn_nyq)):
        angles = xm_nyq[imode]*thetas2D - xn_nyq[imode]*phis2D
        b += bmnc[1,imode]*np.cos(angles) + bmns[1,imode]*np.sin(angles)
    Bmax = np.max(b)
    Bmin = np.min(b)
    m = (Bmax-Bmin)/(Bmax+Bmin)
    print("Mirror =",m)
    pen = np.max([0,m-t])
    return pen

# Penalize the configuration's VMEC aspect ratio
#   If aspect_ratio > t, then the penalty is triggered, else penalty is zero.
# For this reason, if you're using this penalty, I suggest choosing a very
#   large weight, as this will essentially act as a "wall" and prevent the 
#   aspect ratio from exceeding whatever you set as "t"
def AspectRatioPen(vmec,t=10):
    vmec.run()
    asp = vmec.wout.aspect
    print("Aspect Ratio =",asp)
    pen = np.max([0,asp-t])
    return pen

# Penalizes the configuration's maximum elongation
#   If max_elongation > t, then the penalty is triggered, else penalty is zero.
# For this reason, if you're using this penalty, I suggest choosing a very
#   large weight, as this will essentially act as a "wall" and prevent the 
#   elongation from exceeding whatever you set as "t"
def MaxElongationPen(vmec,t=6.0):
    nfp = vmec.wout.nfp
    ntheta = 50
    nphi = int(50)
    # Load variables
    if 1 == 1:
        xm = vmec.wout.xm
        xn = vmec.wout.xn
        rmnc = vmec.wout.rmnc.T
        zmns = vmec.wout.zmns.T
        lasym = vmec.wout.lasym
        raxis_cc = vmec.wout.raxis_cc
        zaxis_cs = vmec.wout.zaxis_cs
        if lasym == True:
            raxis_cs = vmec.wout.raxis_cs
            zaxis_cc = vmec.wout.zaxis_cc
            rmns = vmec.wout.rmns
            zmnc = vmec.wout.zmnc
        else:
            raxis_cs = 0*raxis_cc
            zaxis_cc = 0*zaxis_cs
            rmns = rmnc*0
            zmnc = zmns*0

        # Set up variables
        theta1D = np.linspace(0,2*np.pi,num=ntheta)
        phi1D = np.linspace(0,2*np.pi/nfp,num=nphi)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    def FindBoundary(theta,phi):
        phi = phi[0]
        rb = np.sum(rmnc[-1,:] * np.cos(xm*theta + xn*phi))
        zb = np.sum(zmns[-1,:] * np.sin(xm*theta + xn*phi))
        xb = rb * np.cos(phi)
        yb = rb * np.sin(phi)

        return np.array([xb,yb,zb])

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    # Set up axis
    if 1 == 1:
        Rax = np.zeros(nphi)
        Zax = np.zeros(nphi)
        Raxp = np.zeros(nphi)
        Zaxp = np.zeros(nphi)
        Raxpp = np.zeros(nphi)
        Zaxpp = np.zeros(nphi)
        Raxppp = np.zeros(nphi)
        Zaxppp = np.zeros(nphi)
        for jn in range(len(raxis_cc)):
            n = jn * nfp
            sinangle = np.sin(n * phi1D)
            cosangle = np.cos(n * phi1D)

            Rax += raxis_cc[jn] * cosangle
            Zax += zaxis_cs[jn] * sinangle
            Raxp += raxis_cc[jn] * (-n * sinangle)
            Zaxp += zaxis_cs[jn] * (n * cosangle)
            Raxpp += raxis_cc[jn] * (-n * n * cosangle)
            Zaxpp += zaxis_cs[jn] * (-n * n * sinangle)
            Raxppp += raxis_cc[jn] * (n * n * n * sinangle)
            Zaxppp += zaxis_cs[jn] * (-n * n * n * cosangle)

            if lasym == True:
                Rax += raxis_cs[jn] * sinangle
                Zax += zaxis_cc[jn] * cosangle + zaxis_cs[jn] * sinangle
                Raxp += raxis_cs[jn] * (n * cosangle)
                Zaxp += zaxis_cc[jn] * (-n * sinangle)
                Raxpp += raxis_cs[jn] * (-n * n * sinangle)
                Zaxpp += zaxis_cc[jn] * (-n * n * cosangle)
                Raxppp += raxis_cs[jn] * (-n * n * n * cosangle)
                Zaxppp += zaxis_cc[jn] * (n * n * n * sinangle) 

        Xax = Rax * np.cos(phi1D)
        Yax = Rax * np.sin(phi1D)

        #############################################################################################
        #############################################################################################
        #############################################################################################
        #############################################################################################

        d_l_d_phi = np.sqrt(Rax * Rax + Raxp * Raxp + Zaxp * Zaxp)
        d2_l_d_phi2 = (Rax * Raxp + Raxp * Raxpp + Zaxp * Zaxpp) / d_l_d_phi

        d_r_d_phi_cylindrical = np.array([Raxp, Rax, Zaxp]).transpose()
        d2_r_d_phi2_cylindrical = np.array([Raxpp - Rax, 2 * Raxp, Zaxpp]).transpose()

        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                            + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

        tangent_cylindrical = np.zeros((nphi, 3))
        d_tangent_d_l_cylindrical = np.zeros((nphi, 3))
        for j in range(3):
            tangent_cylindrical[:,j] = d_r_d_phi_cylindrical[:,j] / d_l_d_phi
            d_tangent_d_l_cylindrical[:,j] = (-d_r_d_phi_cylindrical[:,j] * d2_l_d_phi2 / d_l_d_phi \
                                            + d2_r_d_phi2_cylindrical[:,j]) / (d_l_d_phi * d_l_d_phi)

        tangent_R   = tangent_cylindrical[:,0]
        tangent_phi = tangent_cylindrical[:,1]

        tangent_Z   = tangent_cylindrical[:,2]
        tangent_X   = tangent_R * np.cos(phi1D) - tangent_phi * np.sin(phi1D)
        tangent_Y   = tangent_R * np.sin(phi1D) + tangent_phi * np.cos(phi1D)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    #############################################################################################

    Xp = np.zeros(ntheta)
    Yp = np.zeros(ntheta)
    Zp = np.zeros(ntheta)
    
    elongs = np.zeros(nphi)
    
    #for iphi in range(int(np.floor(nphi/2)) - 2, int(np.floor(nphi/2)) - 1):
    for iphi in range(nphi):
        phi = phi1D[iphi]

        # x,y,z components of the axis tangent
        tx = tangent_X[iphi]
        ty = tangent_Y[iphi]
        tz = tangent_Z[iphi]
        t_ = np.array([tx,ty,tz])

        xax = Xax[iphi]
        yax = Yax[iphi]
        zax = Zax[iphi]
        pax = np.array([xax, yax, zax])
        
        for ipt in range(ntheta):
            theta = theta1D[ipt]
            fdot = lambda p : np.dot( t_ , (FindBoundary(theta, p) - pax) )
            phi_x = fsolve(fdot, phi)

            sbound = FindBoundary(theta, phi_x)
            sbound = sbound - np.dot(sbound,t_)

            Xp[ipt] = sbound[0]
            Yp[ipt] = sbound[1]
            Zp[ipt] = sbound[2]

        perim = np.sum(np.sqrt((Xp-np.roll(Xp,1))**2 + (Yp-np.roll(Yp,1))**2 + (Zp-np.roll(Zp,1))**2))
        A = FindArea(Xp,Yp,Zp)

        perim = perim / np.sqrt(A)
        A = 1

        # Area of ellipse = A = pi*a*b
        #   a = semi-major, b = semi-minor
        # b = A / (pi*a)
        # Eccentricity = e = 1 - b**2/a**2
        #                  = 1 - A**2 / (pi**2 * a**4)
        #                  = 1 - (A / (pi * a**2))**2
        # Circumference = C = 4 * a * ellipe(e) --> Use this to solve for semi-major radius a
        #
        # b = A / (pi * a)
        # Elongation = E = semi-major / semi-minor 
        #                = a / b
        #                = a * (pi * a) / A
        #                = pi * a**2 / A
        perim_resid = lambda a : perim - (4*a*ellipe(1 - ( A / (np.pi * a**2 ) )**2))
        if iphi == 0:
            maj = fsolve(perim_resid, 1)
        else:
            maj = fsolve(perim_resid, maj)
        min = A / (np.pi * maj)
        """print("semimajor =",maj)
        print("semiminor =",min)
        print("Area =",np.pi*maj*min)
        e_sq = 1 - min**2/maj**2
        print("Perim =",4*maj*ellipe(e_sq))"""
        elongs[iphi] = np.pi * maj**2 / A

    e = np.max(elongs)
    print("Max Elongation =",e)
    print("Mean Elongation =",np.mean(elongs))
    pen = np.max([0,e-t])
    return pen

