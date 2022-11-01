import numpy as np

# A helper function that outputs the values of phiBs when Ba=Bj
def GetBranches(phiBs,Ba,Bj,Bmax,Bmin):
    diffs = Ba - Bj
            
    # The only case when this is negative is when B = Bj somewhere between two indices
    diffsgn = diffs[0:-1]*diffs[1:]

    # Indices where Bj crosses B
    inds = np.where(diffsgn<0)[0]
    inds = np.sort(inds)

    # ind1 = first crossing, ind2 = second crossing
    if np.abs(Bj - Bmin) < 1e-15 or Bj < Bmin:
        imin = np.argmin(Ba)
        phimin = phiBs[imin]
        return phimin,phimin,imin,imin
    elif np.abs(Bj - Bmax) < 1e-15 or Bj > Bmax:
        return 0, phiBs[-1], 0, len(Ba)-1
    
    if len(inds) != 2:
        inds = np.where((np.abs(diffsgn) < 1e-15) + (diffsgn<0))[0]
        for iind in range(1,len(inds)):
            if inds[iind] != inds[iind-1]+1:
                inds = [inds[iind-1],inds[-1]]
                break

    if len(inds) == 1:
        if Bj > Ba[-1]:
            ind1 = inds[0]
            dy1 = Ba[ind1] - Ba[ind1+1]
            dx1 = phiBs[ind1] - phiBs[ind1+1]
            m1 = dy1/dx1
            b1 = Ba[ind1] - m1*phiBs[ind1]
            if m1 != 0:
                phiB1 = (Bj - b1)/m1
            else:
                phiB1 = phiBs[ind1]
            return phiB1, phiBs[-1], m1, 0
        elif Bj > Ba[0]:
            ind2 = inds[0]
            dy2 = Ba[ind2] - Ba[ind2+1]
            dx2 = phiBs[ind2] - phiBs[ind2+1]
            m2 = dy2/dx2
            b2 = Ba[ind2] - m2*phiBs[ind2]
            if m2 != 0:
                phiB2 = (Bj - b2)/m2
            else:
                phiB2 = phiBs[ind2+1]
            return 0,phiB2, 0, m2

    ind1 = inds[0]
    ind2 = inds[1]

    # Linearly interpolate to find first crossing
    dy1 = Ba[ind1] - Ba[ind1+1]
    dx1 = phiBs[ind1] - phiBs[ind1+1]
    m1 = dy1/dx1
    b1 = Ba[ind1] - m1*phiBs[ind1]
    if m1 != 0:
        phiB1 = (Bj - b1)/m1
    else:
        phiB1 = phiBs[ind1]
    phiB1 = (Bj - b1)/m1

    # Linearly interpolate to find second crossing
    dy2 = Ba[ind2] - Ba[ind2+1]
    dx2 = phiBs[ind2] - phiBs[ind2+1]
    m2 = dy2/dx2
    b2 = Ba[ind2] - m2*phiBs[ind2]
    if m2 != 0:
        phiB2 = (Bj - b2)/m2
    else:
        phiB2 = phiBs[ind2+1]

    return phiB1, phiB2, m1, m2

#unit normal vector of plane defined by points a, b, and c
def FindUnitNormal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def FindArea(X,Y,Z):
    total = [0,0,0]

    for i in range(len(X)):
        x1 = X[i]
        y1 = Y[i]
        z1 = Z[i]
        
        x2 = X[(i+1)%(len(X))]
        y2 = Y[(i+1)%(len(Y))]
        z2 = Z[(i+1)%(len(Z))]

        vi1 = [x1,y1,z1]
        vi2 = [x2,y2,z2]

        prod = np.cross(vi1,vi2)
        total += prod
    pt0 = [X[0], Y[0], Z[0]]
    pt1 = [X[1], Y[1], Z[1]]
    pt2 = [X[2], Y[2], Z[2]]
    result = np.dot(total,FindUnitNormal(pt0,pt1,pt2))
    return abs(result/2)
