#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR(S) AND VERSION-HISTORY

Author:   Jean-Baptiste Renault 
Copyright 2020 Jean-Baptiste Renault

Pyhton Version:

Coding by Emiliano Ravera, Apr 2024. Adapted to Python 3.11
email:    emiliano.ravera@uner.edu.ar

@author: emi
"""
# ----------- import packages --------------
import numpy as np
from stl import mesh
import pandas as pd
import os, shutil
from pathlib import Path
import sys
import time
import logging

#%% ---------------------------------------------------------------------------
# SUBFUNTIONS
# -----------------------------------------------------------------------------
def Subexpressions (w0 , w1 , w2):
    # -------------------------------------------------------------------------
    
    temp0 = w0 + w1
    f1 = temp0 + w2
    temp1 = w0*w0 
    temp2 = temp1 + w1*temp0
    f2 = temp2 + w2*f1
    f3 = w0*temp1 + w1*temp2 + w2*f2
    g0 = f2 + w0*(f1 + w0)
    g1 = f2 + w1*(f1 + w1)
    g2 = f2 + w2*(f1 + w2)
    
    return w0 , w1 , w2 , f1 , f2 , f3 , g0 , g1 , g2

# -----------------------------------------------------------------------------
def TriInertiaPpties(Tr = {}):
    # -------------------------------------------------------------------------
    # Get inertia of polyhedra of triangular faces
    
    eigVctrs = np.zeros((3,3))
    CenterVol = np.zeros((1,3))
    InertiaMatrix = np.zeros((3,3))
    D = np.zeros((3,3))
    mass = -1
    
    Elmts = Tr['ConnectivityList']
    Points = Tr['Points']
    
    tmp_mesh = mesh.Mesh(np.zeros(Elmts.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(Elmts):
        for j in range(3):
            tmp_mesh.vectors[i][j] = Points[f[j],:]
    
    if tmp_mesh.is_close() == 'False':
        logging.exception('The inertia properties are for hole-free triangulations. \n')
        logging.exception(' Close your mesh before use, try with TriFillPlanarHoles. \n')
        logging.exception(' For 2D mesh use TriMesh2DProperties. \n')
        
    PseudoCenter = np.mean(Points, 0)
    Nodes = Points - PseudoCenter
           
    mult = np.array([1/6, 1/24, 1/24, 1/24, 1/60, 1/60, 1/60, 1/120, 1/120, 1/120])

    intg = np.zeros((1,10))
    
    for trian in tmp_mesh.vectors:
        # vertices of elements #trian
        P1 = trian[0]
        P2 = trian[1]
        P3 = trian[2]
        # Get cross product
        d = np.cross(P2 - P1, P3 - P1)
        
        x0 , x1 , x2 , f1x , f2x , f3x , g0x , g1x , g2x = Subexpressions(P1[0] , P2[0], P3[0])
        y0 , y1 , y2 , f1y, f2y , f3y , g0y , g1y , g2y = Subexpressions(P1[1] , P2[1], P3[1])
        z0 , z1 , z2 , f1z, f2z , f3z , g0z , g1z , g2z = Subexpressions(P1[2] , P2[2], P3[2])
        
        # Update integrals
        intg[0] += d[0]*f1x
        
        intg[2] += d[0]*f2x
        intg[5] += d[0]*f3x
        
        intg[3] += d[1]*f2y
        intg[6] += d[1]*f3y
        
        intg[4] += d[2]*f2z
        intg[7] += d[2]*f3z
        
        intg[8] += d[0]*(y0*g0x + y1*g1x + y2*g2x)
        intg[9] += d[1]*(z0*g0y + z1*g1y + z2*g2y)
        intg[10] += d[2]*(x0*g0z + x1*g1z + x2*g2z)
        
    intg *= mult
            
    mass = intg[0]
    
    CenterVol[0] = intg[1]/mass
    CenterVol[1] = intg[2]/mass
    CenterVol[2] = intg[3]/mass
    
    InertiaMatrix[0,0] = intg[5] + intg[6] - mass*((np.linalg.norm(CenterVol[0, 1:]))**2) # CenterVol([2 3])
    InertiaMatrix[1,1] = intg[4] + intg[6] - mass*((np.linalg.norm(CenterVol[0, 0:3:2]))**2) # CenterVol([3 1]) o CenterVol([1 3]) 
    InertiaMatrix[2,2] = intg[4] + intg[5] - mass*((np.linalg.norm(CenterVol[0, 0:2]))**2) # CenterVol([1 2])
    
    InertiaMatrix[0,1] = -(intg[7] - mass*(CenterVol[0, 0]*CenterVol[0, 1]))
    InertiaMatrix[1,2] = -(intg[8] - mass*(CenterVol[0, 1]*CenterVol[0, 2]))
    InertiaMatrix[2,0] = -(intg[9] - mass*(CenterVol[0, 2]*CenterVol[0, 0]))
    
    i_lower = np.tril_indices(3, -1)
    InertiaMatrix[i_lower] = InertiaMatrix.T[i_lower] # make the matrix symmetric
    
    CenterVol += PseudoCenter.T
    
    eigValues, eigVctrs = np.linalg.eig(InertiaMatrix)
    D = np.diag(eigValues)
    
    return eigVctrs, CenterVol, InertiaMatrix, D, mass 

# -----------------------------------------------------------------------------







#%% ---------------------------------------------------------------------------

