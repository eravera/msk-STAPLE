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
from pykdtree.kdtree import KDTree
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


#%% ---------------------------------------------------------------------------
# SUBFUNTIONS
# TriangulationFun
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
    
    # if not tmp_mesh.is_close(): # == 'False'
    #     logging.exception('The inertia properties are for hole-free triangulations. \n')
    #     logging.exception(' Close your mesh before use, try with TriFillPlanarHoles. \n')
    #     logging.exception(' For 2D mesh use TriMesh2DProperties. \n')
    
    # update normals
    tmp_mesh.update_normals()
    
    PseudoCenter = np.mean(Points, 0)
    Nodes = Points - PseudoCenter
           
    mult = np.array([1/6, 1/24, 1/24, 1/24, 1/60, 1/60, 1/60, 1/120, 1/120, 1/120])

    intg = np.zeros(10)
    
    for trian in Elmts:
        # vertices of elements #trian
        P1 = Nodes[trian[0]]
        P2 = Nodes[trian[1]]
        P3 = Nodes[trian[2]]
        # Get cross product
        d = np.cross(P2 - P1, P3 - P1)
        
        x0 , x1 , x2 , f1x , f2x , f3x , g0x , g1x , g2x = Subexpressions(P1[0] , P2[0], P3[0])
        y0 , y1 , y2 , f1y, f2y , f3y , g0y , g1y , g2y = Subexpressions(P1[1] , P2[1], P3[1])
        z0 , z1 , z2 , f1z, f2z , f3z , g0z , g1z , g2z = Subexpressions(P1[2] , P2[2], P3[2])
        
        # Update integrals
        intg[0] += d[0]*f1x
        
        intg[1] += d[0]*f2x
        intg[4] += d[0]*f3x
        
        intg[2] += d[1]*f2y
        intg[5] += d[1]*f3y
        
        intg[3] += d[2]*f2z
        intg[6] += d[2]*f3z
        
        intg[7] += d[0]*(y0*g0x + y1*g1x + y2*g2x)
        intg[8] += d[1]*(z0*g0y + z1*g1y + z2*g2y)
        intg[9] += d[2]*(x0*g0z + x1*g1z + x2*g2z)
        
    intg *= mult
            
    mass = intg[0]
    
    CenterVol[0,0] = intg[1]/mass
    CenterVol[0,1] = intg[2]/mass
    CenterVol[0,2] = intg[3]/mass
    
    InertiaMatrix[0,0] = intg[5] + intg[6] - mass*((np.linalg.norm(CenterVol[0, 1:]))**2) # CenterVol([2 3])
    InertiaMatrix[1,1] = intg[4] + intg[6] - mass*((np.linalg.norm(CenterVol[0, 0:3:2]))**2) # CenterVol([3 1]) o CenterVol([1 3]) 
    InertiaMatrix[2,2] = intg[4] + intg[5] - mass*((np.linalg.norm(CenterVol[0, 0:2]))**2) # CenterVol([1 2])
    
    InertiaMatrix[0,1] = -(intg[7] - mass*(CenterVol[0, 0]*CenterVol[0, 1]))
    InertiaMatrix[1,2] = -(intg[8] - mass*(CenterVol[0, 1]*CenterVol[0, 2]))
    InertiaMatrix[0,2] = -(intg[9] - mass*(CenterVol[0, 2]*CenterVol[0, 0]))
    
    i_lower = np.tril_indices(3, -1)
    InertiaMatrix[i_lower] = InertiaMatrix.T[i_lower] # make the matrix symmetric
    
    CenterVol += PseudoCenter.T
    
    eigValues, eigVctrs = np.linalg.eig(InertiaMatrix)
    # sort the resulting vector in descending order
    idx = eigValues.argsort()
    eigValues = eigValues[idx]
    eigVctrs = eigVctrs[:,idx]
    
    D = np.diag(eigValues)
    
    return eigVctrs.T, CenterVol.T, InertiaMatrix, D, mass 

# -----------------------------------------------------------------------------
def TriMesh2DProperties(Tr = {}):
    # -------------------------------------------------------------------------
    # Compute some 2D properties of a triangulation object
    Properties = {}
    # Convert tiangulation dict to mesh object
    Tr_mesh = mesh.Mesh(np.zeros(Tr['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(Tr['ConnectivityList']):
        for j in range(3):
            Tr_mesh.vectors[i][j] = Tr['Points'][f[j],:]
    # update normals
    Tr_mesh.update_normals()
    
    # get triangle object name
    Properties['Name'] = f'{Tr=}'.partition('=')[0]
    
    # compute the area for each trinagle of the mesh
    Properties['Areas'] = Tr_mesh.areas
    
    # compute the total area of the mesh
    Properties['TotalArea'] = np.sum(Properties['Areas'])
    
    # compute the center of the mesh
    Properties['Center'] = np.sum(Tr_mesh.centroids*Properties['Areas']/Properties['TotalArea'], 0)
    
    # compute the mean normal vector for each normal vector from triangles' mesh
    tmp_meanNormal = np.sum(Tr_mesh.get_unit_normals()*Properties['Areas']/Properties['TotalArea'], 0)
    Properties['meanNormal'] = tmp_meanNormal/np.linalg.norm(tmp_meanNormal)
    
    # compute the coordinates of the nearest neighbor to the center of the mesh.
    tree = KDTree(Tr['Points'])
    pts = np.array([Properties['Center']])
    dist, idx = tree.query(pts)

    Tr['onMeshCenter'] = Tr['Points'][idx]
            
    return Properties

# -----------------------------------------------------------------------------
def TriChangeCS(Tr = {}, V = np.zeros((3,3)), T = np.zeros(3)):
    # -------------------------------------------------------------------------
    # Change the triangulation coordinate system.
    # If only one argument is provided Tr is moved to its principal inertia 
    # axis CS
    TrNewCS = {}
    # If new basis matrices and translation vector is not provided the PIA of
    # the shape are calculated to move it to it.
    if np.linalg.norm(V) == 0 and np.linalg.norm(T) == 0:
        V, T = TriInertiaPpties(Tr)
    elif np.linalg.norm(V) != 0 and np.linalg.norm(T) == 0:
        logging.exception('Wrong number of input argument, 1 move to PIA CS, 3 move to CS defined by V and T \n')
    
    # Translate the point by T
    Pts_T = Tr['Points'] - T.T
    
    # Rotate the points with V
    Pts_T_R = np.dot(Pts_T,V)
    
    # Construct the new triangulation
    TrNewCS = {'Points': Pts_T_R, 'ConnectivityList': Tr['ConnectivityList']}
    
    return TrNewCS, V, T

# -----------------------------------------------------------------------------
def TriReduceMesh(TR = {}, ElmtsKept = [], NodesKept = []):
    # -------------------------------------------------------------------------
    # Remove unnecessary Node and Renumber the elements accordingly 
    # 
    # Inputs:
    # TR: A triangulation dict with n Elements
    # ElmtsKept: A nx1 array of index of the rows of kept elments, or a
    # binary indicating kept Elements
    # NodesKept: ID of kept nodes of corresponding to TR connectibity list OR 
    # list of nodes coordinates
    TRout = {}
    NodesIDKept = []
    if not NodesKept:

        if np.sum(np.mod(NodesKept, 1)) == 0: # NodesID given
            NodesIDKept = NodesKept
        else: # Nodes Coordinates given
            NodesIDKept = [np.where(TR['Points'] == coord)[0][0] for coord in NodesKept]
    
    if not ElmtsKept:
        print('ENTRE!!!')
        tmp_ElmtsKept = []
        for nID in NodesIDKept:
            tmp_ElmtsKept += [pos for pos, val in enumerate(TR['ConnectivityList']) if nID in val]
        ElmtsKept = list(np.unique(tmp_ElmtsKept))
        
    PointsKept = TR['Points'][NodesKept]
    
        
    TRout['Points'] = TR['Points'][NodesKept]
    
    
    
    return TRout



#%% ---------------------------------------------------------------------------
# PlotFun
# -----------------------------------------------------------------------------
def plotDot(centers, ax, color = 'k', r = 1.75):
        
    # Create a sphere
    phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    
    # Plot the surface
    ax.plot_surface(x + centers[0], y + centers[1], z + centers[2], color = color)
    
    return ax

# -----------------------------------------------------------------------------
def quickPlotRefSystem(CS, ax, length_arrow = 60):
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    
    if 'V' in CS and not 'X' in CS:
        CS['X'] = CS['V'][:,0]
        CS['Y'] = CS['V'][:,1]
        CS['Z'] = CS['V'][:,2]
    
    if 'X' in CS and 'Origin' in CS:
        # plot X vector
        ax.quiver(CS['Origin'][0], CS['Origin'][1], CS['Origin'][2], \
                  CS['X'][0], CS['X'][1], CS['X'][2], \
                  color='r', length = length_arrow)
        # plot Y vector
        ax.quiver(CS['Origin'][0], CS['Origin'][1], CS['Origin'][2], \
                  CS['Y'][0], CS['Y'][1], CS['Y'][2], \
                  color='g', length = length_arrow)
        # plot Z vector
        ax.quiver(CS['Origin'][0], CS['Origin'][1], CS['Origin'][2], \
                  CS['Z'][0], CS['Z'][1], CS['Z'][2], \
                  color='b', length = length_arrow)
    else:
        logging.exception('plotting AXES X0-Y0-Z0 (ijk) \n')
        # plot X vector
        ax.quiver(CS['Origin'][0], CS['Origin'][1], CS['Origin'][2], \
                  1, 0, 0, \
                  color='r', length = length_arrow)
        # plot Y vector
        ax.quiver(CS['Origin'][0], CS['Origin'][1], CS['Origin'][2], \
                  0, 1, 0, \
                  color='g', length = length_arrow)
        # plot Z vector
        ax.quiver(CS['Origin'][0], CS['Origin'][1], CS['Origin'][2], \
                  0, 0, 1, \
                  color='b', length = length_arrow)
            
    plotDot(CS['Origin'], ax, 'k', 4*length_arrow/60)
    
    return ax





