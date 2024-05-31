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
from matplotlib.colors import LightSource
from sklearn import preprocessing

from Public_functions import freeBoundary

#%% ---------------------------------------------------------------------------
# SUBFUNTIONS
# -----------------------------------------------------------------------------
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
    if NodesKept  !=  []:
        if np.sum(np.mod(NodesKept, 1)) == 0: # NodesID given
            NodesIDKept = NodesKept
        else: # Nodes Coordinates given
            NodesIDKept = [np.where(TR['Points'] == coord)[0][0] for coord in NodesKept]
        
        NodesIDKept = np.sort(NodesIDKept)
        PointsKept = TR['Points'][NodesIDKept]
        
    if ElmtsKept  !=  []:
        if np.sum(np.mod(ElmtsKept, 1)) == 0: # ElmtsID given
            tmp_NodesIDKept = TR['ConnectivityList'][ElmtsKept]
            NodesIDKept = np.unique(tmp_NodesIDKept.ravel())
            
            ElmtsIDKept = ElmtsKept
        else: # Elements connectivity given
            tmp_ElmtsIDKept = [np.where(TR['ConnectivityList'] == tri)[0][0] for tri in ElmtsKept]
            ElmtsIDKept = np.unique(np.array(tmp_ElmtsIDKept))
        
        ElmtsKept = TR['ConnectivityList'][ElmtsIDKept]
        tmp_ElmtsKept = np.zeros(np.shape(ElmtsKept))
        for pos, val in enumerate(NodesIDKept):
            ind = np.where(ElmtsKept == val)
            tmp_ElmtsKept[ind] = pos
        
        PointsKept = TR['Points'][NodesIDKept]
        ElmtsKept = tmp_ElmtsKept
    else:
        tmp_ElmtsKept = []
        for tri in TR['ConnectivityList']:
            if set(tri).issubset(NodesIDKept):
                tmp_ElmtsKept.append(tri)
        
        ElmtsKept = np.array(tmp_ElmtsKept)
        tmp_ElmtsKept = np.zeros(np.shape(ElmtsKept))
        for pos, val in enumerate(NodesIDKept):
            ind = np.where(ElmtsKept == val)
            tmp_ElmtsKept[ind] = pos
        
    TRout['Points'] = PointsKept
    TRout['ConnectivityList'] = ElmtsKept.astype(int)
            
    return TRout

# -----------------------------------------------------------------------------
def TriFillPlanarHoles(Tr = {}):
    # -------------------------------------------------------------------------
    # Fill planar convex holes in the triangulation
    # For now the holes have to be planar
    # FOR NOW WORKS WITH ONLY ONE HOLE
    # 
    # Author: Emiliano P. Ravera (emiliano.ravera@uner.edu.ar)
    # -------------------------------
    Trout = {}
    FreeB = freeBoundary(Tr)
    
    if not FreeB:
        print('No holes on triangulation.')
        Trout = Tr
        return Trout

    # Fill the holes
    # center of the triangulation with hole
    TriCenter = np.mean(Tr['Points'], axis=0)
    TriCenter = np.reshape(TriCenter,(TriCenter.size, 1)) # convert d (3,) to 2d (3,1) vector

    # center onf the hole
    HoleCenter = np.mean(FreeB['Coord'], axis=0)
    HoleCenter = np.reshape(HoleCenter,(HoleCenter.size, 1)) # convert d (3,) to 2d (3,1) vector

    NewNode = np.max(Tr['ConnectivityList']) + 1

    U = preprocessing.normalize(HoleCenter-TriCenter, axis=0)

    ConnecL = []
    
    # create triangles from Free Points
    for IDpoint in FreeB['ID']:
        #  identify triangles that included a free point
        triangles = list(np.where(Tr['ConnectivityList'] == IDpoint))[0]

        for tri in triangles:
            # identify the triangle and line that include two free ponits
            points = [p for p in Tr['ConnectivityList'][tri] if p in FreeB['ID']]
            
            if len(points) == 2:

                ind_p1 = np.where(FreeB['ID'] == points[0])[0][0]
                ind_p2 = np.where(FreeB['ID'] == points[1])[0][0]
                
                p1 = FreeB['Coord'][ind_p1]
                p2 = FreeB['Coord'][ind_p2]
                
                Vctr1 = p1 - HoleCenter.T
                Vctr2 = p2 - HoleCenter.T
                
                normal = preprocessing.normalize(np.cross(Vctr1, Vctr2), axis=1)
                
                # Invert node ordering if the normals are inverted
                if np.dot(normal, U) < 0:
                    ConnecL.append(np.array([points[0], NewNode, points[1]]))
                else:
                    ConnecL.append(np.array([points[0], points[1], NewNode]))
          
    ConnecL = list(map(tuple, ConnecL))        
    ConnecL = list(np.array(ConnecL, dtype = 'int'))
                
    tmp_Points = list(Tr['Points'])
    tmp_Points.append(HoleCenter[:,0].T)
    NewPoints = np.array(tmp_Points)

    tmp_ConnectivityList = list(Tr['ConnectivityList'])
    tmp_ConnectivityList += ConnecL 

    Trout['Points'] = NewPoints
    Trout['ConnectivityList'] = np.array(tmp_ConnectivityList)
    
    # -----------------------------------
    # remove possible lonely triangles
    FreeB1 = freeBoundary(Trout)

    while len(FreeB1['ID']) > 0:

        tmp_ConnectList_out = list(Trout['ConnectivityList'])
        tmp_Points_out = list(Trout['Points'])
        list_delete = []
        
        for IDp in FreeB1['ID']:
            triangles = list(np.where(Trout['ConnectivityList'] == IDp))[0]
            if len(triangles) == 1:
                list_delete.append(triangles[0])
        
        list_delete.sort()
        list_delete = list_delete[::-1]
                
        tmp_ConnectList_out = [val for pos, val in enumerate(tmp_ConnectList_out) if pos not in list_delete]
        
        new_points = np.array(tmp_ConnectList_out)
        for id_tri in list_delete:
            new_points[new_points >= id_tri] -= 1
        
        tmp_Points_out = [val for pos, val in enumerate(tmp_Points_out) if pos not in list_delete]
        
        Trout['Points'] = np.array(tmp_Points_out)
        Trout['ConnectivityList'] = new_points
        
        # check the condition to finish while loop
        FreeB1 = freeBoundary(Trout)
    # -----------------------------------
    
    return Trout

# -----------------------------------------------------------------------------
def computeTriCoeffMorpho(Tr = {}):
    # -------------------------------------------------------------------------
    # Get the mean edge length of the triangles composing the femur
    # This is necessary because the functions were originally developed for
    # triangulation with constant mean edge lengths of 0.5 mm
    # -------------------------------------------------------------------------
    CoeffMorpho = 1
    PptiesTriObj = TriMesh2DProperties(Tr)
    
    # Assume triangles are equilaterals
    meanEdgeLength = np.sqrt( (4/np.sqrt(3))*(PptiesTriObj['TotalArea']/np.size(Tr['ConnectivityList'])) )
    
    # Get the coefficient for morphology operations
    CoeffMorpho = 0.5 / meanEdgeLength
    
    return CoeffMorpho

# -----------------------------------------------------------------------------
def TriDilateMesh(TRsup = {}, TRin = {}, nbElmts = 0):
    # -------------------------------------------------------------------------
    # This function is analog to a dilate function performed on binary images 
    # (https://en.wikipedia.org/wiki/Mathematical_morphology)
    # 
    # Input :
    # - TRsup : a support triangulation object (analog to the whole image)
    # - TRin : a triangulation object to be dilated (analog to the white pixels of the binary image)
    # TRin must me a subset (a region) of the TRsup triangulation, meaning that all vertices and elements of TRin
    # are included in TRsup even if they don't share the same numberings of vertices and elements.
    # - nbElemts : The number of neigbour elements/facets that will be dilated (analog to the number of pixel of the dilation)
    # 
    # Output :
    # - TRout : the dilated triangulation dict
    # -------------------------------------------------------------------------
    TRout = {}
    
    # Round the number of elements to upper integer
    nbElmts = int(np.ceil(nbElmts))
    
    # returns the rows of the intersection in the same order as they appear in 
    # the first vector given as input.
    ia = [i for i, v in enumerate(TRsup['Points']) for u in TRin['Points'] if np.all(v == u)]
    
    # Get the elements attached to the identified vertices
    ElmtsOK = []
    # Initially, ElmtsOk are the elements on TRsup that correspond to the geometry of the TRin
    for tri in ia:
       ElmtsOK += list(list(np.where(TRsup['ConnectivityList'] == tri))[0])
    # remove duplicated elements
    ElmtsOK = list(set(ElmtsOK))
    ElmtsInitial = TRsup['ConnectivityList'][ElmtsOK].reshape(-1, 1)

    # Get the neighbours of the identified elements, loop
    for dil in range(nbElmts):
        # Identify the neighbours of the elements of the ElmtsOK subset
        ElmtNeighbours = []
        for nei in ElmtsInitial:
            ElmtNeighbours += list(list(np.where(TRsup['ConnectivityList'] == nei))[0])
                    
        # remove duplicated elements
        ElmtNeighbours = list(set(ElmtNeighbours))
        ElmtsInitial = TRsup['ConnectivityList'][ElmtNeighbours].reshape(-1, 1)
        
        # Add the new neighbours to the list of elements ElmtsOK
        ElmtsOK += ElmtNeighbours
        # remove duplicated elements
        ElmtsOK = list(set(ElmtsOK))
    
    # The output is a subset of TRsup with ElmtsOK
    TRout = TriReduceMesh(TRsup, ElmtsOK)    
    
    return TRout

# -----------------------------------------------------------------------------
def TriUnite(Tr1 = {}, Tr2 = {}):
    # -------------------------------------------------------------------------
    # UNITE two unconnected triangulation dict, Tr1, Tr2
    # -------------------------------------------------------------------------
    Trout = {}
    # Points
    Tr1_Points = list(Tr1['Points'])
    Tr2_Points = list(Tr2['Points'])

    ind_SharedPoints = [p for p, v2 in enumerate(Tr2_Points) for v1 in Tr1_Points if np.linalg.norm(v1-v2) == 0]

    New_Points = Tr1_Points + [p for i, p in enumerate(Tr2_Points) if i not in ind_SharedPoints]
    # New_ConnectivityList = Tr1_ConnectivityList + [v for i, v in enumerate(Tr2_ConnectivityList) if i not in ind_SharedPoints]

    # Connectility List
    Tr1_ConnectivityList = list(Tr1['ConnectivityList'])
    Tr2_ConnectivityList = list(Tr2['ConnectivityList'])

    New_ConnectivityList = Tr1_ConnectivityList
    # update the indexes of triangles for Tr2
    for tri in Tr2_ConnectivityList:
        v0 = [pos for pos, val in enumerate(New_Points) if all(val == Tr2_Points[tri[0]])]
        v1 = [pos for pos, val in enumerate(New_Points) if all(val == Tr2_Points[tri[1]])]
        v2 = [pos for pos, val in enumerate(New_Points) if all(val == Tr2_Points[tri[2]])]
        
        New_ConnectivityList.append(np.array([v0[0], v1[0], v2[0]]))

    Trout['Points'] = np.array(New_Points)
    Trout['ConnectivityList'] = np.array(New_ConnectivityList)
    
    return Trout
# -----------------------------------------------------------------------------
# GeometricFun
# -----------------------------------------------------------------------------
def cutLongBoneMesh(TrLB, U_0 = np.reshape(np.array([0, 0, 1]),(3, 1)), L_ratio = 0.33):
    # Separate the Mesh of long bone in two parts: a proximal and a distal one.
    # 
    # inputs :
    # TrLB - The triangulation of a long bone
    # U_0 - A unit vector defining the wanted distal to proximal 
    # orientation of the principal direction
    # L_ratio - The ratio of the bone length kept to define the prox. 
    # and distal part of the long bone.
    # 
    # Outputs:
    # TrProx - Triangulation dict of the proximal part of the long bone
    # TrDist - Triangulation of the distal part of the long bone
    # -------------------------------------------------------------------------
    TrProx = {}
    TrDist = {}
    
    if np.all(U_0 == np.array([0, 0, 1])):
        print('Distal to proximal direction of long bone is based on the' + \
        ' assumption that the bone distal to proximal axis is oriented' + \
        ' +Z_CT or +Z_MRI vector of the imaging system. If it`s not' + \
        ' the case the results might be wrong.')
    
    # Convert tiangulation dict to mesh object --------
    tmp_TrLB = mesh.Mesh(np.zeros(TrLB['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(TrLB['ConnectivityList']):
        for j in range(3):
            tmp_TrLB.vectors[i][j] = TrLB['Points'][f[j],:]
    # update normals
    tmp_TrLB.update_normals()
    # ------------------------------------------------
    
    V_all, _, _, _, _ = TriInertiaPpties(TrLB)

    # Initial estimate of the Distal-to-Proximal (DP) axis Z0
    Z0 = V_all[0]
    Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector

    # Reorient Z0 according to U_0
    Z0 *= np.sign(np.dot(U_0.T, Z0))

    # Fast and dirty way to split the bone
    LengthBone = np.max(np.dot(TrLB['Points'], Z0)) - np.min(np.dot(TrLB['Points'], Z0))
    
    # create the proximal bone part
    Zprox = np.max(np.dot(TrLB['Points'], Z0)) - L_ratio*LengthBone
    ElmtsProx = np.where(np.dot(tmp_TrLB.centroids, Z0) > Zprox)[0]
    TrProx = TriReduceMesh(TrLB, ElmtsProx)
    TrProx = TriFillPlanarHoles(TrProx)
    
    # create the distal bone part
    Zdist = np.min(np.dot(TrLB['Points'], Z0)) + L_ratio*LengthBone
    ElmtsDist = np.where(np.dot(tmp_TrLB.centroids, Z0) < Zdist)[0]
    TrDist = TriReduceMesh(TrLB, ElmtsDist)
    TrDist = TriFillPlanarHoles( TrDist)
    
    return TrProx, TrDist
    



#%% ---------------------------------------------------------------------------
# PlotFun
# -----------------------------------------------------------------------------
def plotDot(centers, ax, color = 'k', r = 1.75):
    # -------------------------------------------------------------------------
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
    # -------------------------------------------------------------------------
        
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

# -----------------------------------------------------------------------------
def plotTriangLight(Triang = {}, CS = {}, ax = None, alpha = 0.7):
    # -------------------------------------------------------------------------
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
           
    # define the lighting
    if CS:
        # if there are no axes but there is a pose matrix, use the matrix as 
        # reference
        if not 'Y' in CS and 'V' in CS:
            CS['X'] = CS['V'][0]
            CS['Y'] = CS['V'][1]
            CS['Z'] = CS['V'][2]
        
        # handle lighting of objects
        # Create light source object.
        angle = np.arccos(np.dot(CS['X'], CS['Z']))
        ls = LightSource(azdeg=0, altdeg=angle)
        # Shade data, creating an rgb array.
        # rgb = ls.shade(Triang['Points'][:,2], plt.cm.RdYlBu)
        shade = True
    else:
        shade = False

    # Plot the triangulation object with grey color
    ax.plot_trisurf(Triang['Points'][:,0], Triang['Points'][:,1], Triang['Points'][:,2], \
                     triangles = Triang['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=alpha, color = 'grey', shade=shade)    
    
    # Remove grid
    ax.grid(False)
    
    return ax

# -----------------------------------------------------------------------------
def plotBoneLandmarks(BLDict, ax, label_switch = 1):
    # -------------------------------------------------------------------------
    # Add point plots, and if required, labels to bone landmarks identified 
    # throught the STAPLE analyses.
    # 
    # Inputs:
    # BLDict - a Python dictionary with fields having as fields the name of 
    # the bone landmarks and as values their coordinates (in global
    # reference system).
    # 
    # label_switch - a binary switch that indicates if the bone landmark
    # names will be added or not (as text) to the plot.
    # 
    # Outputs:
    # none - the points are plotted on the current axes.
    # -------------------------------------------------------------------------
    
    for curr_name in BLDict:
        plotDot(BLDict[curr_name], ax, 'r', 7)
        if label_switch:
            ax.text(BLDict[curr_name][0][0],BLDict[curr_name][1][0],BLDict[curr_name][2][0], \
                    curr_name, size=12, color='k')
    
    return ax


