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
import scipy.spatial as spatial
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import path as mpl_path
from sklearn import preprocessing
from scipy.optimize import curve_fit
from scipy.optimize import nnls
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R
from ellipse import LsqEllipse

from Public_functions import freeBoundary, \
                              PolyArea, \
                               inpolygon

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
    if ElmtsKept == [] and NodesKept == []:
        TRout = TR
        return TRout
    
    TRout = {}
    NodesIDKept = []
    if NodesKept  !=  []:
        if np.sum(np.mod(NodesKept, 1)) == 0: # NodesID given
            NodesIDKept = NodesKept
        else: # Nodes Coordinates given
            # NodesIDKept = [np.where(TR['Points'] == coord)[0][0] for coord in NodesKept]
            NodesIDKept = []
            for coord in NodesKept:
                tmp1 = list(np.where(TR['Points'] == coord)[0])
                if tmp1 != []:
                    NodesIDKept += list(set(tmp1))
                    tmp1 = []
        
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
        for pos, val in enumerate(NodesIDKept[::-1]):
            ind = np.where(ElmtsKept == val)
            tmp_ElmtsKept[ind] = pos
        ElmtsKept = tmp_ElmtsKept
        
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
    # meanEdgeLength = np.sqrt( (4/np.sqrt(3))*(PptiesTriObj['TotalArea']/np.size(Tr['ConnectivityList'])) )
    meanEdgeLength = np.sqrt( (4/np.sqrt(3))*(PptiesTriObj['TotalArea']/len(Tr['ConnectivityList'])) )
    
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
def TriKeepLargestPatch(Tr = {}):
    # -------------------------------------------------------------------------
    # Keep the largest (by area) connected patch of a triangulation dict
    # 
    # Inputs:
    # TR: A triangulation dict with n Elements
    # 
    # author: Emiliano P. Ravera (emiliano.ravera@uner.edu.ar)
    # -------------------------------------------------------------------------
    TRout = {}
    
    Border = freeBoundary(Tr)
        
    border = list(set(list(Border['ID'])))

    Patch = {}
    patch = []
    i = 1
    j = 0

    if len(border) == 3:
        
        Patch[str(i)] = border
        
    else:
        p = border[0]
        border.remove(p)
        patch += [p]
        
        while border:
                
            t = np.where(Tr['ConnectivityList'] == p)[0]
                
            new_vertexs = list(set(list(Tr['ConnectivityList'][t].reshape(-1))))
            new_vertexs.remove(p)
            new_vertexs = np.array(new_vertexs)
            
            ind_nv = np.array([True if (val in border and val not in patch) else False for val in new_vertexs])
            
            if any(ind_nv):
                new_vert_in_bord = list(new_vertexs[ind_nv])
                patch += new_vert_in_bord
                
                # remove vertexs from border list
                border = [v for v in border if v not in patch]
                
            elif j < len(patch):
                p = patch[j]
                j += 1
                        
            elif j == len(patch):
                Patch[str(i)] = patch
                patch = []
                p = border[0]
                patch += [p]
                j = 0
                i += 1
            
    Area_patch = {}
    for key in Patch:
        Area_patch[key] = 0
        indexes = []
        # identify the traingles in each patch
        for p in Patch[key]:
            indexes += list(np.where(Tr['ConnectivityList'] == p)[0])
        indexes = list(set(indexes))
        
        # compute the area for each patch
        for tri in Tr['ConnectivityList'][indexes]:
            v1 = Tr['Points'][tri[1]] - Tr['Points'][tri[0]]
            v2 = Tr['Points'][tri[2]] - Tr['Points'][tri[0]]
            
            Area_patch[key] += 0.5*(np.linalg.norm(np.cross(v1, v2)))

    # patch with greater area
    PwGA = [k for k, v in Area_patch.items() if v == max(Area_patch.values())][0]

    indexes = []
    # identify the traingles in each patch
    for p in Patch[PwGA]:
        indexes += list(np.where(Tr['ConnectivityList'] == p)[0])
    indexes = list(set(indexes))
        
    # Buid triangulation output
    ind_points = list(set(list(Tr['ConnectivityList'][indexes].reshape([-1,1])[:,0])))
    TRout['Points'] = Tr['Points'][ind_points]

    TRout['ConnectivityList'] = np.zeros(np.shape(Tr['ConnectivityList'][indexes]))

    for pos, tri in  enumerate(Tr['ConnectivityList'][indexes]):
        
        ind0 = np.where(np.linalg.norm(TRout['Points']-Tr['Points'][tri[0]], axis=1) == 0)[0][0]
        ind1 = np.where(np.linalg.norm(TRout['Points']-Tr['Points'][tri[1]], axis=1) == 0)[0][0]
        ind2 = np.where(np.linalg.norm(TRout['Points']-Tr['Points'][tri[2]], axis=1) == 0)[0][0]
        
        TRout['ConnectivityList'][pos] = np.array([ind0, ind1, ind2])
        
    TRout['ConnectivityList'] = TRout['ConnectivityList'].astype(np.int64)
    
    return TRout

# -----------------------------------------------------------------------------
def TriErodeMesh(Trin = {}, nbElmts = 0):
    # -------------------------------------------------------------------------
    # 
    # 
    # -------------------------------------------------------------------------
    TRout = {}
    
    nbElmts = np.ceil(nbElmts)
    
    Free = freeBoundary(Trin)    

    ElmtsBorder = []
    for pID in Free['ID']:
        ElmtsBorder += list(np.where(Trin['ConnectivityList'] == pID)[0])

    # remove duplicated elements
    ElmtsBorder = list(set(ElmtsBorder))
    ElmtsInitial = ElmtsBorder

    if nbElmts > 1:
        # Get the neighbours of the identified elements, loop
        for nb in range(int(nbElmts-1)):
            # Identify the neighbours of the elements of the ElmtsOK subset
            ElmtNeighbours = []
            for nei in ElmtsInitial:
                ElmtNeighbours += list(list(np.where(Trin['ConnectivityList'] == nei))[0])
                        
            # remove duplicated elements
            ElmtNeighbours = list(set(ElmtNeighbours))
            ElmtsInitial += ElmtNeighbours

        tmp_kept = list(range(len(Trin['ConnectivityList'])))
        ElemtsKept = list(set(tmp_kept) - set(ElmtsInitial))
        if ElemtsKept == []:
            ElemtsKept = ElmtsInitial
    else:
        ElemtsKept = ElmtsInitial
    
    # The output is a subset of TRsup with ElmtsKept
    TRout = TriReduceMesh(Trin, ElemtsKept)

    return TRout

# -----------------------------------------------------------------------------
def TriOpenMesh(TRsup = {}, TRin = {}, nbElmts=0):
    # -------------------------------------------------------------------------
    # 
    # 
    # -------------------------------------------------------------------------
    TR = TriErodeMesh(TRin, nbElmts)
    TRout = TriDilateMesh(TRsup, TR, nbElmts)
   
    return TRout

# -----------------------------------------------------------------------------
def TriPlanIntersect(Tr = {}, n = np.zeros((3,1)), d = np.zeros((3,1)), debug_plots = 0):
    # -------------------------------------------------------------------------
    # Intersection between a 3D Triangulation object (Tr) and a 3D plan defined 
    # by normal vector n , d
    # 
    # Outputs:
    # Curves: a structure containing the diffirent intersection profile
    # if there is only one Curves(1).Pts gives the intersection
    # curve ordered points vectors (forming a polygon)
    # TotArea: Total area of the cross section accounting for holes
    # InterfaceTri: sparse Matrix with n1 x n2 dimension where n1 and n2 are
    # number of faces in surfaces
    # 
    # -------------------------------------------------------------------------
    # ONLY TESTED : on closed triangulation resulting in close intersection
    # curve
    # -------------------------------------------------------------------------
    
    if np.linalg.norm(n) == 0 and np.linalg.norm(d) == 0:
        logging.error('Not engough input argument for TriPlanIntersect')
        # print('Not engough input argument for TriPlanIntersect')

    Pts = Tr['Points']
    n = preprocessing.normalize(n, axis=0)

    # If d is a point on the plane and not the d parameter of the plane equation        
    if type(d) is np.ndarray and len(d) > 2:
        Op = d
        row, col = d.shape
        if col == 1:
            d = np.dot(-d,n)
        elif row == 1:
            d = np.dot(-d.T,n)
        else:
            logging.error('Third input must be an altitude or a point on plane')
            # print('Third input must be an altitude or a point on plane')
    else:
        # Get a point on the plane
        n_principal_dir = np.argmax(abs(n))
        Pts1 = Pts[0].copy()
        Pts1 = np.reshape(Pts1,(Pts1.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        Op = Pts1.copy()
        Pts1[n_principal_dir] = 0
        Op[n_principal_dir] = (np.dot(-Pts1.T, n) - d)/n[n_principal_dir]

    ## Find the intersected elements (triagles)
    # Get Points (vertices) list as being over or under the plan

    Pts_Over = [1 if np.dot(p.T, n) + d > 0 else 0 for p in Pts]
    Pts_Under = [1 if np.dot(p.T, n) + d < 0 else 0 for p in Pts]
    Pts_OverUnder = np.array(Pts_Over) - np.array(Pts_Under)

    if np.sum(Pts_OverUnder == 0) > 0:
        logging.warning('Points were found lying exactly on the intersecting plan, this case might not be correctly handled')
        # print('Points were found lying exactly on the intersecting plan, this case might not be correctly handled')

    # Get the facets,elements/triangles/ intersecting the plan
    Elmts_Intersecting = []
    Elmts = Tr['ConnectivityList'] 
    Elmts_IntersectScore = np.sum(Pts_OverUnder[Elmts],axis=1)
    Elmts_Intersecting =  Elmts[np.abs(Elmts_IntersectScore)<3]

    # Check the existence of an interaction
    if len(Elmts_Intersecting) == 0:
        TotArea = 0
        InterfaceTri = []
        Curves = {}
        Curves['1'] = {}
        Curves['1']['NodesID'] = []
        Curves['1']['Pts'] = []
        Curves['1']['Area'] = 0
        Curves['1']['Hole'] = 0
        Curves['1']['Text'] = 'No Intersection'
        logging.warning('No intersection found between the plane and the triangulation')
        # print('No intersection found between the plane and the triangulation')
        return Curves, TotArea, InterfaceTri

    # Find the Intersecting Edges among the intersected elements
    # Get an edge list from intersecting elmts
    Nb_InterSectElmts = len(Elmts_Intersecting)
    Edges = np.zeros((3*Nb_InterSectElmts, 2))

    i = np.array(range(1,Nb_InterSectElmts+1))
    Edges[3*i-3] = Elmts_Intersecting[i-1,:2]
    Edges[3*i-2] = Elmts_Intersecting[i-1,1:3]
    Edges[3*i-1,0] = Elmts_Intersecting[i-1,-1]
    Edges[3*i-1,1] = Elmts_Intersecting[i-1,0]
    Edges = Edges.astype(np.int64)

    # Identify the edges crossing the plane
    # They will have an edge status of 0
    Edges_Status = np.sum(Pts_OverUnder[Edges],axis=1)

    I_Edges_Intersecting = np.where(Edges_Status == 0)[0]
    # Find the edge plane intersecting points
    # start and end points of each edges

    P0 = Pts[Edges[I_Edges_Intersecting,0]]
    P1 = Pts[Edges[I_Edges_Intersecting,1]]

    # Vector of the edge
    u = P1 - P0

    # Get vectors from point on plane (Op) to edge ends
    v = P0 - Op.T

    EdgesLength = np.dot(u,n)
    EdgesUnderPlaneLength = np.dot(-v,n)

    ratio = EdgesUnderPlaneLength/EdgesLength

    # Get Intersectiong Points Coordinates
    PtsInter = P0 + u*ratio

    # Make sure the shared edges have the same intersection Points
    # Build an edge correspondance table

    EdgeCorrespondence = np.empty((3*Nb_InterSectElmts, 1))
    EdgeCorrespondence[:] = np.nan

    for pos, edge in enumerate(Edges):
        
        tmp = np.where((Edges[:,1] == edge[0]) & (Edges[:,0] == edge[1]))
        
        if tmp[0].size == 0:
            continue
        elif tmp[0].size == 1:
            i = tmp[0][0]
            if np.isnan(EdgeCorrespondence[pos]):
                EdgeCorrespondence[pos] = pos
                EdgeCorrespondence[i] = pos
        else:
            print('Intersecting edge appear in 3 triangles, not good')

    # Get edge intersection point
    # Edge_IntersectionPtsIndex = np.zeros((3*Nb_InterSectElmts, 1))
    Edge_IntersectionPtsIndex = np.empty((3*Nb_InterSectElmts, 1))
    Edge_IntersectionPtsIndex[:] = np.nan
    tmp_edgeInt = np.array(range(len(I_Edges_Intersecting)))
    tmp_edgeInt = np.reshape(tmp_edgeInt,(tmp_edgeInt.size, 1)) # convert 1d (#,) to 2d (#,1) vector
    Edge_IntersectionPtsIndex[I_Edges_Intersecting] = tmp_edgeInt

    # Don't use intersection point duplicate: only one intersection point per edge
    ind_interP = EdgeCorrespondence[I_Edges_Intersecting]
    ind_interP = list(ind_interP)
    I_Edges_Intersecting = list(I_Edges_Intersecting)
    t = np.sort(np.where(np.isnan(ind_interP))[0])
    t = t[::-1]
    for ind in t:
        del ind_interP[ind]
        del I_Edges_Intersecting[ind]

    ind_interP = np.array(ind_interP).astype(np.int64)
    I_Edges_Intersecting = np.array(I_Edges_Intersecting).astype(np.int64)

    # Edge_IntersectionPtsIndex[I_Edges_Intersecting] = Edge_IntersectionPtsIndex[EdgeCorrespondence[I_Edges_Intersecting].astype(np.int64)][:,0]
    Edge_IntersectionPtsIndex[I_Edges_Intersecting] = Edge_IntersectionPtsIndex[ind_interP][:,0]

    # Get the segments intersecting each triangle
    # The segments are: [Intersecting Point 1 ID , Intersecting Point 2 ID]
    # Segments = Edge_IntersectionPtsIndex[Edge_IntersectionPtsIndex > 0].astype(np.int64)
    Segments = Edge_IntersectionPtsIndex[Edge_IntersectionPtsIndex != np.nan]
    Segments = Segments[~np.isnan(Segments)].astype(np.int64)
    Segments = list(Segments.reshape((-1,2)))

    # Separate the edges to curves structure containing close curves
    Curves = {}
    i = 1
    while Segments:
        # Initialise the Curves Structure, if there are multiple curves this
        # will lead to trailing zeros that will be removed afterwards
        Curves[str(i)] = {}
        Curves[str(i)]['NodesID'] = []
        Curves[str(i)]['NodesID'].append(Segments[0][0])
        Curves[str(i)]['NodesID'].append(Segments[0][1])
        
        # Remove the semgents added to Curves[i] from the segments list
        del Segments[0]
                
        # Find the edge in segments that has a node already in the Curves[i][NodesID]
        # This edge will be the next edge of the current curve because it's
        # connected to the current segment
        # Is, the index of the next edge
        # Js, the index of the node within this edge already present in NodesID
        
        tmp1 = np.where(Segments == Curves[str(i)]['NodesID'][-1])
        if tmp1[0].size == 0:
            break
        else:
            Is = tmp1[0][0]
            Js = tmp1[1][0]
            
        # Nk is the node of the previuously found edge that is not in the
        # current Curves[i][NodesID] list
        # round(Js+2*(0.5-Js)) give 0 if Js = 1 and 1 if Js = 0
        # It gives the other node not yet in NodesID of the identified next edge
        Nk = Segments[Is][int(np.round(Js+2*(0.5-Js)))]
        del Segments[Is]
            
        # Loop until there is no next node
        while Nk:
            Curves[str(i)]['NodesID'].append(Nk)
            if Segments:
                tmp2 = np.where(Segments == Curves[str(i)]['NodesID'][-1])
                if tmp2[0].size == 0:
                    break
                else:
                    Is = tmp2[0][0]
                    Js = tmp2[1][0]
                           
                Nk = Segments[Is][int(np.round(Js+2*(0.5-Js)))]
                del Segments[Is]
            else:
                break
            
        # If there is on next node then we move to the next curve
        i += 1

    # Compute the area of the cross section defined by the curve

    # Deal with cases where a cross section presents holes
    # 
    # Get a matrix of curves inclusion -> CurvesInOut :
    # If the curve(i) is within the curve(j) then CurvesInOut(i,j) = 1
    # else  CurvesInOut(i,j) = 0

    for key in Curves.keys():
        Curves[key]['Pts'] = []
        Curves[key]['Pts'] = PtsInter[Curves[key]['NodesID']]
        
        # Replace the close curve in coordinate system where X, Y or Z is 0
        _, V = np.linalg.eig(np.cov(Curves[key]['Pts'].T))
        
        CloseCurveinRplanar1 = np.dot(V.T, Curves[key]['Pts'].T)
        
        # Get the area of the section defined by the curve 'key'.
        # /!\ the curve.Area value Do not account for the area of potential 
        # holes in the section described by curve 'key'.
        # Curves[key]['Area'] = PolyArea(CloseCurveinRplanar1[0,:],CloseCurveinRplanar1[1,:])
        Curves[key]['Area'] = PolyArea(CloseCurveinRplanar1)
        
        CurvesInOut = np.zeros((len(Curves),len(Curves)))
        
        for key1 in Curves.keys():
            if key1 != key:
                Curves[key1]['Pts'] = []
                Curves[key1]['Pts'] = PtsInter[Curves[key1]['NodesID']]
                
                # Replace the close curve in coordinate system where X, Y or Z is 0
                _, V = np.linalg.eig(np.cov(Curves[key1]['Pts'].T))
                
                CloseCurveinRplanar2 = np.dot(V.T, Curves[key1]['Pts'].T)
                
                # Check if the Curves[key] is within the Curves[key1]
                path1 = CloseCurveinRplanar1[0:3:2,:].T
                path2 = CloseCurveinRplanar2[0:3:2,:].T
                
                p = mpl_path.Path(path1.real)
                if any(p.contains_points(path2.real)):
                    
                    CurvesInOut[int(key)-1, int(key1)-1] = 1

    # if the Curves[key] is within an even number of curves then its area must
    # be added to the total area. If the Curves[key] is within an odd number
    # of curves then its area must be substracted from the total area
    TotArea = 0

    for key in Curves.keys():
        AddOrSubstract = 1 - 2*np.remainder(np.sum(CurvesInOut[int(key)-1]), 2)
        Curves[key]['Hole'] = -AddOrSubstract # 1 if hole -1 if filled
        TotArea -= Curves[key]['Hole']*Curves[key]['Area']
              
    I_X = np.where(np.abs(Elmts_IntersectScore)<3)[0]
    InterfaceTri = TriReduceMesh(Tr, I_X)

    if debug_plots:
        
        V_all, CenterVol, _, _, _ = TriInertiaPpties(Tr)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        
        ax.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  V_all[0,0], V_all[1,0], V_all[2,0], \
                  color='r', length = 250)
        ax.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  V_all[0,1], V_all[1,1], V_all[2,1], \
                  color='g', length = 100)
        ax.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  V_all[0,2], V_all[1,2], V_all[2,2], \
                  color='b', length = 175)
            
        ax.plot_trisurf(Tr['Points'][:,0], Tr['Points'][:,1], Tr['Points'][:,2], triangles = Tr['ConnectivityList'], \
                        edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'blue')
        ax.plot_trisurf(InterfaceTri['Points'][:,0], InterfaceTri['Points'][:,1], InterfaceTri['Points'][:,2], triangles = InterfaceTri['ConnectivityList'], \
                        edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.7, shade=False, color = 'red')
            
        for key in Curves.keys():
            
            plt.plot(Curves[key]['Pts'][:,0], Curves[key]['Pts'][:,1], Curves[key]['Pts'][:,2], 'k-', linewidth=4)
    
    return Curves, TotArea, InterfaceTri 
    
# -----------------------------------------------------------------------------
def TriSliceObjAlongAxis(TriObj, Axis, step, cut_offset = 0.5, debug_plot = 0):
    # -------------------------------------------------------------------------
    # Slice a Dict triangulation object TriObj along a specified axis. 
    # Notation and inputs are consistent with the other GIBOC-Knee functions 
    # used to manipulate triangulations.
    # -------------------------------------------------------------------------
    min_coord = np.min(np.dot(TriObj['Points'], Axis)) + cut_offset
    max_coord = np.max(np.dot(TriObj['Points'], Axis)) - cut_offset
    Alt = np.arange(min_coord, max_coord, step)

    Curves = {}
    Areas = {}

    for it, d in enumerate(-Alt):

        Curves[str(it)], Areas[str(it)], _ = TriPlanIntersect(TriObj, Axis, d)
        
    if debug_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for key in Curves.keys():
            if Curves[key]:
                ax.plot(Curves[key]['1']['Pts'][:,0], Curves[key]['1']['Pts'][:,1], Curves[key]['1']['Pts'][:,2], 'k-', linewidth=2)
        plt.show()

    print('Sliced #' + str(len(Curves)-1) + ' times')
    maxArea = Areas[max(Areas, key=Areas.get)]
    maxAreaInd = int(max(Areas, key=Areas.get))
    maxAlt = Alt[maxAreaInd]
    
    return Areas, Alt, maxArea, maxAreaInd, maxAlt

# -----------------------------------------------------------------------------
def TriVertexNormal(Tri = {}):
    # -------------------------------------------------------------------------
    # Compute the vertex normal of triangulation dict
    # -------------------------------------------------------------------------
    Tri_out = Tri.copy()
    # Convert tiangulation dict to mesh object ------------------------------------
    tmp_Tri = mesh.Mesh(np.zeros(Tri_out['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(Tri_out['ConnectivityList']):
        for j in range(3):
            tmp_Tri.vectors[i][j] = Tri_out['Points'][f[j],:]
    # update normals
    tmp_Tri.update_normals()
    # -----------------------------------------------------------------------------
    
    # compute vertex normal
    Tri_out['vertexNormal'] = np.zeros(np.shape(Tri_out['Points']))
    UnitNormals = tmp_Tri.get_unit_normals()
    for pos, point in enumerate(Tri_out['Points']):
        triangles = np.where(Tri_out['ConnectivityList'] == pos)[0]
        tmp = np.sum(UnitNormals[triangles,:], axis = 0)
        Tri_out['vertexNormal'][pos] = tmp/ np.sqrt(tmp[0]**2 + tmp[1]**2 + tmp[2]**2)
        
    return Tri_out

# -----------------------------------------------------------------------------
def TriCurvature(TR = {}, usethird = False, Rot0 = np.identity(3)):
    # -------------------------------------------------------------------------
    # Calculate the principal curvature directions and values
    # of a triangulated mesh. 
    # 
    # The function first rotates the data so the normal of the current
    # vertex becomes [-1 0 0], so we can describe the data by XY instead of XYZ.
    # Secondly it fits a least-squares quadratic patch to the local 
    # neighborhood of a vertex "f(x,y) = ax^2 + by^2 + cxy + dx + ey + f". 
    # Then the eigenvectors and eigenvalues of the hessian are used to
    # calculate the principal, mean and gaussian curvature.
    # 
    # [Cmean,Cgaussian,Dir1,Dir2,Lambda1,Lambda2]=patchcurvature(FV,usethird)
    # 
    # inputs,
    # TR : A dict triangulation mesh object (see Patch)
    # usethird : Use third order neighbour vertices for the curvature fit, 
    # making it smoother but less local. true/ false (default)
    # 
    # outputs,
    # Cmean : Mean Curvature
    # Cgaussian : Gaussian Curvature
    # Dir1 : XYZ Direction of first Principal component
    # Dir2 : XYZ Direction of second Principal component
    # Lambda1 : value of first Principal component
    # Lambda2 : value of second Principal component
    # 
    # % ---------------------------%
    # Function is written by D.Kroon University of Twente (August 2011)  
    # Last Update, 15-1-2014 D.Kroon at Focal.
    # % ---------------------------%
    # Slightly modified for triangulation inputs
    # Last modification, 15-11-2017; JB Renault at AMU.
    # % ---------------------------%
    # Python version is written by EP Ravera
    # Last modification, 30-07-2024; EP Ravera at CONICET
    # % ---------------------------%
    # -------------------------------------------------------------------------
    
    # Change Triangulation position for conditioning:
    if TR:
        TR['Points'] = np.dot(TR['Points'], Rot0)
    
    # Calculate vertices normals
    TR = TriVertexNormal(TR)
    N = TR['vertexNormal']
    
    # Calculate Rotation matrices for the normals
    M = []
    Minv = []
    for vect in N:
        r = R.from_rotvec(vect)
        M.append(r.as_matrix())
        Minv.append(np.linalg.inv(r.as_matrix()))
    
    # Get neighbours of all vertices
    Ne = []
    for vertex in range(len(TR['Points'])):
        nei = np.where(TR['ConnectivityList'] == vertex)[0]
        Ne.append(nei)
    
    Lambda1 = []
    Lambda2 = []
    Dir1 = []
    Dir2 = []
    for i, nei in enumerate(Ne):
        Nce = []
        SRNei = []
        TRNei = []
        # First Ring Neighbours
        FRNei = list(nei)
        
        if usethird == False:
            # Get first and second ring neighbours
            for neiID in nei:
                # Second Ring Neighbours
                sr = np.where(TR['ConnectivityList'] == neiID)[0]
                if sr != []:
                    SRNei.append(sr[0])
        else:
            # Get first, second and third ring neighbours
            for neiID in nei:
                # Second Ring Neighbours
                SRNei.append(np.where(TR['ConnectivityList'] == neiID)[0][0])
            for neiID in SRNei:
                # Second Ring Neighbours
                TRNei.append(np.where(TR['ConnectivityList'] == neiID)[0][0])
        
        Nce = list(set(FRNei + SRNei + TRNei))
        Ve = TR['Points'][Nce]
        
        # Rotate to make normal [-1 0 0]
        We = np.dot(Ve, Minv[i])
        
        f = We[:,0]
        x = We[:,1]
        y = We[:,2]
        
        # Fit patch
        # f(x,y) = ax^2 + by^2 + cxy + dx + ey + f
        FM = np.array([x**2, y**2, x*y, x, y, np.ones(len(x))]).T
        abcdef = np.linalg.lstsq(FM, f, rcond=None)[0]
        a = abcdef[0]
        b = abcdef[1]
        c = abcdef[2]
        
        Dxx = 2*a
        Dxy = c 
        Dyy = 2*b
                
        hessian = np.array([[Dxx, Dxy],[Dxy,Dyy]])
        eigenvalues, eigenvectors = np.linalg.eig(hessian)
                
        if np.abs(eigenvalues[1]) < np.abs(eigenvalues[0]):
            Lambda1.append(eigenvalues[1])
            Lambda2.append(eigenvalues[0])
            I1 = eigenvectors[1]
            I2 = eigenvectors[0]
        else:
            Lambda1.append(eigenvalues[0])
            Lambda2.append(eigenvalues[1])
            I1 = eigenvectors[0]
            I2 = eigenvectors[1]
        
        dir1 = np.dot(np.array([0, I1[0], I1[1]]), M[i])
        dir2 = np.dot(np.array([0, I2[0], I2[1]]), M[i])
                
        tmp1 = dir1/(np.sqrt(dir1[0]**2 + dir1[1]**2 + dir1[2]**2))
        tmp2 = dir2/(np.sqrt(dir2[0]**2 + dir2[1]**2 + dir2[2]**2))
        Dir1.append(tmp1)
        Dir2.append(tmp2)
        
    Cmean = (np.array(Lambda1) + np.array(Lambda2))/2
    Cgaussian = np.array(Lambda1)*np.array(Lambda2)
    
    return Cmean, Cgaussian, Dir1, Dir2, Lambda1, Lambda2
    
# -----------------------------------------------------------------------------
def TriConnectedPatch(TR, PtsInitial):
    # -------------------------------------------------------------------------
    # Find the connected mesh (elements sharing at least an edge) starting 
    # from a given point (not necessarily lying on the mesh)
    # -------------------------------------------------------------------------
    # nearestNeighbor
    # Get the index of points
    point_tree = spatial.cKDTree(TR['Points'])
    # This finds the index of all points within the kd-tree for nearest neighbors.
    NodeInitial = point_tree.query(PtsInitial, k=1)[1]
    NodeInitial = np.unique(NodeInitial)
    
    ElmtsInitial = np.unique(TR['ConnectivityList'][NodeInitial].reshape(-1, 1))
    ElmtsConnected = list(ElmtsInitial)
    
    test = False
    while test == False:
        PreviousLength = len(ElmtsConnected)
        # Identify the neighbours of the elements of the ElmtsOK subset
        ElmtNeighbours = []
        for nei in ElmtsInitial:
            ElmtNeighbours += list(list(np.where(TR['ConnectivityList'] == nei))[0])
                    
        # remove duplicated elements
        ElmtNeighbours = list(set(ElmtNeighbours))
        ElmtsInitial = TR['ConnectivityList'][ElmtNeighbours].reshape(-1, 1)
    
        # Add the new neighbours to the list of elements ElmtsOK
        ElmtsConnected += list(ElmtsInitial[:,0])
        # remove duplicated elements
        ElmtsConnected = list(set(ElmtsConnected))
        
        if len(ElmtsConnected) == PreviousLength:
            test = True
    
    TRout = TriReduceMesh(TR, ElmtsConnected) 
    return TRout

# -----------------------------------------------------------------------------
def TriCloseMesh(TRsup ,TRin , nbElmts):
    # -------------------------------------------------------------------------
    # 
    # -------------------------------------------------------------------------
    TR = TriDilateMesh(TRsup, TRin, nbElmts)
    TRout = TriErodeMesh(TR, nbElmts)
    
    return TRout
    
# -----------------------------------------------------------------------------
def TriDifferenceMesh(TR1, TR2):
    # -------------------------------------------------------------------------
    # Boolean difference between original mesh TR1 and another mesh TR2
    # /!\ delete all elements in TR1 that contains a node in TR2
    # -------------------------------------------------------------------------
    TRout = {}
    # returns the rows of the intersection in the same order as they appear in
    ia = [i for i, v in enumerate(TR1['Points']) for u in TR2['Points'] if np.all(v == u)]
    
    if ia == []:
        print('No intersection found, the tolerance distance has been set to 1E-5')
        ia = [i for i, v in enumerate(TR1['Points']) for u in TR2['Points'] if np.linalg.norm(v-u) <= 1e-5]
    
    if ia != []:
        Elmts2Delete = []
        for vertex in ia:
            nei = np.where(TR1['ConnectivityList'] == vertex)[0]
            Elmts2Delete.append(nei)
        
        # remove duplicated elements
        Elmts2Delete = list(set(Elmts2Delete))
        
        Elmts2Keep = np.ones(len(TR1['ConnecticityList']), dtype='bool')
        Elmts2Keep[Elmts2Delete] = False
        Elmts2KeepID = np.where(Elmts2Keep)[0]
        
        TRout = TriReduceMesh(TR1, Elmts2KeepID)
    else:
        TRout = TR1
    
    return TRout



    

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

# -----------------------------------------------------------------------------
def LargestEdgeConvHull(Pts, Minertia = []):
    # Compute the convex hull of the points cloud Pts and sort the edges 
    # by their length.
    # 
    # INPUTS:
    # - Pts: A Point Cloud in 2D [nx2] or 3D [nx3]
    # 
    # - Minertia: A matrice of inertia to transform the points beforehand
    # 
    # OUTPUTS:
    # - IdxPointsPair: [mx2] or [mx3] matrix of the index of pair of points 
    # forming the edges
    # 
    # - EdgesLength: a [mx1] matrix of the edges length which rows are in 
    # correspondance with IdxPointsPair matrix
    # 
    # - K: The convex hull of the point cloud
    # 
    # - Edges_Length_and_VerticesIDs_merged_sorted: A [mx3] matrix with first 
    # column corresponding to the edges length and the last two columns 
    # corresponding to the Index of the points forming the the edge.
    # -------------------------------------------------------------------------
    debug_plots = 0
    r, c = np.shape(Pts)

    if c == 2:
        # compute convex hull
        hull = ConvexHull(Pts)
        # transform it in triangulation
        #  ---------------
        # hull object doesn't remove unreferenced vertices
        # create a mask to re-index faces for only referenced vertices
        vid = np.sort(hull.vertices)
        mask = np.zeros(len(hull.points), dtype=np.int64)
        mask[vid] = np.arange(len(vid))
        # remove unreferenced vertices here
        faces = mask[hull.simplices].copy()
        # rescale vertices back to original size
        K = hull.points[vid].copy()
        #  ---------------
        KConvHull = {'Points': K, 'ConnectivityList': faces}
        
        IdxPointsPair = np.zeros((2*len(faces),2)).astype(np.int64)
        
        for pos, tri in enumerate(KConvHull['ConnectivityList']):
            ind0 = np.where(np.linalg.norm(KConvHull['Points'][tri[0]] - Pts, axis = 1) == 0)[0][0]
            ind1 = np.where(np.linalg.norm(KConvHull['Points'][tri[1]] - Pts, axis = 1) == 0)[0][0]
                        
            IdxPointsPair[2*pos] = np.array([ind0, ind1])
            IdxPointsPair[2*pos+1] = np.array([ind1, ind0])
                
        Edge_length = [np.linalg.norm(Pts[v[0]] - Pts[v[1]]) for v in IdxPointsPair]
        
        Ind_Edge = np.argsort(Edge_length)
        Ind_Edge = Ind_Edge[::-1]
        Edge_length = np.sort(Edge_length)
        Edge_length = Edge_length[::-1]
        IdxPointsPair = IdxPointsPair[Ind_Edge]

        
        Edges_Length_and_VerticesIDs_merged_sorted = np.zeros((int(len(IdxPointsPair)/2),2))
        # remove duplicated edges
        Edge_length = Edge_length[::2]
        IdxPointsPair = IdxPointsPair[::2]
        Edges_Length_and_VerticesIDs_merged_sorted[:,0] = Edge_length
        Edges_Length_and_VerticesIDs_merged_sorted[:,1:] = IdxPointsPair
        
    elif c == 3:
        if Minertia:
            Pts = (Pts - np.mean(Pts, axis = 0))*Minertia
        # compute convex hull
        hull = ConvexHull(Pts)
        # transform it in triangulation
        #  ---------------
        # hull object doesn't remove unreferenced vertices
        # create a mask to re-index faces for only referenced vertices
        vid = np.sort(hull.vertices)
        mask = np.zeros(len(hull.points), dtype=np.int64)
        mask[vid] = np.arange(len(vid))
        # remove unreferenced vertices here
        faces = mask[hull.simplices].copy()
        # rescale vertices back to original size
        K = hull.points[vid].copy()
        #  ---------------
        KConvHull = {'Points': K, 'ConnectivityList': faces}
                
        IdxPointsPair = np.zeros((3*len(faces),2)).astype(np.int64)
        
        for pos, tri in enumerate(KConvHull['ConnectivityList']):
            ind0 = np.where(np.linalg.norm(KConvHull['Points'][tri[0]] - Pts, axis = 1) == 0)[0][0]
            ind1 = np.where(np.linalg.norm(KConvHull['Points'][tri[1]] - Pts, axis = 1) == 0)[0][0]
            ind2 = np.where(np.linalg.norm(KConvHull['Points'][tri[2]] - Pts, axis = 1) == 0)[0][0]
            
            IdxPointsPair[3*pos] = np.array([ind0, ind1])
            IdxPointsPair[3*pos+1] = np.array([ind1, ind2])
            IdxPointsPair[3*pos+2] = np.array([ind2, ind0])
        
        Edge_length = [np.linalg.norm(Pts[v[0]] - Pts[v[1]]) for v in IdxPointsPair]
        
        Ind_Edge = np.argsort(Edge_length)
        Ind_Edge = Ind_Edge[::-1]
        Edge_length = np.sort(Edge_length)
        Edge_length = Edge_length[::-1]
        IdxPointsPair = IdxPointsPair[Ind_Edge]

        
        Edges_Length_and_VerticesIDs_merged_sorted = np.zeros((int(len(IdxPointsPair)/2),3))
        # remove duplicated edges
        Edge_length = Edge_length[::2]
        IdxPointsPair = IdxPointsPair[::2]
        Edges_Length_and_VerticesIDs_merged_sorted[:,0] = Edge_length
        Edges_Length_and_VerticesIDs_merged_sorted[:,1:] = IdxPointsPair
        
        if debug_plots:
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            ax.plot_trisurf(KConvHull['Points'][:,0], KConvHull['Points'][:,1], KConvHull['Points'][:,2], triangles = KConvHull['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.2, shade=False, color = 'green')
            for edge in IdxPointsPair:
                ind0 = np.where(np.linalg.norm(Pts[edge[0]] - KConvHull['Points'], axis = 1) == 0)[0][0]
                ind1 = np.where(np.linalg.norm(Pts[edge[1]] - KConvHull['Points'], axis = 1) == 0)[0][0]
                
                # [ER] debugging plot - see the kept points
                ax.scatter(KConvHull['Points'][ind0][0], KConvHull['Points'][ind0][1], KConvHull['Points'][ind0][2], color = 'red', s=100)
                ax.scatter(KConvHull['Points'][ind1][0], KConvHull['Points'][ind1][1], KConvHull['Points'][ind1][2], color = 'blue', s=100)
                
                #  [ER] debugging plot (see lines of axes)
                ax.plot([KConvHull['Points'][ind0][0], KConvHull['Points'][ind1][0]], \
                        [KConvHull['Points'][ind0][1], KConvHull['Points'][ind1][1]], \
                        [KConvHull['Points'][ind0][2], KConvHull['Points'][ind1][2]], \
                        color = 'black', linewidth=4, linestyle='solid')
    
    return IdxPointsPair, Edge_length, K, Edges_Length_and_VerticesIDs_merged_sorted

# -----------------------------------------------------------------------------
def PCRegionGrowing(Pts, Seeds, r):
    # -------------------------------------------------------------------------
    # PointCloud Region growing
    # Pts : Point Cloud
    # r : radius threshold 
    # 
    # From Seeds found all the points that are inside the spheres of radius r
    # Then use the found points as new seed loop until no new points are found 
    # to be in the spheres.
    # -------------------------------------------------------------------------
    Pts_Out = {}
    
    # Seeds in the considerated point cloud
    tree = KDTree(Pts)
    dd, ii = tree.query(Seeds.T, k=1)
    Seeds = Pts[ii]

    idx = []

    # Get the index of points within the spheres
    point_tree = spatial.cKDTree(Pts)
    # This finds the index of all points within distance r of Seeds.
    I = point_tree.query_ball_point(Seeds, r)[0]
    # remove index included in idx
    I = [i for i in I if i not in idx]
    # Update idx with found indexes
    idx += I
    idx = list(set(idx))

    while I:
        # Update Seeds with found points
        Seeds = Pts[idx]
        # Get the index of points within the spheres
        point_tree = spatial.cKDTree(Pts)
        # This finds the index of all points within distance r of Seeds.
        I = point_tree.query_ball_point(Seeds, r)[0]
        # remove index included in idx
        I = [i for i in I if i not in idx]
        # Update idx with found indexes
        idx += I
        idx = list(set(idx))

    Pts_Out = Pts[idx]
    
    return Pts_Out

# -----------------------------------------------------------------------------
def PtsOnCondylesFemur(PtsCondyle_0, Pts_Epiphysis, CutAngle, InSetRatio, ellip_dilat_fact):
    # -------------------------------------------------------------------------
    # Find points on condyles from a first 2D ellipse Fit on points identifies 
    # as certain to be on the condyle [PtsCondyle_0] and get points in +- 5 % 
    # intervall of the fitted ellipse
    # Points must be expressed in Coordinate system where Y has been identified
    # as a good initial candidates for ML axis
    # -------------------------------------------------------------------------
    
    Elps = fit_ellipse(PtsCondyle_0[:,2], PtsCondyle_0[:,0])

    Ux = np.array([np.cos(Elps['phi']), -np.sin(Elps['phi'])])
    Uy = np.array([np.sin(Elps['phi']), np.cos(Elps['phi'])])
    R = np.zeros((2,2))
    R[0] = Ux
    R[1] = Uy

    # the ellipse
    theta_r = np.linspace(0, 2*np.pi, 36)
    ellipse_x_r = InSetRatio*Elps['width']*np.cos(theta_r)
    ellipse_y_r = InSetRatio*Elps['height']*np.sin(theta_r)
    tmp_ellipse_r = np.zeros((2,len(ellipse_x_r)))
    tmp_ellipse_r[0,:] = ellipse_x_r
    tmp_ellipse_r[1,:] = ellipse_y_r

    rotated_ellipse = (R @ tmp_ellipse_r).T
    rotated_ellipse[:,0] = rotated_ellipse[:,0] + Elps['X0']
    rotated_ellipse[:,1] = rotated_ellipse[:,1] + Elps['Y0']


    OUT_Elps = ~inpolygon(Pts_Epiphysis[:,2], Pts_Epiphysis[:,0], rotated_ellipse[:,0], rotated_ellipse[:,1])

    # compute convex hull
    points = PtsCondyle_0[:,[2,0]]
    hull = ConvexHull(points)

    # ConvexHull dilated by 2.5% relative to the ellipse center distance
    IN_CH = inpolygon(Pts_Epiphysis[:,2], Pts_Epiphysis[:,0], \
                      points[hull.vertices,0] + ellip_dilat_fact*(points[hull.vertices,0] - Elps['X0']), \
                      points[hull.vertices,1] + ellip_dilat_fact*(points[hull.vertices,1] - Elps['Y0']))

    # find furthest point
    tmp_Cin = np.array([Elps['X0'], Elps['Y0']])
    PtsinEllipseCF = Pts_Epiphysis[:,[2,0]] - tmp_Cin

    SqrdDist2Center = np.dot(PtsinEllipseCF, Ux)**2 + np.dot(PtsinEllipseCF, Uy)**2 

    I = np.argmax(SqrdDist2Center)

    UEllipseCF = preprocessing.normalize(PtsinEllipseCF, axis=1)


    if np.dot(Pts_Epiphysis[I,[2,0]] - tmp_Cin, Uy) < 0:
        
        EXT_Posterior = (np.dot(UEllipseCF, Uy) < -np.cos(np.pi/2 - CutAngle*np.pi/180)) \
            | ((np.dot(UEllipseCF, Uy) < 0) & (np.dot(UEllipseCF, Ux) > 0))
            
    else:
        
        EXT_Posterior = (np.dot(UEllipseCF, Uy) > np.cos(np.pi/2 - CutAngle*np.pi/180)) \
            | ((np.dot(UEllipseCF, Uy) > 0) & (np.dot(UEllipseCF, Ux) > 0))
            
    # Points must be Outsided of the reduced ellipse and inside the Convexhull
    # I_kept = OUT_Elps & IN_CH & ~EXT_Posterior
    I_kept = OUT_Elps & IN_CH
            
    # outputs
    PtsCondyle_end = Pts_Epiphysis[I_kept]
    PtsKeptID = np.where(I_kept == True)[0]

    # plotting
    debug_plots = 0
    if debug_plots:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter(Pts_Epiphysis[:,2], Pts_Epiphysis[:,0], color = 'green')
        ax.plot(rotated_ellipse[:,0], rotated_ellipse[:,1], color = 'red')
        ax.scatter(Pts_Epiphysis[I_kept,2], Pts_Epiphysis[I_kept,0], color = 'red', marker='s')
                
        ax.scatter(np.mean(Pts_Epiphysis[:,2]), np.mean(Pts_Epiphysis[:,0]), color = 'k', marker='s')
        ax.scatter(Pts_Epiphysis[I,2], Pts_Epiphysis[I,0], color = 'r', marker='d')
        ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'k', lw=2)
        ax.scatter(PtsCondyle_0[:,2], PtsCondyle_0[:,0], color = 'k', marker='d')
        ax.plot(points[hull.vertices,0], points[hull.vertices,1], 'c--', lw=2)
    
    
    
    return PtsCondyle_end, PtsKeptID








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

#%% ---------------------------------------------------------------------------
# FittingFun
# -----------------------------------------------------------------------------
def sphere_fit(point_cloud):   
    # -------------------------------------------------------------------------
    # this fits a sphere to a collection of data using a closed form for the
    # solution (opposed to using an array the size of the data set). 
    # Minimizes Sum((x-xc)^2+(y-yc)^2+(z-zc)^2-r^2)^2
    # x,y,z are the data, xc,yc,zc are the sphere's center, and r is the radius
    #    
    # Assumes that points are not in a singular configuration, real numbers, ...
    # if you have coplanar data, use a circle fit with svd for determining the
    # plane, recommended Circle Fit (Pratt method), by Nikolai Chernov
    # http://www.mathworks.com/matlabcentral/fileexchange/22643
    # 
    # Input:
    # X: n x 3 matrix of cartesian data
    # Outputs:
    # Center: Center of sphere 
    # Radius: Radius of sphere
    # Author:
    # Alan Jennings, University of Dayton
    # 
    # Modified to add distance to sphere -> ErrorDist
    
    A = np.zeros((3,3))
    A[0,0] = np.mean(point_cloud[:,0]*(point_cloud[:,0] - np.mean(point_cloud[:,0])))
    A[0,1] = 2*np.mean(point_cloud[:,0]*(point_cloud[:,1] - np.mean(point_cloud[:,1])))
    A[0,2] = 2*np.mean(point_cloud[:,0]*(point_cloud[:,2] - np.mean(point_cloud[:,2])))
    
    A[1,1] = np.mean(point_cloud[:,1]*(point_cloud[:,1] - np.mean(point_cloud[:,1])))
    A[1,2] = 2*np.mean(point_cloud[:,1]*(point_cloud[:,2] - np.mean(point_cloud[:,2])))
    
    A[2,2] = np.mean(point_cloud[:,2]*(point_cloud[:,2] - np.mean(point_cloud[:,2])))
    
    A += A.T
    
    B = np.zeros((3,1))
    B[0] = np.mean((point_cloud[:,0]**2 + point_cloud[:,1]**2 + point_cloud[:,2]**2)*(point_cloud[:,0] - np.mean(point_cloud[:,0]))) 
    B[1] = np.mean((point_cloud[:,0]**2 + point_cloud[:,1]**2 + point_cloud[:,2]**2)*(point_cloud[:,1] - np.mean(point_cloud[:,1])))
    B[2] = np.mean((point_cloud[:,0]**2 + point_cloud[:,1]**2 + point_cloud[:,2]**2)*(point_cloud[:,2] - np.mean(point_cloud[:,2])))
    
    sphere_center = np.dot(np.linalg.inv(A), B)
    
    radius = np.sqrt(np.mean(np.sum(np.array([(point_cloud[:,0]-sphere_center[0])**2, \
                                              (point_cloud[:,1]-sphere_center[1])**2, \
                                              (point_cloud[:,2]-sphere_center[2])**2]), axis=0)))
    
    ErrorDist = []
    for p in point_cloud:
        ErrorDist.append(np.sum((p - sphere_center.T)**2) - radius**2)
            
    return sphere_center.T, radius, ErrorDist

# -----------------------------------------------------------------------------
def fitCSA(Z, Area):   
    # -------------------------------------------------------------------------
    # Create a fit of the evolution of the cross section area :
    # 1st step is to fit a double gaussian on the curve 
    # 2nd step is to use the result of the first fit to initialize a 2nd
    # fit of a gaussian plus an affine function : a1*exp(-((x-b1)/c1)^2)+d*x+e
    # Separate the diaphysis based on the hypothseis that its cross section area
    # evolve pseudo linearly along its axis while the variation are
    # exponential for the epiphysis
    # -------------------------------------------------------------------------
    Zdiaph = 0 
    Zepi = 0
    Or = 0
    # Fit function types ------------------------------------------------------
    def gauss2(x, a1, b1, c1, a2, b2, c2):
        # a1*exp(-((x-b1)/c1)^2)+a2*exp(-((x-b2)/c2)^2)
        res =   a1*np.exp(-((x - b1)/c1)**2) \
              + a2*np.exp(-((x - b2)/c2)**2)
        return res
    def gauss_lineal(x, a1, b1, c1, d, e):
        # a1*exp(-((x-b1)/c1)^2)+d*x+e
        res =   a1*np.exp(-((x - b1)/c1)**2) \
              + d*x + e
        return res
    # -------------------------------------------------------------------------

    Z0 = np.mean(Z)
    Z = Z - Z0

    Area = Area/np.mean(Area)

    AreaMax = np.max(Area)
    Imax = np.argmax(Area)

    # Orientation of bone along the y axis
    if Z[Imax] < np.mean(Z):
        Or = -1
    else:
        Or = 1

    xData = Z
    yData = Area
    # Fit model to data.
    Bounds = ([-np.infty, -np.infty, 0, -np.infty, -np.infty, 0], np.infty)
    StartPoint = np.array([AreaMax, Z[Imax], 20, np.mean(Area), np.mean(Z), 75]) 

    fitresult1, pcov = curve_fit(gauss2, xData, yData, method = 'trf', p0 = StartPoint, bounds=Bounds)

    # Fit: 'untitled fit 1'.
    StartPoint = np.array([fitresult1[0], fitresult1[1], fitresult1[2], 0, np.mean(Z)]) 

    fitresult2, pcov = curve_fit(gauss_lineal, xData, yData, method = 'trf', p0 = StartPoint)

    # Distance to linear part
    # Dist2linear = Area - (d*Z + e)
    Dist2linear = Area - (fitresult2[3]*Z + fitresult2[4])

    Zkept = Z[np.abs(Dist2linear) < np.abs(0.1*np.median(Area))]
    Zkept = Zkept[2:-2]

    # The End of the diaphysis correspond to the "end" of the gaussian curve
    # (outside the 95 %) of the gaussian curve surface
    if Or==-1:
        ZStartDiaphysis = np.max(Zkept)
    elif Or==1:
        ZStartDiaphysis = np.min(Zkept)

    # ZendDiaphysis = b1 - 3*or*c1
    ZendDiaphysis = fitresult1[1] - 3*Or*fitresult1[2]
    Zdiaph = Z0 + np.array([ZStartDiaphysis, ZendDiaphysis])

    ZStartEpiphysis = fitresult1[1] - 1.5*Or*fitresult1[2]
    Zepi = Z0 + ZStartEpiphysis
    
    # plt.figure()
    # plt.plot(xData,yData)
    # plt.plot(xData, gauss2(xData, *fitresult1), 'g--')
    # plt.plot(xData, gauss_lineal(xData, *fitresult2), 'r--')
    
    return Zdiaph, Zepi, Or


# -----------------------------------------------------------------------------
def fit_ellipse(x, y):   
    # -------------------------------------------------------------------------
    # finds the best fit to an ellipse for the given set of points.
    # 
    # Input:    x,y         - a set of points in 2 column vectors. 
    # 
    # Output:   ellipse_t - dictionary that defines the best fit to an ellipse
    #           phi       - orientation in radians of the ellipse (tilt)
    #           X0        - center at the X axis
    #           Y0        - center at the Y axis
    #           width - size of the long axis of the ellipse
    #           height- size of the short axis of the ellipse
    # 
    # Author: eravera
    # -------------------------------------------------------------------------
    ellipse_t = {}
        
    X = np.array(list(zip(x, y)))
    reg = LsqEllipse().fit(X)
    center, width, height, phi = reg.as_parameters()
        
    ellipse_t['phi'] = phi
    ellipse_t['X0'] = center[0]
    ellipse_t['Y0'] = center[1]
    ellipse_t['height'] = height
    ellipse_t['width'] = width
    
    return ellipse_t






# -----------------------------------------------------------------------------
# LSGE ------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def lsplane(X, a0 = np.array([0,0,0])):   
    # -------------------------------------------------------------------------
    # LSPLANE.M   Least-squares plane (orthogonal distance regression).
    # 
    # Version 1.0    
    # Last amended   I M Smith 27 May 2002. 
    # Created        I M Smith 08 Mar 2002
    # Modified       J B Renault 12 Jan 2017
    # 
    # Python Version
    # Created        E P Ravera 15 July 2024
    # -------------------------------------------------------------------------
    # Input    
    # X: Array [x y z] where x = vector of x-coordinates, y = vector of 
    # y-coordinates and z = vector of z-coordinates. Dimension: m x 3. 
    # 
    # <Optional... 
    # a0       Array  [v1; v2; v3] a vector to establish proper orientation 
    # for for plan normal. Dimension: 3 x 1.
    # ...>
    # 
    # Output   
    # x0: Centroid of the data = point on the best-fit plane. Dimension: 1 x 3. 
    # 
    # a: Direction cosines of the normal to the best-fit plane. Dimension: 3 x 1.
    # 
    # <Optional... 
    # d: Residuals. Dimension: m x 1. 
    # 
    # normd: Norm of residual errors. Dimension: 1 x 1. 
    # ...>
    # 
    # -------------------------------------------------------------------------
    
    # check number of data points 
    if len(X) < 3:
        print('ERROR: At least 3 data points required: ' )
    
    # calculate centroid
    x0 = np.mean(X, axis=0)
    
    # form matrix A of translated points
    A = X - x0
    
    # calculate the SVD of A
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    
    # find the smallest singular value in S and extract from V the 
    # corresponding right singular vector
    s = np.min(S)
    i = np.argmin(S)
    
    a = Vh[:,i]
    # Invert (or don"t) normal direction so to have same orientation as a0
    if np.linalg.norm(a0) != 0:
        a = np.sign(np.dot(a0, a))*a
    
    # calculate residual distances, if required
    d = U[:,i]*s
    normd = np.linalg.norm(d)
    
    return x0, a, d, normd








