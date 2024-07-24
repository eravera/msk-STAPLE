#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR(S) AND VERSION-HISTORY

Author:   Luca Modenese & Jean-Baptiste Renault. 
Copyright 2020 Luca Modenese & Jean-Baptiste Renault

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
import scipy.spatial as spatial
from scipy.spatial import ConvexHull
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from geometry import bodySide2Sign

from GIBOC_core import TriInertiaPpties, \
                        TriMesh2DProperties, \
                         TriChangeCS, \
                          plotDot, \
                           quickPlotRefSystem, \
                            TriReduceMesh, \
                             plotTriangLight, \
                              plotBoneLandmarks, \
                               cutLongBoneMesh, \
                                TriFillPlanarHoles, \
                                 computeTriCoeffMorpho, \
                                  TriDilateMesh, \
                                   TriUnite, \
                                    sphere_fit, \
                                     TriKeepLargestPatch, \
                                      TriOpenMesh, \
                                       TriPlanIntersect, \
                                        TriSliceObjAlongAxis, \
                                         fitCSA, \
                                          LargestEdgeConvHull, \
                                           PCRegionGrowing, \
                                            lsplane

from opensim_tools import computeXYZAngleSeq

from Public_functions import freeBoundary

#%% ---------------------------------------------------------------------------
# PRIVATE
# -----------------------------------------------------------------------------
def pelvis_guess_CS(pelvisTri, debug_plots = 0):
    # PELVIS_GUESS_CS Run geometrical checks to correctly estimate the 
    # orientation of an initial pelvis reference system. The convex hull for
    # the pelvis triangulation is computed, and the normal of largest triangle,
    # which connects the ASIS and PUB, identifies the frontal direction. Then, 
    # for the pelvis moved to the reference system defined by its principal
    # axes of inertia, the points that have the largest span along a
    # non-prox-distal axis are identified as the crests of the iliac bones.
    # Using the centroid of the triangulation, a cranial axis is defined and
    # the reference system finalised after the recommendation of the
    # International Society of Biomechanics (ISB).
    # 
    # Inputs :
    # pelvisTri - Dict of triangulation object of the entire pelvic geometry.
    # 
    # debug_plots - enable plots used in debugging. Value: 1 or 0 (default).
    #
    # Output :
    # RotPseudoISB2Glob - rotation matrix containing properly oriented initial
    # guess of the X, Y and Z axis of the pelvis. The axes are not
    # defined as by ISB definitions, but they are pointing in the same
    # directions, which is why is named "PseudoISB". This matrix 
    # represent the body-fixed transformation from this reference system
    # to ground.
    # 
    # LargestTriangle - Dict triangulation object that identifies the largest
    # triangle of the convex hull computed for the pelvis triangulation.
    # 
    # BL - Dict containing the bony landmarks identified 
    # on the bone geometries based on the defined reference systems. Each
    # field is named like a landmark and contain its 3D coordinates.
    # -------------------------------------------------------------------------
    
    RotPseudoISB2Glob = np.zeros((3,3))
    LargestTriangle = {}
    BL = {}
    
    # inertial axes
    V_all, CenterVol, _, D, _ =  TriInertiaPpties(pelvisTri)
    
    # smaller moment of inertia is normally the medio/lateral axis. It will be
    # updated anyway. It can be checked using D from TriInertiaPpties
    tmp_Z0 = V_all[0]
    tmp_Z0 = np.reshape(tmp_Z0,(tmp_Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    # compute convex hull
    hull = ConvexHull(pelvisTri['Points'])
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
    vertices = hull.points[vid].copy()
    #  ---------------
    PelvisConvHull = {'Points': vertices, 'ConnectivityList': faces}
    
    # Get the Post-Ant direction by finding the largest triangle of the pelvis
    # and checking the inertial axis that more closely aligns with it
    
    # Find the largest triangle on the projected Convex Hull
    PelvisConvHull_Ppties = TriMesh2DProperties(PelvisConvHull)
    I = np.argmax(PelvisConvHull_Ppties['Areas'])
    
    # Get the triangle center and normal
    LargestTriangle['Points'] = PelvisConvHull['Points'][PelvisConvHull['ConnectivityList'][I]]
    LargestTriangle['ConnectivityList'] = np.array([[0, 1, 2]])
    
    # Convert tiangulation dict to mesh object
    tmp_LargestTriangle = mesh.Mesh(np.zeros(LargestTriangle['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(LargestTriangle['ConnectivityList']):
        for j in range(3):
            tmp_LargestTriangle.vectors[i][j] = LargestTriangle['Points'][f[j],:]
    # update normals
    tmp_LargestTriangle.update_normals()
    
    # NOTE that we are working using a GIBOC reference system until where the 
    # rotation matrix is assembled using ISB conventions(specified in comments)
    
    # vector pointing forward is X
    ind_X = np.argmax(np.abs(np.dot(V_all.T, tmp_LargestTriangle.get_unit_normals().T)))
    X0 = V_all[ind_X]
    X0 = np.reshape(X0,(X0.size, 1)) # convert 1d (3,) to 2d (3,1) vector 
    
    # Reorient X0 to point posterior to anterior
    anterior_v = tmp_LargestTriangle.centroids.T - CenterVol
    X0 = preprocessing.normalize(np.sign(np.dot(anterior_v.T,X0))*X0, axis=0)
    
    # Y0 is just normal to X0 and Y0 (direction non inportant for now)
    # NOTE: Z normally points medio-laterally, Y will be cranio-caudal.
    # Directions not established yet
    Y0 = preprocessing.normalize(np.cross(tmp_Z0.T, X0.T)).T
    
    # transform the pelvis to the new set of inertial axes
    Rot = np.array([X0, Y0, tmp_Z0])
    Rot = np.squeeze(Rot)
    PelvisInertia, _, _ = TriChangeCS(pelvisTri, Rot, CenterVol)
    
    # get points that could be on iliac crests
    L1y = np.max(PelvisInertia['Points'][:, 1])
    ind_P1y = np.argmax(PelvisInertia['Points'][:, 1])
    L2y = np.min(PelvisInertia['Points'][:, 1])
    ind_P2y = np.argmin(PelvisInertia['Points'][:, 1])
    
    spanY = np.abs(L1y) + np.abs(L2y)    
    # get points that could be on iliac crests (remaning axis)
    L1z = np.max(PelvisInertia['Points'][:, 2])
    ind_P1z = np.argmax(PelvisInertia['Points'][:, 2])
    L2z = np.min(PelvisInertia['Points'][:, 2])
    ind_P2z = np.argmin(PelvisInertia['Points'][:, 2])
    
    spanZ = np.abs(L1z) + np.abs(L2z)
    # the largest span will identify the iliac crests points and discard other
    # directions
    if spanY > spanZ:
        ind_P1 = ind_P1y
        ind_P2 = ind_P2y
    else:
        ind_P1 = ind_P1z
        ind_P2 = ind_P2z
    
    # these are the most external points in the iliac wings these are iliac 
    # crest tubercles (ICT)
    P1 = pelvisTri['Points'][ind_P1]
    P2 = pelvisTri['Points'][ind_P2]
    P3 = (P1 + P2)/2 # midpoint
    
    # upward vector (perpendicular to X0)
    upw_ini = preprocessing.normalize(P3 - CenterVol.T)
    upw = upw_ini.T - (np.dot(upw_ini, X0)*X0)
    
    # vector pointing upward is Z
    ind_Z = np.argmax(np.abs(np.dot(V_all, upw)))
    Z0 = V_all[ind_Z]
    Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector 
    Z0 = preprocessing.normalize(np.sign(np.dot(upw.T,Z0))*Z0, axis=0)
    
    # Until now I have used GIBOC convention, now I build the ISB one!
    # X0 = X0_ISB, Z0 = Y_ISB
    RotPseudoISB2Glob = np.zeros((3,3))
    RotPseudoISB2Glob[:,0] = X0.T
    RotPseudoISB2Glob[:,1] = Z0.T
    RotPseudoISB2Glob[:,2] = np.cross(X0.T, Z0.T)
    
    # export markers
    BL['ICT1'] = P1
    BL['ICT2'] = P2
    
    # debugs Plots
    if debug_plots:
        
        # Figure 1: X0 should be pointing anteriorly-no interest in other axes
        fig = plt.figure(1)
        ax1 = fig.add_subplot(111, projection = '3d')
        # plot mesh
        ax1.plot_trisurf(pelvisTri['Points'][:,0], pelvisTri['Points'][:,1], pelvisTri['Points'][:,2], \
                         triangles = pelvisTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False)
        # plot X0 arrow
        ax1.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  X0[0], X0[1], X0[2], \
                  color='r', length = 60)
        # plot Y0 arrow
        ax1.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  Y0[0], Y0[1], Y0[2], \
                  color='g', length = 60)
        # plot Z0 arrow
        ax1.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  tmp_Z0[0], tmp_Z0[1], tmp_Z0[2], \
                  color='b', length = 60)
        
        #label axes 
        ax1.set_xlabel('X') 
        ax1.set_ylabel('Y') 
        ax1.set_zlabel('Z')  
        
        ax1.set_title('X0 should be pointing anteriorly - no interest in other axes')
        
        # Figure 2: Check axes of inertia orientation
        fig = plt.figure(2)
        ax2 = fig.add_subplot(111, projection = '3d')
        # plot mesh
        ax2.plot_trisurf(PelvisInertia['Points'][:,0], PelvisInertia['Points'][:,1], PelvisInertia['Points'][:,2], \
                         triangles = PelvisInertia['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False)
        # plot i unit vector
        ax2.quiver(0, 0, 0, 1, 0, 0, color='r', length = 60)
        # plot j unit vector
        ax2.quiver(0, 0, 0, 0, 1, 0, color='g', length = 60)
        # plot k unit vector
        ax2.quiver(0, 0, 0, 0, 0, 1, color='b', length = 60)
        
        #label axes 
        ax2.set_xlabel('X') 
        ax2.set_ylabel('Y') 
        ax2.set_zlabel('Z')  
        
        ax2.set_title('Check axes of inertia orientation')
        
        # Figure 3: Points should be external points in the iliac wings (inertia ref syst)
        fig = plt.figure(3)
        ax3 = fig.add_subplot(111, projection = '3d')
        # plot mesh
        ax3.plot_trisurf(PelvisInertia['Points'][:,0], PelvisInertia['Points'][:,1], PelvisInertia['Points'][:,2], \
                         triangles = PelvisInertia['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False)
        # plot landmark 1 in ASIS
        plotDot(PelvisInertia['Points'][ind_P1,:], ax3, 'g', 7)
        # plot landmark 2 in ASIS
        plotDot(PelvisInertia['Points'][ind_P2,:], ax3, 'g', 7)
        
        #label axes 
        ax3.set_xlabel('X') 
        ax3.set_ylabel('Y') 
        ax3.set_zlabel('Z')  
        
        ax3.set_title('Points should be external points in the iliac wings (inertia ref syst)')
        
        # Figure 4: Points should be external points in the iliac wings (glob ref syst)
        fig = plt.figure(4)
        ax4 = fig.add_subplot(111, projection = '3d')
        # plot mesh
        ax4.plot_trisurf(pelvisTri['Points'][:,0], pelvisTri['Points'][:,1], pelvisTri['Points'][:,2], \
                         triangles = pelvisTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False)
        # plot landmark 1 in ASIS
        plotDot(P1, ax4, 'k', 7)
        # plot landmark 2 in ASIS
        plotDot(P2, ax4, 'k', 7)
        # plot landmark 3: ASIS midpoint
        plotDot(P3, ax4, 'k', 7)
        # plot Z0 arrow
        ax4.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  Z0[0], Z0[1], Z0[2], \
                  color='b', length = 60)
        
        #label axes 
        ax4.set_xlabel('X') 
        ax4.set_ylabel('Y') 
        ax4.set_zlabel('Z')  
                
        ax4.set_title('Points should be external points in the iliac wings (glob ref syst)')
        
        # Figure 5: plot pelvis, convex hull, largest identified triangle and ISB pelvis reference system
        fig = plt.figure(5)
        ax5 = fig.add_subplot(111, projection = '3d')
        
        tmp = {}
        tmp['V'] = RotPseudoISB2Glob
        tmp['Origin'] = P3
        # plot pelvis, convex hull, largest identified triangle
        ax5.plot_trisurf(pelvisTri['Points'][:,0], pelvisTri['Points'][:,1], pelvisTri['Points'][:,2], \
                         triangles = pelvisTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, color = 'b', shade=False)
        ax5.plot_trisurf(PelvisConvHull['Points'][:,0], PelvisConvHull['Points'][:,1], PelvisConvHull['Points'][:,2], \
                         triangles = PelvisConvHull['ConnectivityList'], edgecolor=[[0.3,0.3,0.3]], linewidth=1.0, alpha=0.2, color = 'c', shade=False)
        ax5.plot_trisurf(LargestTriangle['Points'][:,0], LargestTriangle['Points'][:,1], LargestTriangle['Points'][:,2], \
                         triangles = LargestTriangle['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.8, color = 'r', shade=False)
        # plot axes of pelvis (ISB)
        quickPlotRefSystem(tmp, ax5)
        # plot landmarks
        plotDot(P1, ax5, 'k', 7)
        plotDot(P2, ax5, 'k', 7)
        plotDot(P3, ax5, 'k', 7)
        
        #label axes 
        ax5.set_xlabel('X') 
        ax5.set_ylabel('Y') 
        ax5.set_zlabel('Z')  
        # Remove grid
        ax5.grid(False)
    # END debugs Plots -----
    
    return RotPseudoISB2Glob, LargestTriangle, BL

# -----------------------------------------------------------------------------
def femur_guess_CS(Femur, debug_plots = 0):
    # Provide an informed guess of the femur orientation.
    # 
    # General Idea
    # The idea is to exploit the fact that the distal epiphysis of the bone is
    # relatively symmetrical relative to the diaphysis axis, while the one from
    # the proximal epiphysis is not because of the femoral head.
    # So if you deform the femur along the second principal inertial deirection
    # which is relatively medial to lateral, the centroid of the distal
    # epiphysis should be closer to the first principal inertia axis than the
    # one from the proximal epiphysis.
    # 
    # Inputs:
    # Femur - A Dict triangulation of the complete femur.
    # 
    # debug_plots - enable plots used in debugging. Value: 1 or 0 (default).
    # 
    # Output:
    # Z0 - A unit vector giving the distal to proximal direction.
    # -------------------------------------------------------------------------
    # Convert tiangulation dict to mesh object --------
    tmp_Femur = mesh.Mesh(np.zeros(Femur['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(Femur['ConnectivityList']):
        for j in range(3):
            tmp_Femur.vectors[i][j] = Femur['Points'][f[j],:]
    # update normals
    tmp_Femur.update_normals()
    # ------------------------------------------------
    
    Z0 = np.zeros((3,1))
    
    # Get the principal inertia axis of the femur (potentially wrongly orientated)
    V_all, CenterVol, _, _, _ = TriInertiaPpties(Femur)
    Z0 = V_all[0]
    Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    Y0 = V_all[1]
    Y0 = np.reshape(Y0,(Y0.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    # Deform the femur along the 2nd principal direction
    Pts = Femur['Points'] - CenterVol.T
    def_matrix = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
    Pts_deformed = np.dot(np.dot(np.dot(Pts, V_all), def_matrix), V_all.T)
    Femur['Points'] = Pts_deformed + CenterVol.T
    
    # Get both epiphysis of the femur (10% of the length at both ends)
    # Tricks : Here we use Z0 as the initial direction for
    TrEpi1, TrEpi2 = cutLongBoneMesh(Femur, Z0, 0.10)
    
    # Get the central 60% of the bone -> The femur diaphysis
    LengthBone = np.max(np.dot(Femur['Points'], Z0)) - np.min(np.dot(Femur['Points'], Z0))
    L_ratio = 0.20
    
    # First remove the top 20% percent
    alt_top = np.max(np.dot(Femur['Points'], Z0)) - L_ratio*LengthBone
    ElmtsTmp1 = np.where(np.dot(tmp_Femur.centroids, Z0) < alt_top)[0]
    TrTmp1 = TriReduceMesh(Femur, ElmtsTmp1)
    TrTmp1 = TriFillPlanarHoles(TrTmp1)
    # Convert tiangulation dict to mesh object --------
    tmp_TrTmp1 = mesh.Mesh(np.zeros(TrTmp1['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(TrTmp1['ConnectivityList']):
        for j in range(3):
            tmp_TrTmp1.vectors[i][j] = TrTmp1['Points'][f[j],:]
    # update normals
    tmp_TrTmp1.update_normals()
    # ------------------------------------------------
    
    # Then remove the bottom 20% percent
    alt_bottom = np.min(np.dot(Femur['Points'], Z0)) + L_ratio*LengthBone
    ElmtsTmp2 = np.where(np.dot(tmp_TrTmp1.centroids, Z0) > alt_bottom)[0]
    TrTmp2 = TriReduceMesh(TrTmp1, ElmtsTmp2)
    FemurDiaphysis = TriFillPlanarHoles(TrTmp2)
    
    # Get the principal inertia axis of the diaphysis (potentially wrongly orientated)
    V_all, CenterVol_dia, _, _, _ = TriInertiaPpties(FemurDiaphysis)
    Z0_dia = V_all[0]
    Z0_dia = np.reshape(Z0_dia,(Z0_dia.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    # Get the distance of the centroids of each epihyisis part to the diaphysis
    # axis
    _, CenterEpi1, _, _, _ = TriInertiaPpties(TrEpi1)
    _, CenterEpi2, _, _, _ = TriInertiaPpties(TrEpi2)
    
    distToDiaphAxis1 = np.linalg.norm(np.cross((CenterEpi1 - CenterVol).T, Z0_dia.T))
    distToDiaphAxis2 = np.linalg.norm(np.cross((CenterEpi2 - CenterVol).T, Z0_dia.T))
    
    if distToDiaphAxis1 < distToDiaphAxis2:
        # It means that epi1 is the distal epihysis and epi2 the proximal
        U_DistToProx = CenterEpi2 - CenterEpi1
        
    elif distToDiaphAxis1 > distToDiaphAxis2:
        # It means that epi1 is the proximal epihysis and epi2 the distal
        U_DistToProx = CenterEpi1 - CenterEpi2
    
    # Reorient Z0 of the femur according to the found direction
    Z0 = np.sign(np.dot(U_DistToProx.T, Z0))*Z0
    
    # Warning flag for unclear results
    if np.abs(distToDiaphAxis1 - distToDiaphAxis2)/(distToDiaphAxis1 + distToDiaphAxis2) < 0.20:
        logging.exception('The distance to the femur diaphysis axis for the femur' + \
        ' epihysis where not very different. Orientation of Z0, distal to proximal axis, of the femur could be incorrect. \n')
    
    # debug plot
    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        
        ax.scatter(CenterEpi1[0], CenterEpi1[1], CenterEpi1[2], marker='*', color = 'r')
        ax.scatter(CenterEpi2[0], CenterEpi2[1], CenterEpi2[2], marker='*', color = 'b')
        
        ax.quiver(CenterVol_dia[0], CenterVol_dia[1], CenterVol_dia[2], \
                  Z0_dia[0], Z0_dia[1], Z0_dia[2], \
                  color='k', length = 200)
        ax.quiver(CenterVol_dia[0], CenterVol_dia[1], CenterVol_dia[2], \
                  -Z0_dia[0], -Z0_dia[1], -Z0_dia[2], \
                  color='gray', length = 200)
        ax.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  Z0[0], Z0[1], Z0[2], \
                  color='b', length = 300)
        ax.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  Y0[0], Y0[1], Y0[2], \
                  color='g', length = 50)
        ax.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  -Y0[0], -Y0[1], -Y0[2], \
                  color='olivedrab', length = 50)
        
        ax.plot_trisurf(Femur['Points'][:,0], Femur['Points'][:,1], Femur['Points'][:,2], \
                         triangles = Femur['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, color = 'cyan', shade=False)
        
        ax.set_box_aspect([1,1,1])
        ax.grid(True)
        
    return Z0

# -----------------------------------------------------------------------------
def GIBOC_femur_fitSphere2FemHead(ProxFem = {}, CSs = {}, CoeffMorpho = 1, debug_plots = 0, debug_prints = 0):
    # -------------------------------------------------------------------------
    
    FemHead = {}
    # Convert tiangulation dict to mesh object --------
    tmp_ProxFem = mesh.Mesh(np.zeros(ProxFem['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(ProxFem['ConnectivityList']):
        for j in range(3):
            tmp_ProxFem.vectors[i][j] = ProxFem['Points'][f[j],:]
    # update normals
    tmp_ProxFem.update_normals()
    # ------------------------------------------------
    print('Computing centre of femoral head:')

    # Find the most proximal on femur top head
    I_Top_FH = []
    I_Top_FH.append(np.argmax(np.dot(tmp_ProxFem.centroids, CSs['Z0'])))

    # most prox point (neighbors of I_Top_FH)
    for nei in ProxFem['ConnectivityList'][I_Top_FH].reshape(-1, 1):
        I_Top_FH += list(list(np.where(ProxFem['ConnectivityList'] == nei))[0])

    # remove duplicated vertex
    I_Top_FH = list(set(I_Top_FH))
    # triang around it
    Face_Top_FH = TriReduceMesh(ProxFem, I_Top_FH)

    # create a triang with them
    Patch_Top_FH = TriDilateMesh(ProxFem,Face_Top_FH,40*CoeffMorpho)

    # Get an initial ML Axis Y0 (pointing medio-laterally)
    # NB: from centerVol, OT points upwards to ~HJC, that is more medial than
    # Z0, hence cross(CSs.Z0,OT) points anteriorly and Y0 medially
    OT = np.mean(Patch_Top_FH['Points'], axis=0) - CSs['CenterVol'].T
    tmp_Y0 = np.cross(np.cross(CSs['Z0'].T, OT), CSs['Z0'].T)
    CSs['Y0'] = preprocessing.normalize(tmp_Y0.T, axis=0)

    # Find a the most medial (MM) point on the femoral head (FH)
    I_MM_FH = []
    I_MM_FH.append(np.argmax(np.dot(tmp_ProxFem.centroids, CSs['Y0'])))

    # most prox point (neighbors of I_MM_FH)
    for nei in ProxFem['ConnectivityList'][I_MM_FH].reshape(-1, 1):
        I_MM_FH += list(list(np.where(ProxFem['ConnectivityList'] == nei))[0])

    # remove duplicated vertex
    I_MM_FH = list(set(I_MM_FH))
    # triang around it
    Face_MM_FH = TriReduceMesh(ProxFem, I_MM_FH)

    # create a triang with them
    Patch_MM_FH = TriDilateMesh(ProxFem, Face_MM_FH, 40*CoeffMorpho)

    # STEP1: first sphere fit
    FemHead0 = TriUnite(Patch_MM_FH, Patch_Top_FH)

    Center, Radius, ErrorDist = sphere_fit(FemHead0['Points'])
    sph_RMSE = np.mean(np.abs(ErrorDist))/10

    # print
    print('     Fit #1: RMSE: ' + str(sph_RMSE) + ' mm');

    if debug_prints:
        print('----------------')
        print('First Estimation')
        print('----------------')
        print('Centre: ' + str(Center))
        print('Radius: ' + str(Radius))
        print('Mean Res: ' + str(sph_RMSE))
        print('-----------------')

    # Write to the results dictionary
    CSs['CenterFH']  = Center.T
    CSs['RadiusFH']  =  Radius
    CSs['sph_RMSEFH']  =  sph_RMSE

    # STEP2: dilate femoral head mesh and sphere fit again
    # IMPORTANT: TriDilateMesh "grows" the original mesh, does not create a larger one!
    FemHead_dil_coeff = 1.5

    DilateFemHeadTri = TriDilateMesh(ProxFem, FemHead0, round(FemHead_dil_coeff*Radius*CoeffMorpho))
    CenterFH, RadiusDil, ErrorDistCond = sphere_fit(DilateFemHeadTri['Points'])

    sph_RMSECond = np.mean(np.abs(ErrorDistCond))/10

    # print
    print('     Fit #2: RMSE: ' + str(sph_RMSECond) + ' mm');

    if debug_prints:
        print('----------------')
        print('Cond Estimation')
        print('----------------')
        print('Centre: ' + str(CenterFH))
        print('Radius: ' + str(RadiusDil))
        print('Mean Res: ' + str(sph_RMSECond))
        print('-----------------')

    # check
    if ~(RadiusDil > Radius):
        print('Dilated femoral head smaller than original mesh. Please check manually.')

    # Theorical Normal of the face (from real fem centre to dilate one)
    # Convert tiangulation dict to mesh object --------
    TriDilateFemHead = mesh.Mesh(np.zeros(DilateFemHeadTri['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(DilateFemHeadTri['ConnectivityList']):
        for j in range(3):
            TriDilateFemHead.vectors[i][j] = DilateFemHeadTri['Points'][f[j],:]
    # update normals
    TriDilateFemHead.update_normals()
    # ------------------------------------------------
    CPts_PF_2D = TriDilateFemHead.centroids - CenterFH
    normal_CPts_PF_2D = preprocessing.normalize(CPts_PF_2D, axis=1)

    # COND1: Keep points that display a less than 10deg difference between the actual
    # normals and the sphere simulated normals
    FemHead_normals_thresh = 0.975 # acosd(0.975) = 12.87 deg

    # sum((normal_CPts_PF_2D.*DilateFemHeadTri.faceNormal),2)
    Cond1 = [1 if np.abs((np.dot(val, TriDilateFemHead.get_unit_normals()[pos]))) > \
              FemHead_normals_thresh else 0 for pos, val in enumerate(normal_CPts_PF_2D)]

    # COND2: Delete points far from sphere surface outside [90%*Radius 110%*Radius]
    Cond2 = [1 if abs(np.sqrt(np.dot(val, CPts_PF_2D[pos])) - Radius) < \
              0.1*Radius else 0 for pos, val in enumerate(CPts_PF_2D)]

    # [LM] I have found both conditions do not work always, when combined
    # check if using both conditions produces results
    single_cond = 0
    min_number_of_points = 20

    if np.sum(np.logical_and(Cond1, Cond2)) > min_number_of_points:
        applied_Cond = list(np.logical_and(Cond1, Cond2))
    else:
        # flag that only one condition is used
        single_cond = 1
        # cond1_count = np.sum(np.array(Cond1))
        applied_Cond = Cond1

    # search within conditions Cond1 and Cond2
    Face_ID_PF_2D_onSphere = np.where(applied_Cond)[0]

    # get the mesh and points on the femoral head 
    FemHead = TriReduceMesh(DilateFemHeadTri, Face_ID_PF_2D_onSphere)

    # if just one condition is active JB suggests to keep largest patch
    if single_cond:
        FemHead = TriKeepLargestPatch(FemHead)

    FemHead = TriOpenMesh(ProxFem, FemHead, 3*CoeffMorpho)

    # Fit the last Sphere
    CenterFH_Renault, Radius_Renault, ErrorDistFinal = sphere_fit(FemHead['Points'])

    sph_RMSEFinal = np.mean(np.abs(ErrorDistFinal))/10

    # print
    print('     Fit #3: RMSE: ' + str(sph_RMSEFinal) + ' mm')

    # feedback on fitting
    # chosen as large error based on error in regression equations (LM)
    fit_thereshold = 25;
    if sph_RMSE > fit_thereshold:
        logging.warning('Large sphere fit RMSE: ' + str(sph_RMSE) + ' (> ' + str(fit_thereshold) + 'mm).')
    else:
        print('Reasonable sphere fit error (RMSE < ' + str(fit_thereshold) + 'mm).')

    if debug_prints:
        print('----------------')
        print('Final Estimation')
        print('----------------')
        print('Centre: ' + str(CenterFH_Renault))
        print('Radius: ' + str(Radius_Renault))
        print('Mean Res: ' + str(sph_RMSEFinal))
        print('-----------------')

    # Write to the results dictionary
    CSs['CenterFH_Renault']  = CenterFH_Renault.T
    CSs['RadiusFH_Renault']  =  Radius_Renault
    CSs['sph_RMSEFH_Renault']  =  sph_RMSEFinal

    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(ProxFem['Points'][:,0], ProxFem['Points'][:,1], ProxFem['Points'][:,2], \
                        triangles = ProxFem['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'gray')
        ax.plot_trisurf(FemHead['Points'][:,0], FemHead['Points'][:,1], FemHead['Points'][:,2], \
                        triangles = FemHead['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'green')
        # Plot sphere
        # Create a sphere
        phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
        x = CSs['RadiusFH_Renault']*np.sin(phi)*np.cos(theta)
        y = CSs['RadiusFH_Renault']*np.sin(phi)*np.sin(theta)
        z = CSs['RadiusFH_Renault']*np.cos(phi)

        ax.plot_surface(x + CSs['CenterFH_Renault'][0], y + CSs['CenterFH_Renault'][1], z + CSs['CenterFH_Renault'][2], \
                        color = 'blue', alpha=0.4)
        
        ax.set_box_aspect([1,1,1])
        plt.show()

    return CSs, FemHead

# -----------------------------------------------------------------------------
def Kai2014_femur_fitSphere2FemHead(ProxFem = {}, CS = {}, CoeffMorpho = 1, debug_plots = 0, debug_prints = 0):
    # -------------------------------------------------------------------------
    # Custom implementation of the method for defining a reference system of 
    # the femur described in the following publication: 
    # Kai, Shin, et al. Journal of biomechanics 47.5 (2014): 1229-1233.
    # https://doi.org/10.1016/j.jbiomech.2013.12.013.
    # 
    # The algorithm slices the tibia along the vertical axis identified via
    # principal component analysis, identifies the largest section and fits an
    # ellips to it. It finally uses the ellipse axes to define the reference
    # system. This implementation includes several non-obvious checks to ensure 
    # that the bone geometry is always sliced in the correct direction.
    # 
    # Inputs:
    # ProxFem - Dictionary triangulation object of the proximal femoral geometry.
    # CS - Dictionary containing preliminary information about the bone
    # morphology. This function requires the following fields:
    # * CS.Z0: an estimation of the proximal-distal direction.
    # * CS.CentreVol: estimation of the centre of the volume (for plotting).
    # 
    # debug_plots - enable plots used in debugging. Value: 1 or 0 (default).
    # 
    # debug_prints - enable prints for debugging. Value: 1 or 0 (default).
    # 
    # Outputs:
    # CS - updated dictionary with the following fields:
    # * CS.Z0: axis pointing cranially.
    # * CS.Y0: axis pointing medially
    # * CS.X0: axis perpendicular to the previous two.
    # * CS.CentreVol: coords of the "centre of mass" of the triangulation
    # * CS.CenterFH_Kai: coords of the centre of the sphere fitted 
    # to the centre of the femoral head.
    # * CS.RadiusFH_Kai: radius of the sphere fitted to the centre of the
    # femoral head.
    # MostProxPoint - coordinates of the most proximal point of the femur,
    # located on the femoral head.
    # -------------------------------------------------------------------------
    # Convert tiangulation dict to mesh object --------
    tmp_ProxFem = mesh.Mesh(np.zeros(ProxFem['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(ProxFem['ConnectivityList']):
        for j in range(3):
            tmp_ProxFem.vectors[i][j] = ProxFem['Points'][f[j],:]
    # update normals
    tmp_ProxFem.update_normals()
    # ------------------------------------------------

    # parameters used to identify artefact sections.
    sect_pts_limit = 15

    # plane normal must be negative (GIBOC)
    corr_dir = -1
    print('Computing centre of femoral head:')

    # Find the most proximal point
    I_Top_FH = np.argmax(np.dot(ProxFem['Points'], CS['Z0']))
    MostProxPoint = ProxFem['Points'][I_Top_FH]
    MostProxPoint = np.reshape(MostProxPoint,(MostProxPoint.size, 1)) # convert 1d (3,) to 2d (3,1) vector

    # completing the inertia-based reference system the most proximal point on the
    # fem head is medial wrt to the inertial axes.
    medial_to_z = MostProxPoint - CS['CenterVol']
    tmp_Y0 = np.cross(np.cross(CS['Z0'].T, medial_to_z.T), CS['Z0'].T)
    CS['Y0'] = preprocessing.normalize(tmp_Y0.T, axis=0)
    CS['X0'] = np.cross(CS['Y0'].T, CS['Z0'].T).T
    CS['Origin']  = CS['CenterVol']

    # Slice the femoral head starting from the top
    Ok_FH_Pts = []
    Ok_FH_Pts_med = []
    # d = MostProxPoint*CS['Z0'] - 0.25
    d = np.dot(MostProxPoint.T, CS['Z0']) - 0.25
    keep_slicing = 1
    count = 1
    max_area_so_far = 0

    # print
    print('  Slicing proximal femur...')

    while keep_slicing > 0:
    #     # slice the proximal femur
        Curves , _, _ = TriPlanIntersect(ProxFem, corr_dir*CS['Z0'], d)
        Nbr_of_curves = len(Curves)
        
        # counting slices
        if debug_prints:
            print('section #' + str(count) + ': ' + str(Nbr_of_curves) + ' curves.')
        count += 1
        
        # stop if there is one curve after Curves>2 have been processed
        if Nbr_of_curves == 1 and Ok_FH_Pts_med == []:
            Ok_FH_Pts_med += list(Curves['1']['Pts'])
            break
        else:
            d -= 1
        
        # with just one curve save the slice: it's the femoral head
        if Nbr_of_curves == 1:
            Ok_FH_Pts += Curves['1']['Pts']
            # with more than one slice
        elif Nbr_of_curves > 1:
            # keep just the section with largest area.
            # the assumption is that the femoral head at this stage is larger
            # than the tip of the greater trocanter
            if Nbr_of_curves == 2 and len(Curves['2']['Pts']) < sect_pts_limit:
                print('Slice recognized as artefact. Skipping it.')
                continue
            else:
                areas = [Curves[key]['Area'] for key in Curves.keys()] 
                max_area = np.max(areas)
                ind_max_area = np.argmax(areas)
                Ok_FH_Pts_med += Curves[str(ind_max_area)]['Pts']
                areas = []
                
                if max_area >= max_area_so_far:
                    max_area_so_far = max_area
                else:
                    print('Reached femoral neck. End of slicing...')
                    keep_slicing = 0
                    continue
        # -------------------------------
        # THIS ATTEMPT DID NOT WORK WELL
        # -------------------------------
        # if I assume centre of CT/MRI is always more medial than HJC
        # then medial points can be identified as closer to mid
        # it is however a weak solution - depends on medical images.
        #       ind_med_point = abs(Curves(i).Pts(:,1))<abs(MostProxPoint(1))
        # -------------------------------
        # 
        # -------------------------------
        # THIS ATTEMPT DID NOT WORK WELL
        # -------------------------------
        # More robust (?) to check if the cross product of
        # dot ( cross( (x_P-x_MostProx), Z0 ) , front ) > 0
        # v_MostProx2Points = bsxfun(@minus,  Curves(i).Pts, MostProxPoint);
        # this condition is valid for right leg, left should be <0
        #       ind_med_point = (medial_dir'*bsxfun(@cross, v_MostProx2Points', up))>0;
        #       Ok_FH_Pts_med = [Ok_FH_Pts_med; Curves(i).Pts(ind_med_point,:)];
        # -------------------------------
                    
    # print
    print('  Sliced #' + str(count) + ' times') 
            
    # assemble the points from one and two curves
    fitPoints = Ok_FH_Pts + Ok_FH_Pts_med
    fitPoints = np.array(fitPoints)
    # NB: exclusind this check did NOT alter the results in most cases and
    # offered more point for fitting
    # -----------------
    # keep only the points medial to MostProxPoint according to the reference
    # system X0-Y0-Z0
    ind_keep = np.dot(fitPoints-MostProxPoint.T, CS['Y0']) > 0
    fitPoints = fitPoints[np.where(ind_keep)[0]]
    # -----------------

    # fit sphere
    CenterFH, Radius, ErrorDist = sphere_fit(fitPoints)

    sph_RMSE = np.mean(np.abs(ErrorDist))/10

    if debug_prints:
        print('----------------')
        print('Final Estimation')
        print('----------------')
        print('Centre: ' + str(CenterFH))
        print('Radius: ' + str(Radius))
        print('Mean Res: ' + str(sph_RMSE))
        print('-----------------')

    # feedback on fitting
    # chosen as large error based on error in regression equations (LM)
    fit_thereshold = 20
    if sph_RMSE > fit_thereshold:
        logging.warning('Large sphere fit RMSE: ' + str(sph_RMSE) + '(>' + str(fit_thereshold) + 'mm).')
        # print('Large sphere fit RMSE: ' + str(sph_RMSE) + '(>' + str(fit_thereshold) + 'mm).')
    else:
        print('  Reasonable sphere fit error (RMSE<' + str(fit_thereshold) + 'mm).')

    # body reference system
    CS['CenterFH_Kai'] = CenterFH[0]
    CS['RadiusFH_Kai'] = Radius

    if debug_plots:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(ProxFem['Points'][:,0], ProxFem['Points'][:,1], ProxFem['Points'][:,2], \
                        triangles = ProxFem['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'gray')
        
        ax.scatter(fitPoints[:,0], fitPoints[:,1], fitPoints[:,2], color = "green")
        # Plot sphere
        # Create a sphere
        phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
        x = CS['RadiusFH_Kai']*np.sin(phi)*np.cos(theta)
        y = CS['RadiusFH_Kai']*np.sin(phi)*np.sin(theta)
        z = CS['RadiusFH_Kai']*np.cos(phi)

        ax.plot_surface(x + CS['CenterFH_Kai'][0], y + CS['CenterFH_Kai'][1], z + CS['CenterFH_Kai'][2], \
                        color = 'blue', alpha=0.4)
        
        plotDot(MostProxPoint, ax, 'r', 4)
        plotDot(CS['Origin'], ax, 'k', 6)
        tmp = {}
        tmp['V'] = CS['V_all']
        tmp['Origin'] = CS['CenterVol']
        quickPlotRefSystem(tmp, ax)

    return CS, MostProxPoint   

# -----------------------------------------------------------------------------
def GIBOC_isolate_epiphysis(TriObj, Z0, prox_epi):
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------    
    EpiTri = {}
    
    # Convert tiangulation dict to mesh object --------
    tmp_TriObj = mesh.Mesh(np.zeros(TriObj['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(TriObj['ConnectivityList']):
        for j in range(3):
            tmp_TriObj.vectors[i][j] = TriObj['Points'][f[j],:]
    # update normals
    tmp_TriObj.update_normals()
    # ------------------------------------------------

    # -------
    # First 0.5 mm in Start and End are not accounted for, for stability.
    slice_step = 1 # mm 
    Areas, Alt, _, _, _ = TriSliceObjAlongAxis(TriObj, Z0, slice_step)

    # removes mesh above the limit of epiphysis (Zepi)
    Areas = np.array(list(Areas.values()))
    _ , Zepi, _ = fitCSA(Alt, Areas)

    # choose the bone part of interest
    if prox_epi == 'proximal':
        ElmtsEpi = np.where(np.dot(tmp_TriObj.centroids, Z0) > Zepi)[0]
    elif prox_epi == 'distal':
        ElmtsEpi = np.where(np.dot(tmp_TriObj.centroids, Z0) < Zepi)[0]

    # return the triangulation
    EpiTri = TriReduceMesh(TriObj, ElmtsEpi)
    
    return EpiTri
    
# -----------------------------------------------------------------------------
def GIBOC_femur_processEpiPhysis(EpiFem, CSs, Inertia_Vects, edge_threshold = 0.5, axes_dev_thresh = 0.75):
    # -------------------------------------------------------------------------
    
    debug_plots = 1
    debug_prints = 0

    # gets largest convex hull
    IdxPointsPair, Edges, _, _ = LargestEdgeConvHull(EpiFem['Points'])

    # facets are free boundaries if referenced by only one triangle
    EpiFem_freeBoundary = freeBoundary(EpiFem)

    # Keep elements that are not connected to the proximal cut and that are longer 
    # than half of the longest Edge (default)
    # NB Edges is an ordered vector -> first is largest, and vector Ikept should 
    # store the largest elements

    # Index of nodes identified on condyles
    IdCdlPts = []
    i = 0
    Ikept = []
    while len(Ikept) != np.sum(Edges > edge_threshold*Edges[1]):
        
        if (IdxPointsPair[i,0] not in EpiFem_freeBoundary['ID']) and (IdxPointsPair[i,1] not in EpiFem_freeBoundary['ID']):
            Ikept.append(i)
            
        i += 1
    IdCdlPts = IdxPointsPair[Ikept]
    # for pos, edge in enumerate(IdxPointsPair):
    #     if 2*len(IdCdlPts) > np.sum(Edges > edge_threshold*Edges[1]):
    #         break
    #     else:
    #         if (edge[0] not in EpiFem_freeBoundary['ID']) and (edge[1] not in EpiFem_freeBoundary['ID']):
    #             IdCdlPts.append(edge)
    # IdCdlPts = np.array(IdCdlPts)
    
    if debug_plots:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(EpiFem['Points'][:,0], EpiFem['Points'][:,1], EpiFem['Points'][:,2], triangles = EpiFem['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.2, shade=False, color = 'yellow')

        for edge in IdCdlPts:
            # [LM] debugging plot - see the kept points
            ax.scatter(EpiFem['Points'][edge[0]][0], EpiFem['Points'][edge[0]][1], EpiFem['Points'][edge[0]][2], color = 'red', s=100)
            ax.scatter(EpiFem['Points'][edge[1]][0], EpiFem['Points'][edge[1]][1], EpiFem['Points'][edge[1]][2], color = 'blue', s=100)
            
            #  [LM] debugging plot (see lines of axes)
            ax.plot([EpiFem['Points'][edge[0]][0], EpiFem['Points'][edge[1]][0]], \
                    [EpiFem['Points'][edge[0]][1], EpiFem['Points'][edge[1]][1]], \
                    [EpiFem['Points'][edge[0]][2], EpiFem['Points'][edge[1]][2]], \
                    color = 'black', linewidth=4, linestyle='solid')
    
    # check on number of saved edges
    if debug_prints:
        N_edges = len(Edges)
        N_saved_edges = len(IdCdlPts)
        print('Processing ' + str(np.round(N_saved_edges/N_edges*100,2)) + ' % of edges in convex hull.')

    # Axes vector of points pairs
    Axes = EpiFem['Points'][[item[0] for item in IdCdlPts]] - EpiFem['Points'][[item[1] for item in IdCdlPts]]

    # Remove duplicate Axes that are not directed from Lateral to Medial (CSs.Y0)
    I_Axes_NOduplicate = np.where(np.dot(Axes,CSs['Y0']) >= 0)[0]

    IdCdlPts = IdCdlPts[I_Axes_NOduplicate]
    Axes = Axes[I_Axes_NOduplicate]

    # normalize Axes to get unitary vectors
    U_Axes = preprocessing.normalize(Axes, axis=1)

    # delete if too far from inertial medio-Lat axis;
    # [LM] 0.75 -> acod(0.75) roughly 41 deg
    ind_NOdeviant_axes = np.abs(np.dot(U_Axes,Inertia_Vects[:,2])) >= axes_dev_thresh
    IdCdlPts = IdCdlPts[ind_NOdeviant_axes]
    U_Axes = U_Axes[ind_NOdeviant_axes]

    # region growing (point, seed, radius)
    S = np.mean(U_Axes, axis = 0)
    S = np.reshape(S,(S.size, 1)) # convert 1d (3,) to 2d (3,1) vector 
    Seeds = preprocessing.normalize(S, axis=0)
    r = 0.1
    U_Axes_Good = PCRegionGrowing(U_Axes, Seeds, r)

    LIA = np.all(np.isin(U_Axes, U_Axes_Good), axis=1)
    U_Axes = U_Axes[LIA]
    IdCdlPts = IdCdlPts[LIA]

    # Compute orientation just to check, should be = 1
    Orientation = np.round(np.mean(np.sign(np.dot(U_Axes,CSs['Y0']))))

    # Assign indices of points on Lateral or Medial Condyles Variable
    if Orientation < 0:
        print('Orientation of Lateral->Medial U_Axes vectors of femoral distal epiphysis is not what expected. Please check manually.')
        med_lat_ind = [1, 0]
    else:
        med_lat_ind = [0, 1]
    
    #  debug plot
    if debug_plots:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(EpiFem['Points'][:,0], EpiFem['Points'][:,1], EpiFem['Points'][:,2], triangles = EpiFem['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.2, shade=False, color = 'yellow')

        for edge in IdCdlPts:
            # [LM] debugging plot - see the kept points
            ax.scatter(EpiFem['Points'][edge[0]][0], EpiFem['Points'][edge[0]][1], EpiFem['Points'][edge[0]][2], color = 'red', s=100)
            ax.scatter(EpiFem['Points'][edge[1]][0], EpiFem['Points'][edge[1]][1], EpiFem['Points'][edge[1]][2], color = 'blue', s=100)
            
            #  [LM] debugging plot (see lines of axes)
            ax.plot([EpiFem['Points'][edge[0]][0], EpiFem['Points'][edge[1]][0]], \
                    [EpiFem['Points'][edge[0]][1], EpiFem['Points'][edge[1]][1]], \
                    [EpiFem['Points'][edge[0]][2], EpiFem['Points'][edge[1]][2]], \
                    color = 'black', linewidth=4, linestyle='solid')

    return IdCdlPts, U_Axes, med_lat_ind

# -----------------------------------------------------------------------------
def GIBOC_femur_getCondyleMostProxPoint(EpiFem, CSs, PtsCondylesTrace, U):

    # Convert tiangulation dict to mesh object --------
    tmp_EpiFem = mesh.Mesh(np.zeros(EpiFem['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(EpiFem['ConnectivityList']):
        for j in range(3):
            tmp_EpiFem.vectors[i][j] = EpiFem['Points'][f[j],:]
    # update normals
    tmp_EpiFem.update_normals()
    # ------------------------------------------------
    sphere_search_radius = 7.5
    plane_search_thick = 2.5
    
    # fitting a lq plane to point in the trace
    # [Centroid,  Direction cosines of the normal to the best-fit plane]
    P_centr, lsplane_norm, _, _ = lsplane(PtsCondylesTrace)
    
    # looking for points 2.5 mm away from the fitting plane
    dMed = np.dot(-P_centr,lsplane_norm)
    IonPlan = np.where((np.abs(np.dot(EpiFem['Points'],lsplane_norm) + dMed) < plane_search_thick) & \
                       (np.dot(EpiFem['Points'],CSs['Z0']) > np.max(np.dot(PtsCondylesTrace,CSs['Z0']) - plane_search_thick)))[0]
    IonPlan = list(np.unique(IonPlan))   
    
    # searches points in a sphere around (7.5 mm)
    # Get the index of points within the spheres
    point_tree = spatial.cKDTree(EpiFem['Points'])
    # This finds the index of all points within distance r of Seeds.
    IonC = point_tree.query_ball_point(PtsCondylesTrace, sphere_search_radius)[0]
    
    # intersect the two sets
    IOK = list(set(IonPlan) & set(IonC))
    IOK = list(np.sort(IOK))
    
    Imax = np.argmax(np.dot(tmp_EpiFem.get_unit_normals()[IOK],U))
    PtTopCondyle = EpiFem['Points'][IOK[Imax]]
    PtTopCondyle = np.reshape(PtTopCondyle,(PtTopCondyle.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    return PtTopCondyle







#%% ---------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
def CS_pelvis_ISB(RASIS, LASIS, RPSIS, LPSIS):
    # defining the ref system (global)
    # with origin at midpoint of ASIS
    # -------------------------------------------------------------------------
    V = np.zeros((3,3))
    
    Z = preprocessing.normalize(RASIS-LASIS, axis=0)
    
    temp_X = ((RASIS+LASIS)/2) - ((RPSIS+LPSIS)/2)
    pseudo_X = preprocessing.normalize(temp_X, axis=0)
    
    Y = np.cross(Z, pseudo_X, axis=0)
    X = np.cross(Y, Z, axis=0)
    
    V[0] = X.T
    V[1] = Y.T
    V[2] = Z.T
    
    return V.T

# -----------------------------------------------------------------------------
def STAPLE_pelvis(Pelvis, side_raw = 'right', result_plots = 1, debug_plots = 0, in_mm = 1):
    # -------------------------------------------------------------------------
    BCS = {}
    JCS = {} 
    PelvisBL = {}
    
    if in_mm == 1:
        dim_fact = 0.001
        
    # get side id correspondent to body side (used for hip joint parent)
    # no need for sign, left and right rf are identical
    _, side_low = bodySide2Sign(side_raw)
    
    print('---------------------')
    print('   STAPLE - PELVIS   ')
    print('---------------------')
    print('* Hip Joint: ' + side_low.upper())
    print('* Method: convex hull')
    print('* Result Plots: ' + ['Off','On'][result_plots])
    print('* Debug  Plots: ' + ['Off','On'][debug_plots])
    print('* Triang Units: mm')
    print('---------------------')
    print('Initializing method...')
    
    # guess of direction of axes on medical images (not always correct)
    # Z : pointing cranially
    # Y : pointing posteriorly
    # X : pointing medio-laterally
    # translating this direction in ISB reference system:
    # ---------------------------------------------------
    
    # inertial axes
    _, CenterVol, InertiaMatrix, D, _ =  TriInertiaPpties(Pelvis)
    
    # Modification of initial guess of CS direction [JB]
    print('Analyzing pelvis geometry...')
    
    RotPseudoISB2Glob, LargestTriangle, _ = pelvis_guess_CS(Pelvis, debug_plots)
    
    # Get the RPSIS and LPSIS raw BoneLandmarks (BL)
    PelvisPseudoISB, _, _ = TriChangeCS(Pelvis, RotPseudoISB2Glob, CenterVol)
    
    # get the bony landmarks
    # Along an axis oriented superiorly and a bit on the right we find
    # projected on this axis succesively RASIS, LASIS then SYMP
    print('Landmarking...')
    
    # project vectors on Z (SYMP is the minimal one)
    I = np.argsort(np.dot(np.abs(LargestTriangle['Points'] - CenterVol.T), RotPseudoISB2Glob[2]))
    ASIS_inds = I[1:]
    ind_RASIS = np.where(np.dot(LargestTriangle['Points'][ASIS_inds] - CenterVol.T, RotPseudoISB2Glob[2]) > 0)
    ind_LASIS = np.where(np.dot(LargestTriangle['Points'][ASIS_inds] - CenterVol.T, RotPseudoISB2Glob[2]) < 0)
    
    SYMP = LargestTriangle['Points'][I[0]]
    SYMP = np.reshape(SYMP,(SYMP.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    RASIS = LargestTriangle['Points'][ASIS_inds[ind_RASIS]].T
    LASIS = LargestTriangle['Points'][ASIS_inds[ind_LASIS]].T
    
    # Get the Posterior, Superior, Right eigth of the pelvis
    Nodes_RPSIS = list(np.where((PelvisPseudoISB['Points'][:,0] < 0) & \
                           (PelvisPseudoISB['Points'][:,1] > 0) & \
                           (PelvisPseudoISB['Points'][:,2] > 0))[0])
    
    Pelvis_RPSIS = TriReduceMesh(Pelvis, [], Nodes_RPSIS)
    
    # Find the most posterior points in this eigth
    Imin = np.argmin(np.dot(Pelvis_RPSIS['Points'], RotPseudoISB2Glob[0]))
    RPSIS = Pelvis_RPSIS['Points'][Imin]
    RPSIS = np.reshape(RPSIS,(RPSIS.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    # Get the Posterior, Superior, Left eigth of the pelvis
    Nodes_LPSIS = list(np.where((PelvisPseudoISB['Points'][:,0] < 0) & \
                           (PelvisPseudoISB['Points'][:,1] > 0) & \
                           (PelvisPseudoISB['Points'][:,2] < 0))[0])
    
    Pelvis_LPSIS = TriReduceMesh(Pelvis, [], Nodes_LPSIS)
    
    # Find the most posterior points in this eigth
    Imin = np.argmin(np.dot(Pelvis_LPSIS['Points'], RotPseudoISB2Glob[0]))
    LPSIS = Pelvis_LPSIS['Points'][Imin]
    LPSIS = np.reshape(LPSIS,(LPSIS.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    # check if bone landmarks are correctly identified or axes were incorrect
    if np.linalg.norm(RASIS-LASIS) < np.linalg.norm(RPSIS-LPSIS):
        # inform user
        print('GIBOK_pelvis.')
        print('Inter-ASIS distance is shorter than inter-PSIS distance. Better check manually.')
        
    # ISB reference system
    PelvisOr = (RASIS+LASIS)/2.0
    
    # segment reference system
    BCS['CenterVol'] = CenterVol
    BCS['Origin'] = PelvisOr
    BCS['InertiaMatrix'] = InertiaMatrix
    BCS['V'] = CS_pelvis_ISB(RASIS, LASIS, RPSIS, LPSIS)
    
    # storing joint details
    JCS['ground_pelvis'] = {}
    JCS['ground_pelvis']['V'] = BCS['V']
    JCS['ground_pelvis']['Origin'] = PelvisOr
    JCS['ground_pelvis']['child_location'] = PelvisOr.T*dim_fact # [1x3] as in OpenSim
    JCS['ground_pelvis']['child_orientation'] = computeXYZAngleSeq(BCS['V']) # [1x3] as in OpenSim
    
    # define hip parent
    hip_name = 'hip_' + side_low
    JCS[hip_name] = {}
    JCS[hip_name]['parent_orientation'] = computeXYZAngleSeq(BCS['V'])
    
    # Export bone landmarks: [3x1] vectors
    PelvisBL['RASIS'] = RASIS
    PelvisBL['LASIS'] = LASIS
    PelvisBL['RPSIS'] = RPSIS
    PelvisBL['LPSIS'] = LPSIS
    PelvisBL['SYMP'] = SYMP
    
    # debug plot
    if result_plots:
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection = '3d')
        
        ax1.set_title('STAPLE | bone: pelvis | side: ' + side_low)
        
        plotTriangLight(Pelvis, BCS, ax1)
        quickPlotRefSystem(BCS, ax1)
        quickPlotRefSystem(JCS['ground_pelvis'], ax1)
        ax1.plot_trisurf(LargestTriangle['Points'][:,0], LargestTriangle['Points'][:,1], LargestTriangle['Points'][:,2], \
                         triangles = LargestTriangle['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.4, color = 'y', shade=False)
        
        # plot markers and labels
        plotBoneLandmarks(PelvisBL, ax1, 1)
        
    
    return BCS, JCS, PelvisBL

# -----------------------------------------------------------------------------
def GIBOC_femur(femurTri, side_raw = 'right', fit_method = 'cylinder', result_plots = 1, debug_plots = 0, in_mm = 1):
    # Automatically define a reference system based on bone morphology by 
    # fitting analytical shapes to the articular surfaces. It is normally used 
    # within processTriGeomBoneSet.m in workflows to generate musculoskeletal 
    # models. The GIBOC algorithm can also extract articular surfaces.
    # 
    # Inputs :
    # femurTri - Dict of triangulation object representing a femur.
    # 
    # side_raw - a string indicating the body side. Valid options: 'R', 'r'  
    # for the right side, 'L' and 'l' for the left side.
    # 
    # fit_method - a string indicating the name to assign to the OpenSim body.
    # 
    # result_plots - 
    # 
    # debug_plots - enable plots used in debugging. Value: 1 or 0 (default).
    # 
    # in_mm - (optional) indicates if the provided geometries are given in mm
    # (value: 1) or m (value: 0). Please note that all tests and analyses
    # done so far were performed on geometries expressed in mm, so this
    # option is more a placeholder for future adjustments.
    #
    # Output :
    # CS -
    # 
    # JCS - 
    # 
    # FemurBL - Dict containing the bony landmarks identified 
    # on the bone geometries based on the defined reference systems. Each
    # field is named like a landmark and contain its 3D coordinates.
    # -------------------------------------------------------------------------
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1
    
    # default algorithm: cylinder fitting (Modenese et al. JBiomech 2018)
    if fit_method == '':
        fit_method = 'cylinder'
    
    # get side id correspondent to body side 
    _, side_low = bodySide2Sign(side_raw)
    
    # inform user about settings
    print('---------------------')
    print('   GIBOC - FEMUR   ')
    print('---------------------')
    print('* Body Side: ' + side_low.upper())
    print('* Fit Method: ' + fit_method)
    print('* Result Plots: ' + ['Off','On'][result_plots])
    print('* Debug  Plots: ' + ['Off','On'][debug_plots])
    print('* Triang Units: mm')
    print('---------------------')
    print('Initializing method...')
    
    # it is assumed that, even for partial geometries, the femoral bone is
    # always provided as unique file. Previous versions of this function did
    # use separated proximal and distal triangulations. Check Git history if
    # you are interested in that.
    U_DistToProx = femur_guess_CS(femurTri, debug_plots)
    ProxFemTri, DistFemTri = cutLongBoneMesh(femurTri, U_DistToProx)
    
    # Compute the coefficient for morphology operations
    CoeffMorpho = computeTriCoeffMorpho(femurTri)
    
    # Get inertial principal vectors V_all of the femur geometry & volum center
    V_all, CenterVol, InertiaMatrix, _, _ = TriInertiaPpties(femurTri)
    
    # -------------------------------------
    # Initial Coordinate system (from inertial axes and femoral head):
    # * Z0: points upwards (inertial axis) 
    # * Y0: points medio-lat 
    # -------------------------------------
    # coordinate system structure to store coordinate system's info
    AuxCSInfo = {}
    AuxCSInfo['CenterVol'] = CenterVol
    AuxCSInfo['V_all'] = V_all
    
    # Check that the distal femur is 'below' the proximal femur or invert Z0
    Z0 = V_all[0]
    Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    Z0 *= np.sign(np.dot((np.mean(ProxFemTri['Points'], axis=0) - np.mean(DistFemTri['Points'], axis=0)),Z0))
    AuxCSInfo['Z0'] = Z0
    
    # Find Femoral Head Center
    try:
        # sometimes Renault2018 fails for sparse meshes 
        # FemHeadAS is the articular surface of the hip
        AuxCSInfo, FemHeadTri = GIBOC_femur_fitSphere2FemHead(ProxFemTri, AuxCSInfo, CoeffMorpho, debug_plots)
    except:
        # use Kai if GIBOC approach fails
        logging.warning('Renault2018 fitting has failed. Using Kai femoral head fitting. \n')
        AuxCSInfo, _ = Kai2014_femur_fitSphere2FemHead(ProxFemTri, AuxCSInfo, debug_plots)
        AuxCSInfo['CenterFH_Renault'] = AuxCSInfo['CenterFH_Kai']
        AuxCSInfo['RadiusFH_Renault'] = AuxCSInfo['RadiusFH_Kai']
    
    # X0 points backwards
    
    
    
    
    
    
    
    
    
    
    return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    