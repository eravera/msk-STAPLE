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
                                    sphere_fit 

from opensim_tools import computeXYZAngleSeq

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
    CSs = {} 
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
    
    # triang around it
    Face_Top_FH = TriReduceMesh(ProxFem, I_Top_FH)
    
    # create a triang with them
    Patch_Top_FH = TriDilateMesh(ProxFem, Face_Top_FH, 40*CoeffMorpho)
    
    # Get an initial ML Axis Y0 (pointing medio-laterally)
    # NB: from centerVol, OT points upwards to ~HJC, that is more medial than
    # Z0, hence cross(CSs.Z0,OT) points anteriorly and Y0 medially
    OT = np.mean(Patch_Top_FH['Points'], axis=0).T - CSs['CenterVol']
    tmp_Y0 = np.cross(np.cross(CSs['Z0'], OT), CSs['Z0'])
    CSs['Y0'] = np.linalg.norm(tmp_Y0)
    
    # Find a the most medial (MM) point on the femoral head (FH)
    I_MM_FH = []
    I_MM_FH.append(np.argmax(np.dot(tmp_ProxFem.centroids, CSs['Y0'])))
    
    # most prox point (neighbors of I_Top_FH)
    for nei in ProxFem['ConnectivityList'][I_MM_FH].reshape(-1, 1):
        I_MM_FH += list(list(np.where(ProxFem['ConnectivityList'] == nei))[0])
    
    # triang around it
    Face_MM_FH = TriReduceMesh(ProxFem, I_MM_FH)
    
    # create a triang with them
    Patch_MM_FH = TriDilateMesh(ProxFem, Face_MM_FH, 40*CoeffMorpho)
    
    # STEP1: first sphere fit
    FemHead0 = TriUnite(Patch_MM_FH,Patch_Top_FH)
    
    Centre, Radius, ErrorDist = sphere_fit(FemHead0['Points'])
    sph_RMSE = np.mean(abs(ErrorDist))

    # print
    print('     Fit #1: RMSE: ' + str(sph_RMSE) + ' mm')

    if debug_prints:
        print('----------------')
        print('First Estimation')
        print('----------------')
        print('Centre: ' + str(Centre))
        print('Radius: ' + str(Radius))
        print('Mean Res: ' + str(sph_RMSE))
        print('-----------------')
    
    # STEP2: dilate femoral head mesh and sphere fit again
    # IMPORTANT: TriDilateMesh "grows" the original mesh, does not create a larger one!
    FemHead_dil_coeff = 1.5

    DilateFemHeadTri = TriDilateMesh(ProxFem, FemHead0, round(FemHead_dil_coeff*Radius*CoeffMorpho))
    CenterFH, RadiusDil, ErrorDistCond = sphere_fit(DilateFemHeadTri['Points'])

    sph_RMSECond = np.mean(abs(ErrorDistCond))

    # print
    print('     Fit #2: RMSE: ' + str(sph_RMSECond) + ' mm');

    if debug_prints:
        print('----------------')
        print('First Estimation')
        print('----------------')
        print('Centre: ' + str(CenterFH))
        print('Radius: ' + str(RadiusDil))
        print('Mean Res: ' + str(sph_RMSECond))
        print('-----------------')
    
    # check
    if ~(RadiusDil > Radius):
        logging.exception('Dilated femoral head smaller than original mesh. Please check manually.')
    
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
    Cond1 = [1 if (np.dot(val, TriDilateFemHead.get_unit_normals()[pos])) > \
             FemHead_normals_thresh else 0 for pos, val in enumerate(normal_CPts_PF_2D)]

    # COND2: Delete points far from sphere surface outside [90%*Radius 110%*Radius]
    Cond2 = [1 if abs(np.sqrt(np.dot(val, CPts_PF_2D[pos])) - Radius) < \
             0.1*Radius else 0 for pos, val in enumerate(CPts_PF_2D)]

    # [LM] I have found both conditions do not work always, when combined
    # check if using both conditions produces results
    single_cond = 0
    min_number_of_points = 20

    if np.sum(np.array(Cond1 + Cond2)) > min_number_of_points:
        applied_Cond = list(np.logical_and(Cond1, Cond2))
    else:
        # flag that only one condition is used
        single_cond = 1
        cond1_count = np.sum(np.array(Cond1))
        applied_Cond = Cond1

    # search within conditions Cond1 and Cond2
    Face_ID_PF_2D_onSphere = np.where(applied_Cond)[0]

    # get the mesh and points on the femoral head 
    FemHead = TriReduceMesh(DilateFemHeadTri, Face_ID_PF_2D_onSphere)

    # if just one condition is active JB suggests to keep largest patch
    # if single_cond:
        # FemHead = TriKeepLargestPatch(FemHead)
    
    
    
    
    
    
    return CSs, FemHead
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
    # try:
    #     # sometimes Renault2018 fails for sparse meshes 
    #     # FemHeadAS is the articular surface of the hip
    #     # AuxCSInfo, FemHeadTri = GIBOC_femur_fitSphere2FemHead(ProxFemTri, AuxCSInfo, CoeffMorpho, debug_plots)
    # except:
    #     # use Kai if GIBOC approach fails
    #     logging.exception('Renault2018 fitting has failed. Using Kai femoral head fitting. \n')
    #     # AuxCSInfo, _ = Kai2014_femur_fitSphere2FemHead(ProxFemTri, AuxCSInfo, debug_plots)
    
    return 0
    
    
    
    
    
    
    
    
    
    
    
    
    
    