#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AUTHOR(S) AND VERSION-HISTORY

Author:   Luca Modenese & Jean-Baptiste Renault. 
Copyright 2020 Luca Modenese & Jean-Baptiste Renault

Pyhton Version:

Coding by Emiliano Ravera, Sep 2024. Adapted to Python 3.11
email:    emiliano.ravera@uner.edu.ar

@author: eravera
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
from scipy.optimize import curve_fit
from scipy.optimize import nnls
from scipy.optimize import leastsq
from scipy.spatial.transform import Rotation as R

from sklearn import preprocessing
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import path as mpl_path

from pykdtree.kdtree import KDTree

from ellipse import LsqEllipse

import fast_simplification

import opensim

from Public_functions import *

# ----------- end import packages --------------

#%%
# ##################################################
# ALGORITHMS #######################################
# ##################################################
# 
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
    
    debug_plots = 0
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
    
    Imax = np.argmax(np.dot(tmp_EpiFem.get_unit_normals()[IOK], U))
    PtTopCondyle = EpiFem['Points'][IOK[Imax]]
    PtTopCondyle = np.reshape(PtTopCondyle,(PtTopCondyle.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    return PtTopCondyle

# -----------------------------------------------------------------------------
def GIBOC_femur_smoothCondyles(EpiFem, PtsFullCondyle, CoeffMorpho):
    # -------------------------------------------------------------------------
    Condyle = {}
    Condyle2 = {}
    fullCondyle = {}
    
    Condyle = TriReduceMesh(EpiFem, [], PtsFullCondyle)
    Condyle2 = TriCloseMesh(EpiFem, Condyle, 5*CoeffMorpho)
    
    fullCondyle = TriOpenMesh(EpiFem, Condyle2, 15*CoeffMorpho)
    
    return fullCondyle

# -----------------------------------------------------------------------------
def GIBOC_femur_filterCondyleSurf(EpiFem, CSs, PtsCondyle, Pts_0_C, CoeffMorpho):
    # -------------------------------------------------------------------------
    # NOTE in the original toolbox lateral and medial condyles were processed 
    # differently in the last step. I assume was a typo, given the entire code
    # of this function was duplicated. I expect the difference to be small
    # between the two versions.
    # -------------------------------------------------------------------------
    
    Y1 = CSs['Y1']
    
    Center, _, _ = sphere_fit(PtsCondyle)
    Condyle = TriReduceMesh(EpiFem, [], PtsCondyle)
    Condyle = TriCloseMesh(EpiFem, Condyle, 4*CoeffMorpho)
        
    # Get Curvature
    Cmean, Cgaussian, _, _, _, _ = TriCurvature(Condyle, False)
    
    # Compute a Curvtr norm
    Curvtr = np.sqrt(4*(Cmean)**2 - 2*Cgaussian)
    
    # Calculate the "probability" of a vertex to be on an edge, depends on :
    # - Difference in normal orientation from fitted cylinder
    # - Curvature Intensity
    # - Orientation relative to Distal Proximal axis
    CylPts = Condyle['Points'] - Center
    Ui = CylPts - np.dot((np.dot(CylPts,Y1)), Y1.T)
    Ui = preprocessing.normalize(Ui, axis=1)
    
    # Calculate vertices normals
    Condyle = TriVertexNormal(Condyle)
    AlphaAngle = np.abs(90 - (np.arccos(np.sum(Condyle['vertexNormal']*Ui, axis=1)))*180/np.pi)
    GammaAngle = (np.arccos(np.dot(Condyle['vertexNormal'], CSs['Z0'])[:,0]))*180/np.pi
    
    # Sigmoids functions to compute probability of vertex to be on an edge
    Prob_Edge_Angle = 1/(1 + np.exp((AlphaAngle-50)/10))
    Prob_Edge_Angle /= np.max(Prob_Edge_Angle)
    
    Prob_Edge_Curv = 1/(1 + np.exp(-((Curvtr-0.25)/0.05)))
    Prob_Edge_Curv /= np.max(Prob_Edge_Curv)
    
    Prob_FaceUp = 1/(1 + np.exp((GammaAngle-45)/15))
    Prob_FaceUp /= np.max(Prob_FaceUp)
    
    Prob_Edge = 0.6*np.sqrt(Prob_Edge_Angle*Prob_Edge_Curv) + \
        0.05*Prob_Edge_Curv + 0.15*Prob_Edge_Angle + 0.2*Prob_FaceUp
    
    Condyle_edges = TriReduceMesh(Condyle, [], list(np.where(Prob_Edge_Curv*Prob_Edge_Angle > 0.5)[0]))
    Condyle_end = TriReduceMesh(Condyle, [], list(np.where(Prob_Edge < 0.2)[0]))
    Condyle_end = TriConnectedPatch(Condyle_end, Pts_0_C)
    Condyle_end = TriCloseMesh(EpiFem, Condyle_end, 10*CoeffMorpho)
    
    # medial condyle (in original script)
    Condyle_end = TriKeepLargestPatch(Condyle_end)
    Condyle_end = TriDifferenceMesh(Condyle_end , Condyle_edges)
    
    return Condyle_end

# -----------------------------------------------------------------------------
def GIBOC_femur_ArticSurf(EpiFem, CSs, CoeffMorpho, art_surface, debug_plots=0):
    # -------------------------------------------------------------------------
    CSs = CSs.copy()
    V_all = CSs['V_all'] 
    Z0 = CSs['Z0']

    if art_surface == 'full_condyles':
        # Identify full articular surface of condyles (points)
        # PARAMETERS
        CutAngle_Lat = 70
        CutAngle_Med = 85
        InSetRatio = 0.8
        ellip_dilat_fact = 0.025
    elif art_surface == 'post_condyles':
        # Identify posterior part of condyles (points)
        # PARAMETERS
        CutAngle_Lat = 10
        CutAngle_Med = 25
        InSetRatio = 0.6
        ellip_dilat_fact = 0.025
    elif art_surface == 'pat_groove':
        # ame as posterior
        CutAngle_Lat = 10
        CutAngle_Med = 25
        InSetRatio = 0.6
        ellip_dilat_fact = 0.025
        
    # Analyze epiphysis to traces of condyles (lines running on them - plots)
    # extracts:
    # * indices of points on condyles (lines running on them)
    # * well oriented M-L axes joining these points
    # * med_lat_ind: indices [1,2] or [2, 1]. 1st comp is medial cond, 2nd lateral.
    # ============
    # PARAMETERS
    # ============
    edge_threshold = 0.5 # used also for new coord syst below
    axes_dev_thresh = 0.75

    IdCdlPts, U_Axes, med_lat_ind = GIBOC_femur_processEpiPhysis(EpiFem, CSs, V_all, edge_threshold, axes_dev_thresh)

    # ============
    # Assign indices of points on Lateral or Medial Condyles Variable
    # These are points, almost lines that "walk" on the condyles
    PtsCondylesMed = EpiFem['Points'][IdCdlPts[:,med_lat_ind[0]]]
    PtsCondylesLat = EpiFem['Points'][IdCdlPts[:,med_lat_ind[1]]]

    # debugging plots: plotting the lines between the points identified
    #  debug plot
    debug_plots = 0
    if debug_plots:
        
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(EpiFem['Points'][:,0], EpiFem['Points'][:,1], EpiFem['Points'][:,2], triangles = EpiFem['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'yellow')
        
        ax.scatter(PtsCondylesMed[:,0], PtsCondylesMed[:,1], PtsCondylesMed[:,2], color = 'green', s=200)
        ax.scatter(PtsCondylesLat[:,0], PtsCondylesLat[:,1], PtsCondylesLat[:,2], color = 'black', s=200)
        
        for i in range(len(PtsCondylesMed)):
            ax.plot([PtsCondylesMed[i,0], PtsCondylesLat[i,0]], \
                    [PtsCondylesMed[i,1], PtsCondylesLat[i,1]], \
                    [PtsCondylesMed[i,2], PtsCondylesLat[i,2]], \
                    color = 'black', linewidth=4)

    # New temporary coordinate system (new ML axis guess)
    # The reference system:
    # -------------------------------------
    # Y1: based on U_Axes (MED-LAT??)
    # X1: cross(Y1, Z0), with Z0 being the upwards inertial axis
    # Z1: cross product of prev
    # -------------------------------------
    Y1 = np.sum(U_Axes, axis=0)
    Y1 = np.reshape(Y1,(Y1.size, 1)) # convert 1d (3,) to 2d (3,1) vector 
    Y1 = preprocessing.normalize(Y1, axis=0)
    X1 = preprocessing.normalize(np.cross(Y1.T,Z0.T), axis=1).T
    Z1 = np.cross(X1.T, Y1.T).T
    # VC = np.array([X1[:,0], Y1[:,0], Z1[:,0]])
    VC = np.zeros((3,3))
    VC[:,0] = X1[:,0]
    VC[:,1] = Y1[:,0]
    VC[:,2] = Z1[:,0]
    # The intercondyle distance being larger posteriorly the mean center of
    # 50% longest edges connecting the condyles is located posteriorly.
    # NB edge threshold is customizable

    n_in = int(np.ceil(len(IdCdlPts)*edge_threshold))
    MidPtPosterCondyle = np.mean(0.5*(PtsCondylesMed[:n_in] + PtsCondylesLat[:n_in]), axis=0)

    # centroid of all points in the epiphysis
    MidPtEpiFem = np.mean(EpiFem['Points'], axis=0)

    # Select Post Condyle points :
    # Med & Lat Points is the most distal-Posterior on the condyles
    X1 = np.sign(np.dot((MidPtEpiFem - MidPtPosterCondyle), X1))*X1
    U =  preprocessing.normalize(3*Z0 - X1, axis=0)

    # Add ONE point (top one) on each proximal edges of each condyle that might
    # have been excluded from the initial selection
    PtMedTopCondyle = GIBOC_femur_getCondyleMostProxPoint(EpiFem, CSs, PtsCondylesMed, U)
    PtLatTopCondyle = GIBOC_femur_getCondyleMostProxPoint(EpiFem, CSs, PtsCondylesLat, U)

    # # [LM] plotting for debugging
    # if debug_plots:
    #     plot3(PtMedTopCondyle(:,1), PtMedTopCondyle(:,2), PtMedTopCondyle(:,3),'go');
    #     plot3(PtLatTopCondyle(:,1), PtLatTopCondyle(:,2), PtLatTopCondyle(:,3),'go');

    # Separate medial and lateral condyles points
    # The middle point of all edges connecting the condyles is
    # located distally :
    PtMiddleCondyle = np.mean(0.5*(PtsCondylesMed + PtsCondylesLat), axis = 0)
    PtMiddleCondyle = np.reshape(PtMiddleCondyle,(PtMiddleCondyle.size, 1)) # convert 1d (3,) to 2d (3,1) vector 

    # transformations on the new refernce system: x_n = (R*x')'=x*R' [TO CHECK]
    Pt_AxisOnSurf_proj = np.dot(PtMiddleCondyle.T,VC).T # middle point
    # Pt_AxisOnSurf_proj = np.dot(VC, PtMiddleCondyle) # middle point
    Epiphysis_Pts_DF_2D_RC  = np.dot(EpiFem['Points'],VC) # distal femur

    # THESE TRANSFORMATION ARE INVERSE [LM]
    # ============================
    row = len(PtsCondylesLat) + 2
    Pts_Proj_CLat  = np.zeros((row,3))
    Pts_Proj_CLat[0:row-2] = PtsCondylesLat
    Pts_Proj_CLat[-2] = PtLatTopCondyle.T
    Pts_Proj_CLat[-1] = PtLatTopCondyle.T
    Pts_Proj_CLat =  np.dot(Pts_Proj_CLat,VC)

    row = len(PtsCondylesMed) + 2
    Pts_Proj_CMed  = np.zeros((row,3))
    Pts_Proj_CMed[0:row-2] = PtsCondylesMed
    Pts_Proj_CMed[-2] = PtMedTopCondyle.T
    Pts_Proj_CMed[-1] = PtMedTopCondyle.T
    Pts_Proj_CMed =  np.dot(Pts_Proj_CMed,VC)

    Pts_0_C1 = np.dot(Pts_Proj_CLat,VC.T)
    Pts_0_C2 = np.dot(Pts_Proj_CMed,VC.T)
    # ============================

    # divides the epiphysis in med and lat based on where they stand wrt the
    # midpoint identified above
    C1_Pts_DF_2D_RC = Epiphysis_Pts_DF_2D_RC[Epiphysis_Pts_DF_2D_RC[:,1] - Pt_AxisOnSurf_proj[1] < 0]
    C2_Pts_DF_2D_RC = Epiphysis_Pts_DF_2D_RC[Epiphysis_Pts_DF_2D_RC[:,1] - Pt_AxisOnSurf_proj[1] > 0]

    # Identify full articular surface of condyles (points) by fitting an ellipse 
    # on long convexhull edges extremities
    ArticularSurface_Lat, _ = PtsOnCondylesFemur(Pts_Proj_CLat, C1_Pts_DF_2D_RC, \
                                                 CutAngle_Lat, InSetRatio, ellip_dilat_fact)
    ArticularSurface_Lat = ArticularSurface_Lat @ VC.T
    ArticularSurface_Med, _ = PtsOnCondylesFemur(Pts_Proj_CMed, C2_Pts_DF_2D_RC, \
                                                 CutAngle_Med, InSetRatio, ellip_dilat_fact)
    ArticularSurface_Med = ArticularSurface_Med @ VC.T

    # locate notch using updated estimation of X1
    MidPtPosterCondyleIt2 = np.mean(np.concatenate([ArticularSurface_Lat, ArticularSurface_Med]), axis = 0)
    MidPtPosterCondyleIt2 = np.reshape(MidPtPosterCondyleIt2,(1, MidPtPosterCondyleIt2.size)) # convert 1d (3,) to 2d (3,1) vector 

    X1 = np.sign(np.dot((MidPtEpiFem - MidPtPosterCondyleIt2), X1))*X1
    U =  preprocessing.normalize(-Z0 - 3*X1, axis=0)

    # compute vertex normal
    EpiFem = TriVertexNormal(EpiFem)

    NodesOk = EpiFem['Points'][(np.dot(EpiFem['vertexNormal'], U) > 0.98)[:,0]]

    U = preprocessing.normalize(Z0 - 3*X1, axis=0)
    IMax = np.argmin(np.dot(NodesOk,U))
    PtNotch = NodesOk[IMax]
    PtNotch = np.reshape(PtNotch,(PtNotch.size,1))

    # store geometrical elements useful externally
    CSs['BL'] = {}
    CSs['BL']['PtNotch'] = PtNotch

    # stored for use in functions (cylinder ref system)
    CSs['X1'] = X1
    CSs['Y1'] = Y1 # axis guess for cyl ref system
    CSs['Z1'] = Z1

    art_surface = 'full_condyles'

    if art_surface == 'full_condyles':
        # if output is full condyles then just filter and create triang
        DesiredArtSurfLat_Tri = GIBOC_femur_smoothCondyles(EpiFem, ArticularSurface_Lat, CoeffMorpho)
        DesiredArtSurfMed_Tri = GIBOC_femur_smoothCondyles(EpiFem, ArticularSurface_Med, CoeffMorpho)
        
    elif art_surface == 'post_condyles':
        # Delete points that are anterior to Notch
        ArticularSurface_Lat = ArticularSurface_Lat[(np.dot(ArticularSurface_Lat, X1) <= np.dot(PtNotch.T, X1))[:,0]]
        ArticularSurface_Med = ArticularSurface_Med[(np.dot(ArticularSurface_Med, X1) <= np.dot(PtNotch.T, X1))[:,0]]
        Pts_0_C1 = Pts_0_C1[(np.dot(Pts_0_C1, X1) <= np.dot(PtNotch.T, X1))[:,0]]
        Pts_0_C2 = Pts_0_C2[(np.dot(Pts_0_C2, X1) <= np.dot(PtNotch.T, X1))[:,0]]
        
        # Filter with curvature and normal orientation to keep only the post parts 
        # these are triangulations
        DesiredArtSurfLat_Tri = GIBOC_femur_filterCondyleSurf(EpiFem, CSs, ArticularSurface_Lat, Pts_0_C1, CoeffMorpho)
        DesiredArtSurfMed_Tri = GIBOC_femur_filterCondyleSurf(EpiFem, CSs, ArticularSurface_Med, Pts_0_C2, CoeffMorpho)

    elif art_surface == 'pat_groove':
        # Generating patellar groove triangulations (med and lat) initial 
        # estimations of anterior patellar groove (anterior to mid point) (points)
        ant_lat = C1_Pts_DF_2D_RC[C1_Pts_DF_2D_RC[:,0] - Pt_AxisOnSurf_proj[0] > 0]
        ant_lat = ant_lat @ VC.T
        ant_med = C2_Pts_DF_2D_RC[C2_Pts_DF_2D_RC[:,0] - Pt_AxisOnSurf_proj[0] > 0]
        ant_med = ant_med @ VC.T
        # anterior to notch (points)
        PtsGroove_Lat = ant_lat[(np.dot(ant_lat, X1) > np.dot(PtNotch.T, X1))[:,0]]
        PtsGroove_Med = ant_med[(np.dot(ant_med, X1) > np.dot(PtNotch.T, X1))[:,0]]
        # triangulations of medial and lateral patellar groove surfaces
        DesiredArtSurfLat_Tri = GIBOC_femur_filterCondyleSurf(EpiFem, CSs, PtsGroove_Lat, Pts_0_C1, CoeffMorpho)
        DesiredArtSurfMed_Tri = GIBOC_femur_filterCondyleSurf(EpiFem, CSs, PtsGroove_Med, Pts_0_C1, CoeffMorpho)
        

    return DesiredArtSurfMed_Tri, DesiredArtSurfLat_Tri, CSs

# -----------------------------------------------------------------------------
def tibia_guess_CS(Tibia = {}, debug_plots = 1):
    # -------------------------------------------------------------------------
    # Function to test putting back together a correct orientation of the femur
    # Inputs :
    # Tibia : A triangulation of a complete tibia
    # debug_plots : A boolean to display plots useful for debugging
    # 
    # Output :
    # Z0 : A unit vector giving the distal to proximal direction
    # -------------------------------------------------------------------------
    #                           General Idea
    # The largest cross section along the principal inertia axis is located at
    # the tibial plateau. From that information it's easy to determine the
    # distal to proximal direction.
    # -------------------------------------------------------------------------
    # Get principal inertia axis
    # Get the principal inertia axis of the tibia (potentially wrongly orientated)
    Z0 = np.zeros((3,1))
    V_all, CenterVol, _, _, _ = TriInertiaPpties(Tibia)
    Z0 = V_all[0]
    Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector

    # Get CSA
    cut_offset = 0.5
    min_coord = np.min(np.dot(Tibia['Points'], Z0)) + cut_offset
    max_coord = np.max(np.dot(Tibia['Points'], Z0)) - cut_offset

    Alt = np.linspace(min_coord, max_coord, 100)

    Curves = {}
    Areas = []
    Centroids = {}

    for it, d in enumerate(-Alt):

        Curves[str(it)], _, _ = TriPlanIntersect(Tibia, Z0, d)
        max_area = 0
        for key in Curves[str(it)].keys():
            Centroids[str(it)], area_j = PlanPolygonCentroid3D(Curves[str(it)][key]['Pts'])
            if area_j > max_area:
                max_area = area_j
        Areas.append(max_area)
        
    i_max_Area = np.argmax(Areas)

    if i_max_Area > 0.66*len(Alt):
        Z0 *= 1
    elif i_max_Area < 0.33*len(Alt):
        Z0 *= -1
    else:
        # loggin.warning('Identification of the initial distal to proximal axis of the tibia went wrong. Check the tibia geometry')
        print('Identification of the initial distal to proximal axis of the tibia went wrong. Check the tibia geometry')
 
    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        plotDot(Centroids[str(i_max_Area)], ax, 'r', 3)
        # plot Z0 vector
        ax.quiver(CenterVol[0], CenterVol[1], CenterVol[2], \
                  Z0[0], Z0[1], Z0[2], \
                  color='b', length = 220)
        ax.plot_trisurf(Tibia['Points'][:,0], Tibia['Points'][:,1], Tibia['Points'][:,2], triangles = Tibia['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.2, shade=False, color = 'cyan')
        ax.set_box_aspect([1,3,1])
    
    return Z0

# -----------------------------------------------------------------------------
def tibia_identify_lateral_direction(DistTib = [], Z0 = np.zeros((3,1))):
    # -------------------------------------------------------------------------
    # slice at centroid of distal tibia
    _, CenterVolTibDist, _, _, _ = TriInertiaPpties(DistTib)
    d = 0.9*np.dot(CenterVolTibDist.T, Z0)
    DistCurves , _, _ = TriPlanIntersect(DistTib, Z0, -d)
    
    # check the number of curves on that slice
    N_DistCurves = len(DistCurves)
    just_tibia = True
    
    if N_DistCurves == 2:
        print('Tibia and fibula have been detected.')
        just_tibia = False
    elif N_DistCurves > 2:
        # loggin.warning('There are ' + str(N_DistCurves) + ' section areas.')
        # loggin.error('This should not be the case (only tibia and possibly fibula should be there).')
        print('There are ' + str(N_DistCurves) + ' section areas.')
        print('This should not be the case (only tibia and possibly fibula should be there).')
    
    # Find the most distal point, it will be medial
    # even if not used when tibia and fibula are available it is used in
    # plotting
    I_dist_fib = np.argmax(np.dot(DistTib['Points'], -Z0))
    MostDistalMedialPt = DistTib['Points'][I_dist_fib]
    MostDistalMedialPt = np.reshape(MostDistalMedialPt,(MostDistalMedialPt.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    # compute a vector pointing from the most distal point (medial) to the
    # centre of distal part of tibia. Points laterally for any leg side, so 
    # it is Z_ISB for right and -Z_ISB for left side.
    
    if just_tibia:
        # vector pointing laterally
        U_tmp = (CenterVolTibDist - MostDistalMedialPt)
    else:
        # tibia and fibula
        # check which area is larger (Tibia)
        if DistCurves['1']['Area'] > DistCurves['2']['Area']:
            # vector from tibia section to fibular section (same considerations
            # as just_tibia = True using the centroid.
            U_tmp = np.mean(DistCurves['2']['Pts'], axis = 0) - np.mean(DistCurves['1']['Pts'], axis = 0)
            U_tmp = np.reshape(U_tmp,(U_tmp.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        else:
            U_tmp = np.mean(DistCurves['1']['Pts'], axis = 0) - np.mean(DistCurves['2']['Pts'], axis = 0)
            U_tmp = np.reshape(U_tmp,(U_tmp.size, 1)) # convert 1d (3,) to 2d (3,1) vector

    return U_tmp, MostDistalMedialPt, just_tibia












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
        print('GIBOC_pelvis.')
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
    JCS['ground_pelvis']['child_location'] = PelvisOr*dim_fact#PelvisOr.T*dim_fact # [1x3] as in OpenSim
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
def GIBOC_femur(femurTri, side_raw = 'r', fit_method = 'cylinder', result_plots = 1, debug_plots = 0, in_mm = 1):
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
    AuxCSInfo['X0'] = np.cross(AuxCSInfo['Y0'].T, AuxCSInfo['Z0'].T).T

    # # Isolates the epiphysis
    EpiFemTri = GIBOC_isolate_epiphysis(DistFemTri, Z0, 'distal')

    # extract full femoral condyles
    print('Extracting femoral condyles articular surfaces...')

    fullCondyle_Med_Tri, fullCondyle_Lat_Tri, AuxCSInfo = GIBOC_femur_ArticSurf(EpiFemTri, AuxCSInfo, CoeffMorpho, 'full_condyles', debug_plots)

    # plot condyles to ensure medial and lateral sides are correct and surfaces are ok
    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(femurTri['Points'][:,0], femurTri['Points'][:,1], femurTri['Points'][:,2], triangles = femurTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'yellow')
        ax.plot_trisurf(fullCondyle_Lat_Tri['Points'][:,0], fullCondyle_Lat_Tri['Points'][:,1], fullCondyle_Lat_Tri['Points'][:,2], triangles = fullCondyle_Lat_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'blue')
        ax.plot_trisurf(fullCondyle_Med_Tri['Points'][:,0], fullCondyle_Med_Tri['Points'][:,1], fullCondyle_Med_Tri['Points'][:,2], triangles = fullCondyle_Med_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'red')
        ax.set_title('Full Condyles (red: medial)')
        
    # extract posterior part of condyles (points) by fitting an ellipse 
    # on long convexhull edges extremities
    postCondyle_Med_Tri, postCondyle_Lat_Tri, AuxCSInfo = GIBOC_femur_ArticSurf(EpiFemTri, AuxCSInfo,  CoeffMorpho, 'post_condyles', debug_plots)

    # exporting articular surfaces (more triangulations can be easily added
    # commenting out the parts of interest
    print('Storing articular surfaces for export...')
    ArtSurf = {}
    ArtSurf['hip_' + side_raw] = FemHeadTri
    ArtSurf['med_cond_' + side_raw] = fullCondyle_Med_Tri
    ArtSurf['lat_cond_' + side_raw] = fullCondyle_Lat_Tri
    ArtSurf['dist_femur_' + side_raw] = DistFemTri
    ArtSurf['condyles_' + side_raw] = TriUnite(fullCondyle_Med_Tri, fullCondyle_Lat_Tri)
    
    # how to compute the joint axes
    print('Fitting femoral distal articular surfaces using ' + fit_method + ' method...')

    if fit_method == 'spheres':
        # Fit two spheres on articular surfaces of posterior condyles
        AuxCSInfo, JCS = CS_femur_SpheresOnCondyles(postCondyle_Lat_Tri, postCondyle_Med_Tri, AuxCSInfo, side_raw)
    elif fit_method == 'cylinder':
        # Fit the posterior condyles with a cylinder
        AuxCSInfo, JCS = CS_femur_CylinderOnCondyles(postCondyle_Lat_Tri, postCondyle_Med_Tri, AuxCSInfo, side_raw)
    # elif fit_method == 'ellipsoids':
        # Fit the entire condyles with an ellipsoid
        # AuxCSInfo, JCS = CS_femur_EllipsoidsOnCondyles(fullCondyle_Lat_Tri, fullCondyle_Med_Tri, AuxCSInfo, side_raw)
    else:
        # logg.error('GIBOC_femur method input has value: spheres, cylinder or ellipsoids. \n To extract the articular surfaces without calculating joint parameters you can use artic_surf_only.')
        print('GIBOC_femur method input has value: spheres, cylinder or ellipsoids. \n To extract the articular surfaces without calculating joint parameters you can use artic_surf_only.')

    # joint names (extracted from JCS defined in the fit_methods)
    joint_name_list = list(JCS.keys())
    hip_name = [name for name in joint_name_list if 'hip' in name][0]
    knee_name = [name for name in joint_name_list if 'knee' in name][0]
    side_low = hip_name[-1]

    # define segment ref system
    BCS = {}
    BCS['CenterVol'] = CenterVol
    BCS['Origin'] = AuxCSInfo['CenterFH_Renault']
    BCS['InertiaMatrix'] = InertiaMatrix
    BCS['V'] = JCS[hip_name]['V'] # needed for plotting of femurTri

    # landmark bone according to CS (only Origin and CS.V are used)
    FemurBL = landmarkBoneGeom(femurTri, BCS, 'femur_' + side_low)

    # result plot
    label_switch = 1

    if result_plots:
        fig = plt.figure()
        fig.suptitle('GIBOC | bone: femur | fit: ' + fit_method + ' | side: ' + side_low)
        alpha = 0.5
        
        # First column
        # plot full femur and final JCSs
        ax1 = fig.add_subplot(121, projection = '3d')
        
        plotTriangLight(femurTri, BCS, ax1)
        quickPlotRefSystem(JCS[hip_name], ax1)
        quickPlotRefSystem(JCS[knee_name], ax1)
        # add articular surfaces
        if fit_method == 'ellipsoids':
            ax1.plot_trisurf(fullCondyle_Lat_Tri['Points'][:,0], fullCondyle_Lat_Tri['Points'][:,1], fullCondyle_Lat_Tri['Points'][:,2], triangles = fullCondyle_Lat_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=alpha, shade=False, color = 'blue')
            ax1.plot_trisurf(fullCondyle_Med_Tri['Points'][:,0], fullCondyle_Med_Tri['Points'][:,1], fullCondyle_Med_Tri['Points'][:,2], triangles = fullCondyle_Med_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=alpha, shade=False, color = 'red')
        else:
            ax1.plot_trisurf(postCondyle_Lat_Tri['Points'][:,0], postCondyle_Lat_Tri['Points'][:,1], postCondyle_Lat_Tri['Points'][:,2], triangles = postCondyle_Lat_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=alpha, shade=False, color = 'blue')
            ax1.plot_trisurf(postCondyle_Med_Tri['Points'][:,0], postCondyle_Med_Tri['Points'][:,1], postCondyle_Med_Tri['Points'][:,2], triangles = postCondyle_Med_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=alpha, shade=False, color = 'red')
        # Remove grid
        ax1.grid(False)
        ax1.set_box_aspect([1,3,1])
        
        # add markers and labels
        plotBoneLandmarks(FemurBL, ax1, label_switch)
        
        # Second column, first row
        # femoral head
        ax2 = fig.add_subplot(222, projection = '3d')
        plotTriangLight(ProxFemTri, BCS, ax2)
        quickPlotRefSystem(JCS[hip_name], ax2)
        # Plot spheres
        # Create a sphere
        phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
        x = AuxCSInfo['RadiusFH_Renault']*np.sin(phi)*np.cos(theta)
        y = AuxCSInfo['RadiusFH_Renault']*np.sin(phi)*np.sin(theta)
        z = AuxCSInfo['RadiusFH_Renault']*np.cos(phi)

        ax2.plot_surface(x + AuxCSInfo['CenterFH_Renault'][0], y + AuxCSInfo['CenterFH_Renault'][1], z + AuxCSInfo['CenterFH_Renault'][2], \
                        color = 'green', alpha=alpha)
        # Remove grid
        ax2.grid(False)
        ax2.set_box_aspect([1,1,1])
        
        # Second column, second row
        # femoral head
        ax3 = fig.add_subplot(224, projection = '3d')
        plotTriangLight(DistFemTri, BCS, ax3)
        quickPlotRefSystem(JCS[knee_name], ax3)
        # plot fitting method
        if fit_method == 'spheres':
            # Plot spheres
            # Create a sphere
            phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
            x = AuxCSInfo['sphere_radius_lat']*np.sin(phi)*np.cos(theta)
            y = AuxCSInfo['sphere_radius_lat']*np.sin(phi)*np.sin(theta)
            z = AuxCSInfo['sphere_radius_lat']*np.cos(phi)

            ax3.plot_surface(x + AuxCSInfo['sphere_center_lat'][0], y + AuxCSInfo['sphere_center_lat'][1], z + AuxCSInfo['sphere_center_lat'][2], \
                            color = 'blue', alpha=alpha)
            
            # Create a sphere
            phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
            x = AuxCSInfo['sphere_radius_med']*np.sin(phi)*np.cos(theta)
            y = AuxCSInfo['sphere_radius_med']*np.sin(phi)*np.sin(theta)
            z = AuxCSInfo['sphere_radius_med']*np.cos(phi)

            ax3.plot_surface(x + AuxCSInfo['sphere_center_med'][0], y + AuxCSInfo['sphere_center_med'][1], z + AuxCSInfo['sphere_center_med'][2], \
                            color = 'red', alpha=alpha)
        elif fit_method == 'cylinder':
            # Plot cylinder
            plotCylinder(AuxCSInfo['Cyl_Y'], AuxCSInfo['Cyl_Radius'], AuxCSInfo['Cyl_Pt'], AuxCSInfo['Cyl_Range']*1.1, ax3, alpha = alpha, color = 'green')
        # elif fit_method == 'ellipsoids':
        #     # Plot ellipsoids
        #     plotEllipsoid(AuxCSInfo['ellips_centre_med'], AuxCSInfo['ellips_radii_med'], AuxCSInfo['ellips_evec_med'], ax3, alpha = alpha, color = 'red')
        #     plotEllipsoid(AuxCSInfo['ellips_centre_lat'], AuxCSInfo['ellips_radii_lad'], AuxCSInfo['ellips_evec_lat'], ax3, alpha = alpha, color = 'blue')
        else:
            # loggin.error('GIBOC_femur.py "method" input has value: "spheres", "cylinder" or "ellipsoids".')
            print('GIBOC_femur.py "method" input has value: "spheres", "cylinder" or "ellipsoids".')
        # Remove grid
        ax3.grid(False)
        ax3.set_box_aspect([1,1,1])

    # final printout
    print('Done.')

    return BCS, JCS, FemurBL, ArtSurf, AuxCSInfo
    
# -----------------------------------------------------------------------------
def CS_femur_SpheresOnCondyles(postCondyle_Lat, postCondyle_Med, CS, side, debug_plots = 0, in_mm = 1):
    # -------------------------------------------------------------------------
    CS = CS.copy()
    JCS = {}
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1
    
    # get sign correspondent to body side
    side_sign, side_low = bodySide2Sign(side)
    
    # joint names
    knee_name = 'knee_' + side_low
    hip_name  = 'hip_' + side_low
    
    # fit spheres to the two posterior condyles
    center_lat, radius_lat, e1 = sphere_fit(postCondyle_Lat['Points']) # lat
    center_med, radius_med, e2 = sphere_fit(postCondyle_Med['Points']) # med
    center_lat = center_lat.T
    center_med = center_med.T
    
    # knee center in the middle
    KneeCenter = 0.5*(center_lat + center_med)
    
    # store axes in structure
    CS['sphere_center_lat'] = center_lat
    CS['sphere_radius_lat'] = radius_lat
    CS['sphere_center_med'] = center_med
    CS['sphere_radius_med'] = radius_med
    
    # common axes: X is orthog to Y and Z, which are not mutually perpend
    Z = preprocessing.normalize((center_lat - center_med)*side_sign, axis=0)
    Z = np.reshape(Z,(Z.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    Y = preprocessing.normalize(CS['CenterFH_Renault'] - KneeCenter, axis=0)
    Y = np.reshape(Y,(Y.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    X = np.cross(Y.T, Z.T).T
    # X = np.reshape(X,(X.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    # define hip joint
    Zml_hip = np.cross(X.T, Y.T).T
    # Zml_hip = np.reshape(Zml_hip,(Zml_hip.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    JCS[hip_name] = {}
    JCS[hip_name]['V'] = np.zeros((3,3))
    JCS[hip_name]['V'][:,0] = X[:,0]
    JCS[hip_name]['V'][:,1] = Y[:,0]
    JCS[hip_name]['V'][:,2] = Zml_hip[:,0]
    JCS[hip_name]['child_location'] = CS['CenterFH_Renault']*dim_fact
    JCS[hip_name]['child_orientation'] = computeXYZAngleSeq(JCS[hip_name]['V'])
    JCS[hip_name]['Origin'] = CS['CenterFH_Renault']
    
    # define knee joint
    Y_knee = np.cross(Z.T, X.T).T
    JCS[knee_name] = {}
    JCS[knee_name]['V'] = np.zeros((3,3))
    JCS[knee_name]['V'][:,0] = X[:,0]
    JCS[knee_name]['V'][:,1] = Y_knee[:,0]
    JCS[knee_name]['V'][:,2] = Z[:,0]
    JCS[knee_name]['parent_location'] = KneeCenter*dim_fact
    JCS[knee_name]['parent_orientation'] = computeXYZAngleSeq(JCS[knee_name]['V'])
    JCS[knee_name]['Origin'] = KneeCenter
    
    # -------------------------
    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(postCondyle_Lat['Points'][:,0], postCondyle_Lat['Points'][:,1], postCondyle_Lat['Points'][:,2], triangles = postCondyle_Lat['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.3, shade=False, color = 'blue')
        ax.plot_trisurf(postCondyle_Med['Points'][:,0], postCondyle_Med['Points'][:,1], postCondyle_Med['Points'][:,2], triangles = postCondyle_Med['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.3, shade=False, color = 'red')
        
        # Plot spheres
        # Create a sphere
        phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
        x = radius_lat*np.sin(phi)*np.cos(theta)
        y = radius_lat*np.sin(phi)*np.sin(theta)
        z = radius_lat*np.cos(phi)

        ax.plot_surface(x + center_lat[0], y + center_lat[1], z + center_lat[2], \
                        color = 'blue', alpha=0.7)
        
        # Create a sphere
        phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
        x = radius_med*np.sin(phi)*np.cos(theta)
        y = radius_med*np.sin(phi)*np.sin(theta)
        z = radius_med*np.cos(phi)

        ax.plot_surface(x + center_med[0], y + center_med[1], z + center_med[2], \
                        color = 'red', alpha=0.7)
            
        ax.set_box_aspect([1,1,1])
    # -------------------------   
    
    return CS, JCS

# -----------------------------------------------------------------------------
def CS_femur_CylinderOnCondyles(Condyle_Lat, Condyle_Med, CS, side, debug_plots = 0, in_mm = 1, th = 1e-08):
    # -------------------------------------------------------------------------
    # NOTE: 
    # -------------------------------------------------------------------------
    CS = CS.copy()
    JCS = {}
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1

    # get sign correspondent to body side
    side_sign, side_low = bodySide2Sign(side)

    # joint names
    knee_name = 'knee_' + side_low
    hip_name  = 'hip_' + side_low

    # get all points of triangulations
    PtsCondyle = np.concatenate((Condyle_Lat['Points'], Condyle_Med['Points']))

    # initialise the least square search for cylinder with the sphere fitting
    # note that this function provides an already adjusted direction of the M-L
    # axis that will be used for aligning the cylinder axis below.
    CSSph, JCSSph = CS_femur_SpheresOnCondyles(Condyle_Lat, Condyle_Med, CS, side, debug_plots, in_mm)

    # initialise variables
    Axe0 = CSSph['sphere_center_lat'] - CSSph['sphere_center_med']
    Center0 = 0.5*(CSSph['sphere_center_lat'] + CSSph['sphere_center_med'])
    Radius0 = 0.5*(CSSph['sphere_radius_lat'] + CSSph['sphere_radius_med'])
    Z_dir = JCSSph[knee_name]['V'][:,2]
    Z_dir = np.reshape(Z_dir,(Z_dir.size, 1)) # convert 1d (3,) to 2d (3,1) vector

    # ----------------------------------------
    tmp_axe = Axe0 - Center0
    # identify plane which proyection is a circunference, i.e.: plane XY, YZ or XZ)
    PoP = np.argmin(np.abs(tmp_axe))
    if PoP == 0:
        x1 = 1
        x2 = 2
    elif PoP == 1:
        x1 = 0
        x2 = 2
    elif PoP == 2:
        x1 = 0
        x2 = 1
    # Symmetry axis
    Saxis = np.argmax(np.abs(tmp_axe))
    # rotation about the two axes that no correspond with symmetry axis  
    alpha = np.arctan2(tmp_axe[x2],tmp_axe[x1])
    beta = np.arctan2(tmp_axe[x1],tmp_axe[x2])

    p = np.array([Center0[x1][0],Center0[x2][0],alpha[0],beta[0],Radius0])
    xyz = PtsCondyle

    est_p =  cylinderFitting(xyz, p, th)
    # ------------------------------------------------------------
    x0n = np.zeros((3,1))
    x0n[x1] = est_p[0]
    x0n[x2] = est_p[1]
    x0n[PoP] = Center0[PoP][0]
    
    rn = est_p[4]
    an = np.zeros((3,1))
    an[x1] = rn*np.cos(est_p[3])
    an[PoP] = rn*np.cos(est_p[2])
    an[Saxis] = tmp_axe[Saxis][0]

    # Y2 is the cylinder axis versor
    Y2 = preprocessing.normalize(an, axis=0)


    # compute areas properties of condyles
    PptiesLat = TriMesh2DProperties(Condyle_Lat)
    PptiesMed = TriMesh2DProperties(Condyle_Med)

    # extract computed centroid
    CenterPtsLat = PptiesLat['Center']
    CenterPtsMed = PptiesMed['Center']

    # project centroid of each condyle on Y2 (the axis of the cylinder)
    OnAxisPtLat = x0n.T + np.dot(CenterPtsLat-x0n.T, Y2) * Y2.T
    OnAxisPtMed = x0n.T + np.dot(CenterPtsMed-x0n.T, Y2) * Y2.T

    # knee centre is the midpoint of the two projects points
    KneeCenter = 0.5*(OnAxisPtLat + OnAxisPtMed).T

    # projecting condyle points on cylinder axis
    PtsCondyldeOnCylAxis = (np.dot(PtsCondyle - x0n.T, Y2) * Y2.T) + x0n.T

    # store cylinder data
    CS['Cyl_Y'] = Y2 # normalised axis from lst
    CS['Cyl_Pt'] = x0n
    CS['Cyl_Radius'] = rn
    CS['Cyl_Range'] = np.max(np.dot(PtsCondyldeOnCylAxis, Y2)) - np.min(np.dot(PtsCondyldeOnCylAxis, Y2))

    # common axes: X is orthog to Y and Z, which are not mutually perpend
    Y = preprocessing.normalize((CS['CenterFH_Renault'] - KneeCenter), axis=0) # mech axis of femur
    Z = preprocessing.normalize(np.sign(np.dot(Y2.T,Z_dir))*Y2, axis=0) # cylinder axis (ALREADY side-adjusted)
    X = np.cross(Y.T,Z.T).T

    # define hip joint
    Zml_hip = np.cross(X.T, Y.T).T
    JCS[hip_name] = {}
    JCS[hip_name]['V'] = np.zeros((3,3))
    JCS[hip_name]['V'][:,0] = X[:,0]
    JCS[hip_name]['V'][:,1] = Y[:,0]
    JCS[hip_name]['V'][:,2] = Zml_hip[:,0]
    JCS[hip_name]['child_location'] = CS['CenterFH_Renault']*dim_fact
    JCS[hip_name]['child_orientation'] = computeXYZAngleSeq(JCS[hip_name]['V'])
    JCS[hip_name]['Origin'] = CS['CenterFH_Renault']

    # define knee joint
    Y_knee = np.cross(Z.T, X.T).T
    JCS[knee_name] = {}
    JCS[knee_name]['V'] = np.zeros((3,3))
    JCS[knee_name]['V'][:,0] = X[:,0]
    JCS[knee_name]['V'][:,1] = Y_knee[:,0]
    JCS[knee_name]['V'][:,2] = Z[:,0]
    JCS[knee_name]['parent_location'] = KneeCenter*dim_fact
    JCS[knee_name]['parent_orientation'] = computeXYZAngleSeq(JCS[knee_name]['V'])
    JCS[knee_name]['Origin'] = KneeCenter

    # -------------------------
    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        ax.plot_trisurf(Condyle_Lat['Points'][:,0], Condyle_Lat['Points'][:,1], Condyle_Lat['Points'][:,2], triangles = Condyle_Lat['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.3, shade=False, color = 'blue')
        ax.plot_trisurf(Condyle_Med['Points'][:,0], Condyle_Med['Points'][:,1], Condyle_Med['Points'][:,2], triangles = Condyle_Med['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.3, shade=False, color = 'red')
        
        # Plot cylinder
        plotCylinder(Y2, rn, KneeCenter, CS['Cyl_Range']*1.1, ax)
        
        ax.set_box_aspect([1,3,1])
    # -------------------------   
    
    return CS, JCS
    
# --------------------------------------------------------------
def Kai2014_tibia(tibiaTri, side_raw = 'r', result_plots = 1, debug_plots = 0, in_mm = 1):
    # -------------------------------------------------------------------------
    # Custom implementation of the method for defining a reference system of 
    # the tibia described in the following publication: Kai, Shin, et al. 
    # Journal of biomechanics 47.5 (2014): 1229-1233. https://doi.org/10.1016/j.jbiomech.2013.12.013.
    # The algorithm slices the tibia along the vertical axis identified via
    # principal component analysis, identifies the largest section and fits an
    # ellips to it. It finally uses the ellipse axes to define the reference
    # system. This implementation includes several non-obvious checks to ensure 
    # that the bone geometry is always sliced in the correct direction.
    # 
    # Inputs:
    # tibiaTri - MATLAB triangulation object of the entire tibial geometry.
    # 
    # side_raw - generic string identifying a body side. 'right', 'r', 'left' 
    # and 'l' are accepted inputs, both lower and upper cases.
    # 
    # result_plots - enable plots of final fittings and reference systems. 
    # Value: 1 (default) or 0.
    # 
    # debug_plots - enable plots used in debugging. Value: 1 or 0 (default).
    # 
    # in_mm - (optional) indicates if the provided geometries are given in mm
    # (value: 1) or m (value: 0). Please note that all tests and analyses
    # done so far were performed on geometries expressed in mm, so this
    # option is more a placeholder for future adjustments.
    # 
    # Outputs:
    # BCS - MATLAB structure containing body reference system and other 
    # geometrical features identified by the algorithm.
    # 
    # JCS - MATLAB structure containing the joint reference systems connected
    # to the bone being analysed. These might NOT be sufficient to define
    # a joint of the musculoskeletal model yet.
    # 
    # femurBL - MATLAB structure containing the bony landmarks identified 
    # on the bone geometries based on the defined reference systems. Each
    # field is named like a landmark and contain its 3D coordinates.
    # -------------------------------------------------------------------------

    AuxCSInfo = {}
    BCS = {}
    JCS = {}
    # Slices 1 mm apart as in Kai et al. 2014
    slices_thickness = 1
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1

    # get side id correspondent to body side 
    side_sign, side_low = bodySide2Sign(side_raw)

    # inform user about settings
    print('---------------------')
    print('   KAI2014 - TIBIA   ')
    print('---------------------')
    print('* Body Side: ' + side_low.upper())
    print('* Fit Method: "N/A"')
    print('* Result Plots: ' + ['Off','On'][result_plots])
    print('* Debug  Plots: ' + ['Off','On'][debug_plots])
    print('* Triang Units: mm')
    print('---------------------')
    print('Initializing method...')

    # it is assumed that, even for partial geometries, the tibial bone is
    # always provided as unique file. Previous versions of this function did
    # use separated proximal and distal triangulations. Check Git history if
    # you are interested in that.
    print('Computing PCA for given geometry...')
    pca = PCA()
    pca.fit(tibiaTri['Points'])
    V_all = np.transpose(pca.components_)

    # guess vertical direction, pointing proximally
    U_DistToProx = tibia_guess_CS(tibiaTri, debug_plots)

    # divide bone in three parts and take proximal and distal 
    ProxTib, DistTib = cutLongBoneMesh(tibiaTri, U_DistToProx)

    # center of the volume
    _, CenterVol, InertiaMatrix, _, _ = TriInertiaPpties(tibiaTri)

    # checks on vertical direction
    Y0 = V_all[:,0]
    Y0 = np.reshape(Y0,(Y0.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    # NOTE: not redundant for partial models, e.g. ankle model. If this check
    # is not implemented the vertical axis is not aligned with the Z axis of
    # the images
    Y0 *= np.sign(np.dot((np.mean(ProxTib['Points'], axis=0) - np.mean(DistTib['Points'], axis=0)),Y0))

    # slice tibia along axis and get maximum height
    print('Slicing tibia longitudinally...')

    _, _, _, _, AltAtMax = TriSliceObjAlongAxis(tibiaTri, Y0, slices_thickness)

    # slice geometry at max area
    Curves , _, _ = TriPlanIntersect(tibiaTri, Y0, -AltAtMax)

    # keep just the largest outline (tibia section)
    maxAreaSection, N_curves, _ = getLargerPlanarSect(Curves)

    # check number of curves
    if N_curves > 2:
        # loggin.warning('There are ' + str(N_curves) + ' section areas at the largest tibial slice.')
        # loggin.error('This should not be the case (only tibia and possibly fibula should be there).')
        print('There are ' + str(N_curves) + ' section areas at the largest tibial slice.')
        print('This should not be the case (only tibia and possibly fibula should be there).')

    # Move the outline curve points in the inertial ref system, so the vertical
    # component [:,0] is orthogonal to a plane
    PtsCurves = np.dot(maxAreaSection['Pts'], V_all)

    # Fit a planar ellipse to the outline of the tibia section
    print('Fitting ellipse to largest section...')
    FittedEllipse = fit_ellipse(PtsCurves[:,1], PtsCurves[:,2])

    # depending on the largest axes, YElpsMax is assigned.
    # vector shapes justified by the rotation matrix used in fit_ellipse
    # R = [[cos_phi, sin_phi], 
    #       [-sin_phi, cos_phi]]

    if FittedEllipse['width'] > FittedEllipse['height']:
        # horizontal ellipse
        tmp = np.array([ 0, np.cos(FittedEllipse['phi']), -np.sin(FittedEllipse['phi'])])
        tmp = np.reshape(tmp,(tmp.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        ZElpsMax = np.dot(V_all, tmp)
    else:
        # vertical ellipse - get
        tmp = np.array([ 0, np.sin(FittedEllipse['phi']), np.cos(FittedEllipse['phi'])])
        tmp = np.reshape(tmp,(tmp.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        ZElpsMax = np.dot(V_all, tmp)

    # create the ellipse
    Ux = np.array([np.cos(FittedEllipse['phi']), -np.sin(FittedEllipse['phi'])])
    Uy = np.array([np.sin(FittedEllipse['phi']), np.cos(FittedEllipse['phi'])])
    R = np.zeros((2,2))
    R[0] = Ux
    R[1] = Uy

    # the ellipse
    theta_r = np.linspace(0, 2*np.pi, 36)
    ellipse_x_r = FittedEllipse['width']*np.cos(theta_r)
    ellipse_y_r = FittedEllipse['height']*np.sin(theta_r)
    tmp_ellipse_r = np.zeros((2,len(ellipse_x_r)))
    tmp_ellipse_r[0,:] = ellipse_x_r
    tmp_ellipse_r[1,:] = ellipse_y_r

    rotated_ellipse = (R @ tmp_ellipse_r).T
    # FittedEllipse['data] = rotated_ellipse
    rotated_ellipse[:,0] = rotated_ellipse[:,0] + FittedEllipse['X0']
    rotated_ellipse[:,1] = rotated_ellipse[:,1] + FittedEllipse['Y0']

    # check ellipse fitting
    if debug_plots:
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter(PtsCurves[:,1], PtsCurves[:,2], color = 'k')    
        ax.plot(rotated_ellipse[:,0], rotated_ellipse[:,1], color = 'green')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        

    # centre of ellipse back to medical images reference system
    tmp_center = np.array([np.mean(PtsCurves[:,0]), FittedEllipse['X0'], FittedEllipse['Y0']])
    tmp_center = np.reshape(tmp_center,(tmp_center.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    CenterEllipse = np.dot(V_all, tmp_center)

    # identify lateral direction
    U_tmp, MostDistalMedialPt, just_tibia = tibia_identify_lateral_direction(DistTib, Y0)

    if just_tibia:
        m_col = 'red'
    else:
        m_col = 'blue'

    # adjust for body side, so that U_tmp is aligned as Z_ISB
    U_tmp *= side_sign

    # making Y0/U_temp normal to Z0 (still points laterally)
    Z0_temp = preprocessing.normalize(U_tmp - np.dot(U_tmp.T, Y0)*Y0, axis=0)

    # here the assumption is that Y0 has correct m-l orientation               
    ZElpsMax *= np.sign(np.dot(Z0_temp.T, ZElpsMax))

    tmp_pts = np.ones((len(rotated_ellipse), 3))
    tmp_pts[:,1:] = rotated_ellipse
    AuxCSInfo['EllipsePts'] = np.dot(V_all, tmp_pts.T).T

    # common axes: X is orthog to Y and Z, in this case ARE mutually perpend
    Y = preprocessing.normalize(Y0, axis=0)
    Z = preprocessing.normalize(ZElpsMax, axis=0)
    X = np.cross(Y.T, Z.T).T
    # X = np.cross(Z.T, Y.T).T

    # z for body ref system
    Z_cs = np.cross(X.T, Y.T).T

    # segment reference system
    BCS['CenterVol'] = CenterVol
    BCS['Origin'] = CenterEllipse
    BCS['InertiaMatrix'] = InertiaMatrix
    BCS['V'] = np.zeros((3,3))
    # BCS['V'][:,0] = X[:,0]
    # BCS['V'][:,1] = Y[:,0]
    # BCS['V'][:,2] = Z_cs[:,0]
    BCS['V'][:,0] = -Z_cs[:,0]
    BCS['V'][:,1] = Y[:,0]
    BCS['V'][:,2] = X[:,0]

    # define the knee reference system
    joint_name = 'knee_' + side_low
    # define knee joint
    Ydp_knee = np.cross(Z.T, X.T).T
    JCS[joint_name] = {}
    JCS[joint_name]['V'] = np.zeros((3,3))
    # JCS[joint_name]['V'][:,0] = X[:,0]
    # JCS[joint_name]['V'][:,1] = Ydp_knee[:,0]
    # JCS[joint_name]['V'][:,2] = Z[:,0]
    JCS[joint_name]['V'][:,0] = -Z[:,0]
    JCS[joint_name]['V'][:,1] = Ydp_knee[:,0]
    JCS[joint_name]['V'][:,2] = X[:,0]
    JCS[joint_name]['Origin'] = CenterEllipse

    # NOTE THAT CS['V'] and JCS['knee_r']['V'] are the same, so the distinction is 
    # here purely formal. This is because all axes are perpendicular.
    JCS[joint_name]['child_orientation'] = computeXYZAngleSeq(JCS[joint_name]['V'])
    JCS[joint_name]['child_location'] = CenterEllipse*dim_fact

    # landmark bone according to CS (only Origin and CS.V are used)
    tibiaBL = landmarkBoneGeom(tibiaTri, BCS, 'tibia_' + side_low)

    if just_tibia == False:
        # add landmark as 3x1 vector
        tibiaBL[side_low.upper() + 'LM'] = MostDistalMedialPt

    label_switch = 1
    # plot reference systems
    if result_plots:
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        
        ax.set_title('Kai2014 | bone: tibia | side: ' + side_low)
        
        plotTriangLight(tibiaTri, BCS, ax)
        quickPlotRefSystem(BCS, ax)
        quickPlotRefSystem(JCS[joint_name], ax)
        
        # plot markers and labels
        plotBoneLandmarks(tibiaBL, ax, label_switch)
        
        # plot largest section
        ax.plot(maxAreaSection['Pts'][:,0], maxAreaSection['Pts'][:,1], maxAreaSection['Pts'][:,2], color='red', linestyle='dashed', linewidth=2)
        # plotDot(MostDistalMedialPt, ax, m_col, 4)
        plotDot(MostDistalMedialPt, ax, 'yellow', 4)
            
        ax.set_box_aspect([1,3,1])
    
    return BCS, JCS, tibiaBL, AuxCSInfo

#%%
# ##################################################
# ANTHROPOMETRY ####################################
# ##################################################
# 
# -----------------------------------------------------------------------------
def gait2392MassProps(segment_name = ''):
    # -------------------------------------------------------------------------
    # Return the mass and inertia of the body segments of model gait2392, 
    # which for the lower limb are the same as Rajagopal's model. The script 
    # saves from reading the OpenSim model directly. Note that side does not 
    # need to be specified as mass properties are the same on both.
    # 
    # Inputs:
    # segment_name - a string with the name of an OpenSim body included in
    # the gait2392 model. i.e. 'pelvis', 'femur', 'tibia', 'talus',
    # 'calcn', 'toes', 'patella' and 'torso'.
    # 
    # Outputs:
    # MP - a dictionary with keys 'mass', 'mass_center', 'inertia_xx',
    # 'inertia_yy', 'inertia_zz'.
    # 
    # -------------------------------------------------------------------------
    # 
    # Note that the Rajagopal model has the same lower limb inertial 
    # properties, which is why I added the patella from that model.
    # torso differs because that model has arms.
    
    MP = {}
    
    if segment_name != 'pelvis' and segment_name != 'torso' and segment_name != 'full_body':
        segment_name = segment_name[:-2]
        
    if segment_name == 'full_body':
        # This is: pelvis + 2*(fem+tibia+talus+foot+toes+patella)+torso 
        MP['mass'] = 11.777 + 2*(9.3014+3.7075+0.1+1.25+0.2166+0.0862)+34.2366
        
    elif segment_name == 'pelvis':
        MP['mass'] = 11.777
        MP['mass_center'] = np.array([-0.0707, 0, 0])
        MP['inertia_xx'] = 0.1028
        MP['inertia_yy'] = 0.0871
        MP['inertia_zz'] = 0.0579
        
    elif segment_name == 'femur':
        MP['mass'] = 9.3014
        MP['mass_center'] = np.array([0, -0.17, 0])
        MP['inertia_xx'] = 0.1339
        MP['inertia_yy'] = 0.0351
        MP['inertia_zz'] = 0.1412
        
    elif segment_name == 'tibia':
        MP['mass'] = 3.7075
        MP['mass_center'] = np.array([0, -0.1867, 0])
        MP['inertia_xx'] = 0.0504
        MP['inertia_yy'] = 0.0051
        MP['inertia_zz'] = 0.0511
    
    elif segment_name == 'talus':
        MP['mass'] = 0.1
        MP['mass_center'] = np.array([0, 0, 0])
        MP['inertia_xx'] = 0.001
        MP['inertia_yy'] = 0.001
        MP['inertia_zz'] = 0.001
    
    elif segment_name == 'calcn':
        MP['mass'] = 1.25
        MP['mass_center'] = np.array([0.1, 0.03, 0])
        MP['inertia_xx'] = 0.0014
        MP['inertia_yy'] = 0.0039
        MP['inertia_zz'] = 0.0041
    
    elif segment_name == 'toes':
        MP['mass'] = 0.2166
        MP['mass_center'] = np.array([0.0346, 0.006, -0.0175])
        MP['inertia_xx'] = 0.0001
        MP['inertia_yy'] = 0.0002
        MP['inertia_zz'] = 0.0001
        
    elif segment_name == 'torso':
        # different from Rajagopal that has arms
        MP['mass'] = 34.2366
        MP['mass_center'] = np.array([-0.03, 0.32, 0])
        MP['inertia_xx'] = 1.4745
        MP['inertia_yy'] = 0.7555
        MP['inertia_zz'] = 1.4314
    
    elif segment_name == 'patella':
        MP['mass'] = 0.0862
        MP['mass_center'] = np.array([0.0018, 0.0264, 0])
        MP['inertia_xx'] = 2.87e-006
        MP['inertia_yy'] = 1.311e-005
        MP['inertia_zz'] = 1.311e-005
        
    else:
        # loggin.error('Please specify a segment name among those included in the gait2392 model')
        print('Please specify a segment name among those included in the gait2392 model')
    
    return MP

# -----------------------------------------------------------------------------
def mapGait2392MassPropToModel(osimModel):
    # -------------------------------------------------------------------------
    # Map the mass properties of the gait2392 model to the equivalent segments 
    # of the model specified as input.
    # 
    # Inputs:
    # osimModel - the OpenSim model for which the mass properties of the 
    # segments will be updated using the gait2392 values.
    # 
    # Outputs:
    # osimModel - the OpenSim model with the updated inertial properties.
    # 
    # -------------------------------------------------------------------------
    
    # loop through the bodies of the model
    N_bodies = osimModel.getBodySet().getSize()
    
    for n_b in range(N_bodies):
        curr_body = osimModel.getBodySet().get(n_b)
        curr_body_name = str(curr_body.getName())
        
        # retried mass properties of gait2392
        massProp = gait2392MassProps(curr_body_name)
        
        # retrieve segment to update
        curr_body = osimModel.getBodySet().get(curr_body_name)
        
        # assign mass
        curr_body.setMass(massProp['mass'])
        
        # build a matrix of inertia with the gait2392 values
        xx = massProp['inertia_xx']
        yy = massProp['inertia_yy']
        zz = massProp['inertia_zz']
        xy = 0.0
        xz = 0.0
        yz = 0.0
        upd_inertia = opensim.Inertia(xx, yy, zz, xy, xz, yz)
        
        # set inertia
        curr_body.setInertia(upd_inertia)
        print('Mapped on body: ' + curr_body_name)
        
    return osimModel

# -----------------------------------------------------------------------------
def scaleMassProps(osimModel, coeff):
    # -------------------------------------------------------------------------
    # Scale mass and inertia of the bodies of an OpenSim model assuming that 
    # the geometry stays constant and only the mass changes proportionally to 
    # a coefficient assigned in input.
    # 
    # Inputs:
    # osimModel - the OpenSim model for which the mass properties of the
    # segments will be scaled.
    # 
    # coeff - ratio of mass new_mass/curr_model_mass. This is used to scale
    # the inertial properties of the gait2392 model to the mass of a
    # specific individual.
    # 
    # Outputs:
    # osimModel - the OpenSim model with the scaled inertial properties.
    # 
    # -------------------------------------------------------------------------
    
    # get bodyset
    subjspec_bodyset = osimModel.getBodySet()
    for n_b in range(subjspec_bodyset.getSize()):
        
        curr_body = subjspec_bodyset.get(n_b)
        
        # updating the mass
        curr_body.setMass(coeff*curr_body.getMass())
        
        # updating the inertia matrix for the change in mass
        m = curr_body.get_inertia()
    
        # components of inertia
        xx = m.get(0)*coeff
        yy = m.get(1)*coeff
        zz = m.get(2)*coeff
        xy = m.get(3)*coeff
        xz = m.get(4)*coeff
        yz = m.get(5)*coeff
        upd_inertia = opensim.Inertia(xx, yy, zz, xy, xz, yz)
        
        # updating Inertia
        curr_body.setInertia(upd_inertia)
        
        del m
        
    return osimModel

# -----------------------------------------------------------------------------
def assignMassPropsToSegments(osimModel, JCS = {}, subj_mass = 0, side_raw = ''):
    # -------------------------------------------------------------------------
    # Assign mass properties to the segments of an OpenSim model created 
    # automatically. Mass and inertia are scaled from the values used in the 
    # gait2392 model.
    # NOTE: this function will be rewritten (prototype).
    # 
    # Inputs:
    # osimModel - an OpenSim model generated automatically for which the mass
    # properties needs to be personalised.
    # 
    # JCS - a dictionary including the joint coordinate system computed from
    # the bone geometries. Required for computing the segment lengths and
    # identifying the COM positions.
    # 
    # subj_mass - the total mass of the individual of which we are building a
    # model, in Kg. Required for scaling the mass properties of the
    # generic gati2392 model.
    # 
    # side_raw - generic string identifying a body side. 'right', 'r', 'left'
    # and 'l' are accepted inputs, both lower and upper cases.
    # 
    # Outputs:
    # osimModel - the OpenSim model with the personalised mass properties.
    # 
    # -------------------------------------------------------------------------
    
    if side_raw == '':
        side = inferBodySideFromAnatomicStruct(JCS)
    else:
        # get sign correspondent to body side
        _, side = bodySide2Sign(side_raw)

    femur_name = 'femur_' + side
    tibia_name = 'tibia_' + side
    talus_name = 'talus_' + side
    calcn_name = 'calcn_' + side
    hip_name = 'hip_' + side
    knee_name = 'knee_' + side
    ankle_name = 'ankle_' + side
    toes_name = 'mtp_' + side

    print('------------------------')
    print('  UPDATING MASS PROPS   ')
    print('------------------------')

    # compute lengths of segments from the bones and COM positions using 
    # coefficients from Winter 2015 (book)

    # Keep in mind that all Origin fields have [3x1] dimensions
    print('Updating centre of mass position (Winter 2015)...')
    if femur_name in JCS:
        # compute thigh length
        thigh_axis = JCS[femur_name][hip_name]['Origin'] - JCS[femur_name][knee_name]['Origin']
        thigh_L = np.linalg.norm(thigh_axis)
        thigh_COM = thigh_L*0.567 * (thigh_axis/thigh_L) + JCS[femur_name][knee_name]['Origin']
        # assign  thigh COM
        thigh_COM /= 1000
        osimModel.getBodySet().get(femur_name).setMassCenter(opensim.ArrayDouble.createVec3(thigh_COM[0][0], thigh_COM[1][0], thigh_COM[2][0]))
        
        # shank
        if talus_name in JCS:
            # compute thigh length
            shank_axis = JCS[talus_name][knee_name]['Origin'] - JCS[talus_name][ankle_name]['Origin']
            shank_L = np.linalg.norm(shank_axis)
            shank_COM = shank_L*0.567 * (shank_axis/shank_L) + JCS[talus_name][ankle_name]['Origin']
            # assign  thigh COM
            shank_COM /= 1000 
            osimModel.getBodySet().get(tibia_name).setMassCenter(opensim.ArrayDouble.createVec3(shank_COM[0][0], shank_COM[1][0], shank_COM[2][0]))
            
            # foot
            if calcn_name in JCS:
                # compute thigh length
                foot_axis = JCS[talus_name][knee_name]['Origin'] - JCS[calcn_name][toes_name]['Origin']
                foot_L = np.linalg.norm(foot_axis)
                calcn_COM = shank_L*0.5 * (foot_axis/foot_L) + JCS[calcn_name][toes_name]['Origin']
                # assign  thigh COM
                calcn_COM /= 1000
                osimModel.getBodySet().get(calcn_name).setMassCenter(opensim.ArrayDouble.createVec3(calcn_COM[0][0], calcn_COM[1][0], calcn_COM[2][0]))
                
    # -----------------------------------------------------------------------------
    # map gait2392 properties to the model segments as an initial value
    print('Mapping segment masses and inertias from gait2392 model.')
    osimModel = mapGait2392MassPropToModel(osimModel)

    # opensim model total mass (consistent in gait2392 and Rajagopal)
    MP = gait2392MassProps('full_body')
    gait2392_tot_mass = MP['mass']

    # calculate mass ratio of subject mass and gait2392 mass
    coeff = subj_mass/gait2392_tot_mass

    # scale gait2392 mass properties to the individual subject
    print('Scaling inertial properties to assigned body weight...')
    scaleMassProps(osimModel, coeff)

    print('Done.')
    
    return osimModel

#%%
# ##################################################
# GIBOC - core #####################################
# ##################################################
# 
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
    
    if len(PointsKept) == 0 or len(ElmtsKept) == 0:
        TRout = TR
    else:
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
        if border == [] and patch != []:
            Patch[str(i)] = patch
            
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
        tmp_Nce = []
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
        
        # Nce = list(set(FRNei + SRNei + TRNei))
        tmp_Nce = FRNei + SRNei + TRNei
        Nce = np.unique(TR['ConnectivityList'][tmp_Nce].reshape(-1,1)[:,0])
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
    
    tmp_ElIn = [np.where(TR['ConnectivityList'] == ni)[0] for ni in NodeInitial]
    ElIn = []
    for values in tmp_ElIn:
        ElIn += list(values)
    # ElmtsInitial = np.unique(TR['ConnectivityList'][NodeInitial].reshape(-1, 1))
    ElmtsInitial = np.unique(TR['ConnectivityList'][ElIn].reshape(-1, 1))
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
    
    # identify ID from TR['ConnectivityList'] where ElmtsConnected are included
    tmp_ElConn = [np.where(TR['ConnectivityList'] == ni)[0] for ni in ElmtsConnected]
    ElConn = []
    for values in tmp_ElConn:
        ElConn += list(values)
    ElmtsConnected = list(set(ElConn))
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
            # Elmts2Delete.append(nei)
            Elmts2Delete += list(nei)
        
        # remove duplicated elements
        Elmts2Delete = list(set(Elmts2Delete))
        
        Elmts2Keep = np.ones(len(TR1['ConnectivityList']), dtype='bool')
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

# -----------------------------------------------------------------------------
def PlanPolygonCentroid3D(Pts):
    # -------------------------------------------------------------------------
    # PlanPolygonCentroid3D Find the centroid of a 2D Polygon, decribed by its
    # boundary (a close curve) in a 3D space.
    # Works with arbitrary shapes (convex or not)
    # -------------------------------------------------------------------------
    Centroid = np.nan
    Area = np.nan
    
    if np.all(np.shape(Pts) == np.array([0,0])):
        logging.warning('PlanPolygonCentroid3D Empty Pts variable.')
        Centroid = np.nan
        Area = np.nan
        return Centroid, Area
    
    if np.any(Pts[0] != Pts[-1]):
        Pts = np.concatenate((Pts, np.array([Pts[0]])), axis=0)
    
    # Initial Guess of center
    Center0 = np.mean(Pts[:-1], axis = 0)
    
    # Middle Point of each polygon side
    PtsMiddle = Pts[:-1] + np.diff(Pts, axis = 0)/2
    
    # Get the centroid of each points connected
    TrianglesCentroid  = PtsMiddle - 1/3*(PtsMiddle-Center0)
    
    # Get the area of each triangles
    _, V = np.linalg.eig(np.cov(Pts[:-1].T))
    n = V[0] # normal to polygon plan
    
    TrianglesArea = 1/2*np.dot(np.cross(np.diff(Pts, axis = 0),-(Pts[:-1]-Center0)), n)
    
    # Barycenter of triangles
    Centroid = np.sum(TrianglesCentroid*np.tile(TrianglesArea, [3,1]).T, axis = 0)/np.sum(TrianglesArea)
    
    Area = np.abs(np.sum(TrianglesArea))
    
    return Centroid, Area
    
# -----------------------------------------------------------------------------
def getLargerPlanarSect(Curves = {}):
    # -------------------------------------------------------------------------
    # gets the larger section in a set of curves. Use mostly by GIBOC_tibia
    # -------------------------------------------------------------------------
    
    N_curves = len(Curves)
    Areas = []
    
    # check to use just the tibial curve, as in GIBOK
    Areas = [Curves[key]['Area'] for key in Curves]
    
    ind_max_area = np.argmax(Areas)
    
    Curve = Curves[list(Curves.keys())[ind_max_area]]
    
    return Curve, N_curves, Areas

# -----------------------------------------------------------------------------
def computeMassProperties_Mirtich1996(Tri = {}):
    # -------------------------------------------------------------------------
    # Script that given the vertices v and facets of a triangular mesh
    # calculates the inertial properties (Volume, Mass, COM and Inertia matrix
    # based on: Mirtich, B., 1996. Fast and accurate computation of polyhedral 
    # mass properties. journal of graphics tools 1, 31-50. The algorithm is not
    # the generic one presented in that publication, which works for any kind
    # of polihedron, but is optimized to work with triangular meshes. The
    # implementation was taken from Eberly, D., 2003. Polyhedral mass 
    # properties (revisited). AVAILABLE AT: 
    # https://www.geometrictools.com/Documentation/PolyhedralMassProperties.pdf
    # 
    # VERIFICATION: this code yealds the same values as NMSBuilder for a femur
    # and a sphere. (Exactly the same values, but it's faster!)
    # 
    # INPUT: 
    # Tri: triangulation dic including the coordinates of n points and 
    # connectivity list collecting the indices of the vertices of
    # each facet of the mesh.
    # 
    # OUTPUT:   
    # MassInfo['mass'] = mass
    # MassInfo['COM'] = COM
    # MassInfo['Imat'] = I (inertia matrix calculated at COM)
    # MassInfo['Ivec'] = inertial vector for use in OpenSim
    # -------------------------------------------------------------------------

    MassInfo = {}
    # feedback to the user
    # I tried a waitbar, but the script was too slow! 
    
    print('Calculating Inertia properties...')
    
    _, COM, I, _, mass = TriInertiaPpties(Tri)
    
    # inertial vector (for use in OpenSim
    Iv = np.array([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]])
    
    # Collecting all results together
    MassInfo['mass'] = mass
    MassInfo['COM'] = COM
    MassInfo['Imat'] = I
    MassInfo['Ivec'] = Iv
    
    return MassInfo






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
        # CS = CS.copy()
        # if there are no axes but there is a pose matrix, use the matrix as 
        # reference
        if not 'Y' in CS and 'V' in CS:
            CS['X'] = CS['V'][:,0]
            CS['Y'] = CS['V'][:,1]
            CS['Z'] = CS['V'][:,2]
        
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

# -----------------------------------------------------------------------------
def plotCylinder(symmetry_axis, radius, center, length, ax, alpha = 0.6, color = 'b'):
    # -------------------------------------------------------------------------
    # Create a cylinder
    height_z = np.linspace(-length/2, length/2, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, height_z)

    x = radius*np.cos(theta_grid)
    y = radius*np.sin(theta_grid)
    z = z_grid
    
    # rotate the samples
    Uz = preprocessing.normalize(symmetry_axis, axis=0)
    i = np.array([1,0,0])
    i = np.reshape(i,(i.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    Ux = np.cross(Uz.T,i.T).T
    Ux = preprocessing.normalize(Ux, axis=0)
    Uy = np.cross(Ux.T, Uz.T).T

    U = np.zeros((3,3))
    U[:,0] = Ux[:,0]
    U[:,1] = Uy[:,0]
    U[:,2] = Uz[:,0]

    t = np.transpose(np.array([x,y,z]), (1,2,0))
    Xrot, Yrot, Zrot = np.transpose(np.dot(t, U), (2,0,1))
    
    # Plot the surface
    ax.plot_surface(Xrot + center[0], Yrot + center[1], Zrot + center[2], color = color, alpha=alpha)
    #label axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # Remove grid
    ax.grid(False)
    
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
    # center, width, height, phi = reg.as_parameters()
    center, width, height, phi = reg.as_parameters()
        
    ellipse_t['phi'] = phi
    ellipse_t['X0'] = center[0]
    ellipse_t['Y0'] = center[1]
    ellipse_t['height'] = height
    ellipse_t['width'] = width
    
    return ellipse_t

# -----------------------------------------------------------------------------
def cylinderFitting(xyz, p, th=1e-08):
    # 
    # This is a fitting for a vertical cylinder fitting
    # Reference:
    # http://www.int-arch-photogramm-remote-sens-spatial-inf-sci.net/XXXIX-B5/169/2012/isprsarchives-XXXIX-B5-169-2012.pdf
    # 
    # xyz is a matrix contain at least 5 rows, and each row stores x y z of a cylindrical surface
    # p is initial values of the parameter;
    # p[0] = Xc, x coordinate of the cylinder centre
    # P[1] = Yc, y coordinate of the cylinder centre
    # P[2] = alpha, rotation angle (radian) about the x-axis
    # P[3] = beta, rotation angle (radian) about the y-axis
    # P[4] = r, radius of the cylinder
    # 
    # th, threshold for the convergence of the least squares
 
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]

    fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
    errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

    est_p , success = leastsq(errfunc, p, args=(x, y, z), gtol=th, maxfev=1000)
    
    return est_p




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

#%%
# ##################################################
# GEOMETRY #########################################
# ##################################################
# 
def inferBodySideFromAnatomicStruct(anat_struct):
    # -------------------------------------------------------------------------
    # Infer the body side that the user wants to process based on a structure 
    # containing the anatomical objects (triangulations or joint definitions) 
    # given as input. The implemented logic is trivial: the fields are checked 
    # for standard names of bones and joints used in OpenSim models.
    # 
    # 
    # Inputs: anat_struct - a list or dic containing anatomical objects, 
    # e.g. a set of bone triangulation or joint definitions.
    # 
    # Outputs: guessed_side - a body side label that can be used in all other 
    # STAPLE functions requiring such input.
    # -------------------------------------------------------------------------
    guessed_side = ''
    
    if isinstance(anat_struct, dict):
        fields_side = list(anat_struct.keys())
    elif isinstance(anat_struct, list):
        fields_side = anat_struct
    else:
        print('inferBodySideFromAnatomicStruct Input must be a list.')
    
    # check using the bone names
    body_set = ['femur', 'tibia', 'talus', 'calcn']
    guess_side_b = []
    guess_side_b = [body[-1] for body in fields_side if body[:-2] in body_set]
    
    # check using the joint names
    joint_set = ['hip', 'knee', 'ankle', 'subtalar']
    guess_side_j = []
    guess_side_j = [joint[-1] for joint in fields_side if joint[:-2] in joint_set]
    
    # composed vectors
    combined_guessed = guess_side_b + guess_side_j
        
    if 'r' in list(set(combined_guessed)) and not 'l' in list(set(combined_guessed)):
        guessed_side = 'r'
    elif 'l' in list(set(combined_guessed)) and not 'r' in list(set(combined_guessed)):
        guessed_side = 'l'
    else:
        print('inferBodySideFromAnatomicStruct Error: it was not possible to infer the body side. Please specify it manually in this occurrance.')
    
    return guessed_side

# -----------------------------------------------------------------------------
def createTriGeomSet(aTriGeomList, geom_file_folder):
    # -------------------------------------------------------------------------
    # CREATETRIGEOMSET Create a dictionary of triangulation objects from a list
    # of files and the folder where those files are located
    
    # triGeomSet = createTriGeomSet(aTriGeomList, geom_file_folder)
    
    # Inputs:
    #   aTriGeomList - a list consisting of names of triangulated geometries 
    #       that will be seeked in geom_file_folder.
    #       The names set the name of the triangulated objects and should
    #       correspond to the names of the bones to include in the OpenSim
    #       models.
    
    #   geom_file_folder - the folder where triangulated geometry files will be
    #       seeked.
    
    # Outputs:
    #   triGeomSet - dictionary with as many fields as the triangulations that
    #       could be loaded from the aTriGeomList from the geom_file_folder.
    #       Each field correspond to a triangulated geometry.
    
    # Example of use:
    # tri_folder = 'test_geometries\dataset1\tri';
    # bones_list = {'pelvis','femur_r','tibia_r','talus_r','calcn_r'};
    # geom_set = createTriGeomSet(bones_list, tri_folder);
    # -------------------------------------------------------------------------
    
    print('-----------------------------------')
    print('Creating set of triangulations')
    print('-----------------------------------')
    
    t0 = time.time() # tf = time.time() - t
    
    print('Reading geometries from folder: ' +  geom_file_folder)
    
    triGeomSet = {}
    for bone in aTriGeomList:
        
        curr_tri_geo_file = os.path.join(geom_file_folder, bone)
        curr_tri_geo = load_mesh(curr_tri_geo_file + '.stl')
        if not curr_tri_geo:
            # skip building the field if this was no gemoetry
            continue
        else:
            triGeomSet[bone] = curr_tri_geo
        
        if not triGeomSet:
            logging.exception('createTriGeomSet No triangulations were read in input. \n')
        else:
            # tell user all went well
            print('Set of triangulated geometries created in ' + str(np.round(time.time() - t0, 4)))
        
    return triGeomSet

# -----------------------------------------------------------------------------
def bodySide2Sign(side_raw):
    # -------------------------------------------------------------------------
    # Returns a sign and a mono-character, lower-case string corresponding to 
    # a body side. Used in several STAPLE functions for:
    # 1) having a standard side label
    # 2) adjust the reference systems to the body side.
    # 
    # Inputs:
    #   side_raw - generic string identifying a body side. 'right', 'r', 'left' 
    # and 'l' are accepted inputs, both lower and upper cases.
    # 
    #   Outputs:
    #   sign_side - sign to adjust reference systems based on body side. Value:
    #   1 for right side, Value: -1 for left side.
    # 
    #   side_low - a single character, lower case body side label that can be 
    #   used in all other STAPLE functions requiring such input.
    # -------------------------------------------------------------------------
    
    side_low = side_raw[0].lower()
    
    if side_low == 'r':
        sign_side = 1
    elif side_low == 'l':
        sign_side = -1
    else:
        print('bodySide2Sign Error: specify right "r" or left "l"')
    
    return sign_side, side_low

# -----------------------------------------------------------------------------
def getBoneLandmarkList(bone_name):
    # -------------------------------------------------------------------------
    # Given a bone name, returns names and description of the bony landmark 
    # identifiable on its surface in a convenient cell array.
    # 
    # Inputs:
    # bone_name - a string indicating a bone of the lower limb
    # 
    # Outputs:
    # LandmarkInfo - cell array containing the name and keywords to identify 
    # the bony landmarks on each bone triangulation. This information can 
    # easily used as input to findLandmarkCoords and landmarkTriGeomBone.
    # -------------------------------------------------------------------------
    
    LandmarkInfo = {}
    
    # used notation to describe the landmarks
    # LandmarkInfo['0'] = BL name
    # LandmarkInfo['1'] = axis
    # LandmarkInfo['2'] = operator (max/min)
    # 1,2 can repeat
    # LandmarkInfo['end'] = proximal/distal (optional)
    
    if bone_name == 'pelvis':
        LandmarkInfo['0'] = ['RASI', 'x', 'max', 'z', 'max']
        LandmarkInfo['1'] = ['LASI', 'x', 'max', 'z', 'min']
        LandmarkInfo['2'] = ['RPSI', 'x', 'min', 'z', 'max']
        LandmarkInfo['3'] = ['LPSI', 'x', 'min', 'z', 'min']
    elif bone_name == 'femur_r':
        LandmarkInfo['0'] = ['RKNE', 'z', 'max', 'distal']
        LandmarkInfo['1'] = ['RMFC', 'z', 'min', 'distal']
        LandmarkInfo['2'] = ['RTRO', 'z', 'max', 'proximal']
    elif bone_name == 'femur_l':
        LandmarkInfo['0'] = ['LKNE', 'z', 'min', 'distal']
        LandmarkInfo['1'] = ['LMFC', 'z', 'max', 'distal']
        LandmarkInfo['2'] = ['LTRO', 'z', 'min', 'proximal']
    elif bone_name == 'tibia_r':
        LandmarkInfo['0'] = ['RTTB', 'x', 'max', 'proximal']
        LandmarkInfo['1'] = ['RHFB', 'z', 'max', 'proximal']
        LandmarkInfo['2'] = ['RANK', 'z', 'max', 'distal']
        LandmarkInfo['3'] = ['RMMA', 'z', 'min', 'distal']
    elif bone_name == 'tibia_l':
        LandmarkInfo['0'] = ['LTTB', 'x', 'max', 'proximal']
        LandmarkInfo['1'] = ['LHFB', 'z', 'max', 'proximal']
        LandmarkInfo['2'] = ['LANK', 'z', 'max', 'distal']
        LandmarkInfo['3'] = ['LMMA', 'z', 'min', 'distal']
    elif bone_name == 'patella_r':
        LandmarkInfo['0'] = ['RLOW', 'y', 'min', 'distal']
    elif bone_name == 'patella_l':
        LandmarkInfo['0'] = ['LLOW', 'y', 'min', 'distal']
    elif bone_name == 'calcn_r':
        LandmarkInfo['0'] = ['RHEE', 'x', 'min']
        LandmarkInfo['1'] = ['RD5M', 'z', 'max']
        LandmarkInfo['2'] = ['RD1M', 'z', 'min']
    elif bone_name == 'calcn_l':
        LandmarkInfo['0'] = ['LHEE', 'x', 'min']
        LandmarkInfo['1'] = ['LD5M', 'z', 'min']
        LandmarkInfo['2'] = ['LD1M', 'z', 'max']
    # included for advanced example on humerus
    elif bone_name == 'humerus_r':
        LandmarkInfo['0'] = ['RLE', 'z', 'max', 'distal']
        LandmarkInfo['1'] = ['RME', 'z', 'min', 'distal']
    elif bone_name == 'humerus_l':
        LandmarkInfo['0'] = ['LLE', 'z', 'min', 'distal']
        LandmarkInfo['1'] = ['LME', 'z', 'max', 'distal']
    else:
        # loggin.error('getBoneLandmarkList.m specified bone name is not supported yet.')
        print('getBoneLandmarkList.m specified bone name is not supported yet.')
    
    return LandmarkInfo
    
# -----------------------------------------------------------------------------
def findLandmarkCoords(points, axis_name, operator):
    # -------------------------------------------------------------------------
    # Pick points from a point cloud using string keywords.
    # 
    # Inputs:
    # points - indicates a point cloud, i.e. a matrix of [N_point, 3] dimensions.
    # 
    # axis_name - specifies the axis where the min/max operator will be applied.
    # Assumes values: 'x', 'y' or 'z', corresponding to first, second or third
    # column of the points input.
    # 
    # operator - specifies which point to pick in the specified axis.
    # Currently assumes values: 'min' or 'max', meaning smalles or
    # largest value in that direction.
    # 
    # Outputs:
    # BL - coordinates of the point identified in the point cloud following
    # the directions provided as input. 
    # 
    # BL_ind - index of the point identified in the point cloud following the
    # direction provided as input.
    # 
    # Example:
    # getBonyLandmark(pcloud,'max','z') asks to pick the point in the pcloud 
    # with largest z coordinate. If pcloud was a right distal femur triangulation
    # with points expressed in the reference system of the International
    # Society of Biomechanics, this could be the lateral femoral epicondyle.
    # -------------------------------------------------------------------------
    
    # interpreting direction of searchv
    if axis_name == 'x':
        dir_ind = 0
    elif axis_name == 'y':
        dir_ind = 1
    elif axis_name == 'z':
        dir_ind = 2
    else:
        print('axis_name no identified')
    
    # interpreting description of search
    if operator == 'max':
        BL_ind = np.argmax(points[:,dir_ind])
    elif operator == 'min':
        BL_ind = np.argmin(points[:,dir_ind])
    else:
        print('operator no identified')
    
    BL = points[BL_ind]
    BL = np.reshape(BL,(BL.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        
    return BL, BL_ind
    
# -----------------------------------------------------------------------------
def landmarkBoneGeom(TriObj, CS, bone_name, debug_plots = 0):
    # -------------------------------------------------------------------------
    # Locate points on the surface of triangulation objects that are bone 
    # geometries.
    # 
    # Inputs:
    # TriObj - a triangulation object.
    # 
    # CS - a structure representing a coordinate systems as a structure with 
    # 'Origin' and 'V' fields, indicating the origin and the axis. 
    # In 'V', each column is the normalise direction of an axis 
    # expressed in the global reference system (CS.V = [x, y, z]).
    # 
    # bone_name - string identifying a lower limb bone. Valid values are:
    # 'pelvis', 'femur_r', 'tibia_r', 'patella_r', 'calcn_r'.
    # 
    # debug_plots - takes value 1 or 0. Plots the steps of the landmarking
    # process. Switched off by default, useful for debugging.
    # 
    # Outputs:
    # Landmarks - structure with as many fields as the landmarks defined in
    # getBoneLandmarkList.m. Each field has the name of the landmark and
    # value the 3D coordinates of the landmark point.
    # -------------------------------------------------------------------------
    # get info about the landmarks in the current bone
    LandmarkDict = getBoneLandmarkList(bone_name)

    # change reference system to bone/body reference system
    TriObj_in_CS, _, _ = TriChangeCS(TriObj, CS['V'], CS['CenterVol'])

    # visualise the bone in the bone/body ref system
    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        
        # create coordinate system centred in [0, 0 ,0] and with unitary axis
        # directly along the direction of the ground ref system
        LocalCS = {}
        LocalCS['Origin'] = np.array([0,0,0])
        LocalCS['V'] = np.eye(3)
        
        plotTriangLight(TriObj_in_CS, LocalCS, ax)
        quickPlotRefSystem(LocalCS, ax)

    # get points from the triangulation
    TriPoints = TriObj_in_CS['Points']

    # define proximal or distal:
    # 1) considers Y the longitudinal direction
    # 2) dividing the bone in three parts
    ub = np.max(TriPoints[:,1])*0.3
    lb = np.max(TriPoints[:,1])*(-0.3)

    if debug_plots:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.set_title('proximal (blue) and distal (red) geom')
        # check proximal geometry
        check_proximal = TriPoints[TriPoints[:,1] > ub]
        for p in check_proximal:
            ax.scatter(p[0], p[1], p[2], color = 'blue')
        
        # check distal geometry
        check_distal = TriPoints[TriPoints[:,1] < lb]
        for p in check_distal:
            ax.scatter(p[0], p[1], p[2], color = 'red')

    Landmarks = {}
    for cur_BL_info in LandmarkDict.values():
        # extract info (see LandmarkStruct for details)
        cur_BL_name = cur_BL_info[0] # landmark name
        cur_axis = cur_BL_info[1] # x/y/z
        cur_operator = cur_BL_info[2] # min/max
        cur_bone_extremity = cur_BL_info[-1]
        
        # is the BL proximal or distal?
        if cur_bone_extremity == 'proximal':
            # identify the landmark
            local_BL, _ = findLandmarkCoords(TriPoints[TriPoints[:,1] > ub], cur_axis, cur_operator)
        elif cur_bone_extremity == 'distal':
            # identify the landmark
            local_BL, _ = findLandmarkCoords(TriPoints[TriPoints[:,1] < lb], cur_axis, cur_operator)
        else:
            # get landmark if no bone extremity is specified
            local_BL, _ = findLandmarkCoords(TriPoints, cur_axis, cur_operator)
        
        # save a landmark structure (transform back to global)
        Landmarks[cur_BL_name] = CS['CenterVol'] + np.dot(CS['V'],local_BL)
        
    return Landmarks

# -----------------------------------------------------------------------------
def processTriGeomBoneSet(triGeomBoneSet, side_raw = '', algo_pelvis = 'STAPLE', algo_femur = 'GIBOC-cylinder', algo_tibia = 'Kai2014', result_plots = 0, debug_plots = 0, in_mm = 1):
    # -------------------------------------------------------------------------
    #  Compute parameters of the lower limb joints associated with the bone 
    # geometries provided as input through a set of triangulations dictionary.
    # Note that:
    # 1) This function does not produce a complete set of joint parameters but 
    # only those available through geometrical analyses, that are:
    #       * from pelvis: ground_pelvis_child
    #       * from femur : hip_child  // knee_parent
    #       * from tibia : knee_child** 
    #       * from talus : ankle_child // subtalar parent
    # other functions then complete the information required to generate a MSK
    # model, e.g. use ankle child location and ankle axis to define an ankle
    # parent reference system etc.
    # ** note that the tibia geometry was not used to define the knee child
    # anatomical coord system in the approach of Modenese et al. JB 2018.
    # 2) Bony landmarks are identified on all bones except the talus.
    # 3) Body-fixed Cartesian coordinate system are defined but not employed in
    # the construction of the models.
    # 
    # Inputs:
    #   geom_set - a set of triangulation dictionary, normally created
    #   using the function createTriGeomSet. See that function for more
    #   details.
    # 
    #   side_raw - generic string identifying a body side. 'right', 'r', 'left' 
    #   and 'l' are accepted inputs, both lower and upper cases.
    # 
    #   algo_pelvis - the algorithm selected to process the pelvis geometry.
    # 
    #   algo_femur - the algorithm selected to process the femur geometry.
    # 
    #   algo_tibia - the algorithm selected to process the tibial geometry.
    # 
    #   result_plots - enable plots of final fittings and reference systems. 
    #   Value: 1 (default) or 0.
    # 
    #   debug_plots - enable plots used in debugging. Value: 1 or 0 (default). 
    # 
    #   in_mm - (optional) indicates if the provided geometries are given in mm
    #   (value: 1) or m (value: 0). Please note that all tests and analyses
    #   done so far were performed on geometries expressed in mm, so this
    #   option is more a placeholder for future adjustments.
    # 
    # Outputs:
    #   JCS - dictionary collecting the parameters of the joint coordinate
    #   systems computed on the bone triangulations.
    # 
    #   BL - dictionary collecting the bone landmarks identified on the
    #   three-dimensional bone surfaces.
    # 
    #   BCS - dictionary collecting the body coordinate systems of the 
    #   processed bones.
    # -------------------------------------------------------------------------
    
    # setting defaults
    
    if side_raw == '':
        side = inferBodySideFromAnatomicStruct(triGeomBoneSet)
    else:
        # get sign correspondent to body side
        _, side = bodySide2Sign(side_raw)
    
    # names of the segments
    femur_name = 'femur_' + side
    tibia_name = 'tibia_' + side
    patella_name = 'patella_' + side
    talus_name = 'talus_' + side
    calcn_name = 'calcn_' + side 
    toes_name  = 'toes_' + side
    
    BCS = {}
    JCS = {}
    BL = {}
    
    print('-----------------------------------')
    print('Processing provided bone geometries')
    print('-----------------------------------')
    
    # visualization of the algorithms that will be used (depends on available
    # segments
    print('ALGORITHMS:')
    
    if 'pelvis' in triGeomBoneSet or 'pelvis_no_sacrum' in triGeomBoneSet:
        BCS['pelvis'] = {}
        JCS['pelvis'] = {}
        BL['pelvis'] = {}
        print('  pelvis: ', algo_pelvis)
    if femur_name in triGeomBoneSet:
        BCS[femur_name] = {}
        JCS[femur_name] = {}
        BL[femur_name] = {}
        print('  femur: ', algo_femur)
    if tibia_name in triGeomBoneSet:
        BCS[tibia_name] = {}
        JCS[tibia_name] = {}
        BL[tibia_name] = {}
        print('  tibia: ', algo_tibia)
    if patella_name in triGeomBoneSet:
    #     print('  patella: ', algo_patella)
        print('  patella: ', 'N/A')
    if talus_name in triGeomBoneSet:
        BCS[talus_name] = {}
        JCS[talus_name] = {}
        BL[talus_name] = {}
        print('  talus: ', 'STAPLE')
    if calcn_name in triGeomBoneSet:
        BCS[calcn_name] = {}
        JCS[calcn_name] = {}
        BL[calcn_name] = {}
        print('  foot: ', 'STAPLE')
    if toes_name in triGeomBoneSet:
        print('  toes: ', 'N/A')
        
    # ---- PELVIS -----
    if 'pelvis' in triGeomBoneSet:
        if algo_pelvis == 'STAPLE':
            BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
                STAPLE_pelvis(triGeomBoneSet['pelvis'], side, result_plots, debug_plots, in_mm)
    #     if algo_pelvis == 'Kai2014':
    #         # BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    #         #     Kai2014_pelvis(triGeomBoneSet['pelvis'], side, result_plots, debug_plots, in_mm)
    elif 'pelvis_no_sacrum' in triGeomBoneSet:
        if algo_pelvis == 'STAPLE':
            BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
                STAPLE_pelvis(triGeomBoneSet['pelvis_no_sacrum'], side, result_plots, debug_plots, in_mm)
    #     if algo_pelvis == 'Kai2014':
    #         # BCS['pelvis'], JCS['pelvis'], BL['pelvis']  = \
    #         #     Kai2014_pelvis(triGeomBoneSet['pelvis_no_sacrum'], side, result_plots, debug_plots, in_mm)
    
    # ---- FEMUR -----
    if femur_name in triGeomBoneSet:
        if 'GIBOC' in triGeomBoneSet:
            BCS[femur_name], JCS[femur_name], BL[femur_name], _, _ = \
                GIBOC_femur(triGeomBoneSet[femur_name], side, algo_femur[6:], result_plots, debug_plots, in_mm)
        # if 'Miranda' in triGeomBoneSet:
        #     BCS[femur_name], JCS[femur_name], BL[femur_name] = \
        #         Miranda2010_buildfACS(triGeomBoneSet['femur_name'])
        # if 'Kai2014' in triGeomBoneSet:
        #     BCS[tibia_name], JCS[tibia_name], BL[tibia_name] = \
        #         Kai2014_femur(triGeomBoneSet['femur_name'], side)
        else:
            BCS[femur_name], JCS[femur_name], BL[femur_name], _, _ = \
                GIBOC_femur(triGeomBoneSet[femur_name], side, algo_femur[6:], result_plots, debug_plots, in_mm)
    
    # ---- TIBIA -----
    if tibia_name in triGeomBoneSet:
        # if 'GIBOC' in triGeomBoneSet:
        #     BCS[tibia_name], JCS[tibia_name], BL[tibia_name] = \
        #         GIBOC_tibia(triGeomBoneSet['tibia_name'], side, triGeomBoneSet[6:], result_plots, debug_plots, in_mm)
        if 'Kai2014' in triGeomBoneSet:
            BCS[tibia_name], JCS[tibia_name], BL[tibia_name], _ = \
                Kai2014_tibia(triGeomBoneSet[tibia_name], side, result_plots, debug_plots, in_mm)
        else:
            BCS[tibia_name], JCS[tibia_name], BL[tibia_name], _ = \
                Kai2014_tibia(triGeomBoneSet[tibia_name], side, result_plots, debug_plots, in_mm)
    

    return JCS, BL, BCS

# -----------------------------------------------------------------------------
def compileListOfJointsInJCSStruct(JCS = {}):
    # -------------------------------------------------------------------------
    # Create a list of joints that can be modelled from the structure that 
    # results from morphological analysis.
    # 
    # Inputs:
    # JCS - dictionary with the joint parameters produced by the morphological 
    # analyses of processTriGeomBoneSet. Not all listed joints are
    # actually modellable, in the sense that the parent and child
    # reference systems might not be present, the model might be incomplete etc.
    # 
    # Outputs:
    # joint_list - a list of unique elements. Each element is the name of a 
    # joint present in the JCS structure.
    # -------------------------------------------------------------------------
    joint_list = []
    
    for body in JCS:
        joint_list += [joint for joint in JCS[body] if joint not in joint_list]

    return joint_list

# -----------------------------------------------------------------------------
def jointDefinitions_auto2020(JCS = {}, jointStruct = {}):
    # -------------------------------------------------------------------------
    # Define the orientation of the parent reference system in the ankle joint 
    # using the ankle axis as Z axis and the long axis of the tibia 
    # (made perpendicular to Z) as Y axis. X defined by cross-product. The 
    # ankle is the only joint that is not defined neither in parent or child 
    # in the "default" joint definition named 'auto2020'. 
    # 
    # Inputs:
    # JCS - dictionary with the joint parameters produced by the morphological 
    # analyses of processTriGeomBoneSet. Not all listed joints are
    # actually modellable, in the sense that the parent and child
    # reference systems might not be present, the model might be incomplete etc.
    # In this function only the field `V` relative to talus and tibia will be 
    # recalled.
    # 
    # jointStruct - Dictionary including all the reference parameters that will
    # be used to generate an OpenSim JointSet.
    # 
    # Outputs:
    # jointStruct - updated dicttionary with a newly defined ankle joint parent
    # ankle V.
    # -------------------------------------------------------------------------
    
    side_low = inferBodySideFromAnatomicStruct(JCS)
    
    # bone names
    tibia_name = 'tibia_' + side_low
    talus_name = 'talus_' + side_low
    
    # joint names
    ankle_name = 'ankle_' + side_low
    knee_name = 'knee_' + side_low
    
    # joint params: JCS[bone_name] will access the geometrical information
    # from the morphological analysis
    
    # take Z from ankle joint (axis of rotation)
    if talus_name in JCS and tibia_name in JCS:
        Zpar = JCS[talus_name][ankle_name]['V'][:,2]
        Zpar = np.reshape(Zpar,(Zpar.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        Ytemp = JCS[tibia_name][knee_name]['V'][:,1]
        Ytemp = np.reshape(Ytemp,(Ytemp.size, 1)) # convert 1d (3,) to 2d (3,1) vector
        
        # Y and Z orthogonal
        Ypar = preprocessing.normalize(Ytemp - Zpar*np.dot(Zpar, Ytemp), axis =0)
        Xpar = preprocessing.normalize(np.cross(Ypar, Zpar), axis =0)
        
        # assigning pose matrix and parent orientation
        jointStruct[ankle_name] = {}
        jointStruct[ankle_name]['V'] = np.zeros((3,3))
        jointStruct[ankle_name]['V'][:,0] = Xpar[:,0]
        jointStruct[ankle_name]['V'][:,1] = Ypar[:,0]
        jointStruct[ankle_name]['V'][:,2] = Zpar[:,0]
        
        jointStruct[ankle_name]['parent_orientation'] = computeXYZAngleSeq(jointStruct[ankle_name]['V'])
    else:
        return jointStruct
    
    return jointStruct

# -----------------------------------------------------------------------------
def jointDefinitions_Modenese2018(JCS = {}, jointStruct = {}):
    # -------------------------------------------------------------------------
    # Define the orientation of lower limb joints as in Modenese et al. 
    # JBiomech 2018, 17;73:108-118 https://doi.org/10.1016/j.jbiomech.2018.03.039
    # Required for the comparisons presented in Modenese and Renault, JBiomech
    # 2020. 
    # 
    # Inputs:
    # JCS - dictionary with the joint parameters produced by the morphological 
    # analyses of processTriGeomBoneSet. Not all listed joints are
    # actually modellable, in the sense that the parent and child
    # reference systems might not be present, the model might be incomplete etc.
    # 
    # jointStruct - Dictionary including all the reference parameters that will
    # be used to generate an OpenSim CustomJoints.
    # 
    # Outputs:
    # jointStruct - updated jointStruct with the joints defined as in
    # Modenese2018, rather than connected directly using the joint
    # coordinate system computed in processTriGeomBoneSet.py plus the
    # default joint definition available in jointDefinitions_auto2020.py
    # -------------------------------------------------------------------------
    
    side_low = inferBodySideFromAnatomicStruct(JCS)
    
    # joint names
    knee_name = 'knee_' + side_low
    ankle_name = 'ankle_' + side_low
    subtalar_name = 'subtalar_' + side_low
    
    # segments names
    femur_name = 'femur_' + side_low
    tibia_name = 'tibia_' + side_low
    talus_name = 'talus_' + side_low
    calcn_name = 'calcn_' + side_low
    talus_name = 'talus_' + side_low
    mtp_name = 'mtp_' + side_low
    
    if talus_name in JCS:
        TalusDict = JCS[talus_name]
        
        if tibia_name in JCS:
            TibiaDict = JCS[tibia_name]
            
            if femur_name in JCS:
                FemurDict = JCS[femur_name]
                
                # knee child orientation
                # ---------------------
                # Z aligned like the medio-lateral femoral joint, e.g. axis of cylinder
                # Y aligned with the tibial axis (v betw origin of ankle and knee)
                # X from cross product
                # ---------------------
                
                # take Z from knee joint (axis of rotation)
                Zparent  = FemurDict[knee_name]['V'][:,2]
                Zparent = np.reshape(Zparent,(Zparent.size, 1)) # convert 1d (3,) to 2d (3,1) vector
                # take line joining talus and knee centres
                TibiaDict[knee_name]['Origin'] = FemurDict[knee_name]['Origin']
                # vertical axis joining knee and ankle joint centres (same used for ankle
                # parent)
                Ytemp = (TibiaDict[knee_name]['Origin'] - FemurDict[ankle_name]['Origin'])
                Ytemp /= np.linalg.norm(TibiaDict[knee_name]['Origin'] - FemurDict[ankle_name]['Origin'], axis=0)
                # make Y and Z orthogonal
                Yparent = preprocessing.normalize(Ytemp - Zparent*np.dot(Zparent, Ytemp), axis =0)
                Xparent = preprocessing.normalize(np.cross(Ytemp, Zparent), axis =0)
                # assigning pose matrix and child orientation
                tmp_V = np.zeros((3,3))
                tmp_V[:,0] = Xparent[:,0]
                tmp_V[:,1] = Yparent[:,0]
                tmp_V[:,2] = Zparent[:,0]
                
                jointStruct[knee_name]['child_orientation'] = computeXYZAngleSeq(tmp_V)
                
            # Ankle parent orientation
            # ---------------------
            # Z aligned like the cilinder of the talar throclear
            # Y aligned with the tibial axis (v betw origin of ankle and knee)
            # X from cross product
            # ---------------------
            # take Z from ankle joint (axis of rotation)
            Zparent = TalusDict[ankle_name]['V'][:,2]
            Zparent = np.reshape(Zparent,(Zparent.size, 1)) # convert 1d (3,) to 2d (3,1) vector
            # take line joining talus and knee centres
            Ytibia = (TibiaDict[knee_name]['Origin'] - TalusDict[ankle_name]['Origin'])
            Ytibia /= np.linalg.norm(TibiaDict[knee_name]['Origin'] - TalusDict[ankle_name]['Origin'], axis=0)
            # make Y and Z orthogonal
            Yparent = preprocessing.normalize(Ytibia - Zparent*np.dot(Zparent, Ytibia), axis =0)
            Xparent = preprocessing.normalize(np.cross(Ytibia, Zparent), axis =0)
            # assigning pose matrix and child orientation
            tmp_V = np.zeros((3,3))
            tmp_V = Xparent[:,0]
            tmp_V = Yparent[:,0]
            tmp_V = Zparent[:,0]
            
            jointStruct[ankle_name]['parent_orientation'] = computeXYZAngleSeq(tmp_V)
            
        # Ankle child orientation
        # ---------------------
        # Z aligned like the cilinder of the talar throclear
        # X like calcaneus, but perpendicular to Z
        # Y from cross product
        # ---------------------
        if calcn_name in JCS:
            CalcnDict = JCS[calcn_name]
            
            # take Z from ankle joint (axis of rotation)
            Zchild = TalusDict[ankle_name]['V'][:,2]
            # take X ant-post axis of the calcaneus
            Xtemp = CalcnDict[mtp_name]['V'][:,0]
            # make X and Z orthogonal
            Xchild = preprocessing.normalize(Xtemp - Zchild*np.dot(Zchild, Xtemp), axis =0)
            Ychild = preprocessing.normalize(np.cross(Zchild, Xtemp), axis =0)
            # assigning pose matrix and child orientation
            tmp_V = np.zeros((3,3))
            tmp_V[:,0] = Xchild[:,0]
            tmp_V[:,1] = Ychild[:,0]
            tmp_V[:,2] = Zchild[:,0]
            
            jointStruct[ankle_name]['child_orientation'] = computeXYZAngleSeq(tmp_V)
            
    # Ankle child orientation
    # ---------------------
    # Z is the subtalar axis of rotation
    # Y from centre of subtalar joint points to femur joint centre
    # X from cross product
    # ---------------------
    if femur_name in JCS:
        # needs to be initialized?
        FemurDict = JCS[femur_name]
        
        # take Z from ankle joint (axis of rotation)
        Zparent = TalusDict[subtalar_name]['V'][:,2]
        # take Y pointing to the knee joint centre
        Ytemp = (FemurDict[knee_name]['parent_location'] - TalusDict[subtalar_name]['parent_location'])
        Ytemp /= np.linalg.norm(FemurDict[knee_name]['parent_location'] - TalusDict[ankle_name]['parent_location'], axis=0)
        # make Y and Z orthogonal
        Yparent = preprocessing.normalize(Ytemp - Zparent*np.dot(Zparent, Ytemp), axis =0)
        Xparent = preprocessing.normalize(np.cross(Yparent, Zparent), axis =0)
        # assigning pose matrix and child orientation
        tmp_V = np.zeros((3,3))
        tmp_V[:,0] = Xparent[:,0]
        tmp_V[:,1] = Yparent[:,0]
        tmp_V[:,2] = Zparent[:,0]
        
        jointStruct[subtalar_name]['parent_orientation'] = computeXYZAngleSeq(tmp_V)
    
    return jointStruct

# -----------------------------------------------------------------------------
def inferBodySideFromAnatomicStruct(anat_struct):
    # -------------------------------------------------------------------------   
    # Infer the body side that the user wants to process based on a structure 
    # containing the anatomical objects (triangulations or joint definitions) 
    # given as input. The implemented logic is trivial: the fields are checked 
    # for standard names of bones and joints used in OpenSim models.
    # 
    # Inputs:
    # anat_struct - a dictionary containing anatomical objects, e.g. a set of 
    # bone triangulation or joint definitions.
    # 
    # Outputs:
    # guessed_side - a body side label that can be used in all other STAPLE
    # functions requiring such input.
    # -------------------------------------------------------------------------
    guessed_side = ''
    
    if isinstance(anat_struct, dict):
        fields_side = list(anat_struct.keys())
    else:
        print('inferBodySideFromAnatomicStruct.py  Input must be dictionary.')
        logging.error('inferBodySideFromAnatomicStruct.py  Input must be dictionary.')
        return 0 
    
    # check using the body names
    body_set = ['femur', 'tibia', 'talus', 'calcn']
    guess_side_b = [fs[-1] for b in body_set for fs in fields_side if b in fs]
    
    # check using the joint names
    joint_set = ['hip', 'knee', 'ankle', 'subtalar']
    guess_side_j = [fs[-1] for j in joint_set for fs in fields_side if j in fs]
    
    # composed list
    combined_guessed = guess_side_b + guess_side_j
    
    if all(i == 'r' for i in combined_guessed):
        guessed_side = 'r'
    elif all(i == 'l' for i in combined_guessed):
        guessed_side = 'l'
    else:
        print('guessBodySideFromAnatomicStruct.py Error: it was not possible to infer the body side. Please specify it manually in this occurrance.')
        logging.error('guessBodySideFromAnatomicStruct.py Error: it was not possible to infer the body side. Please specify it manually in this occurrance.')
        
    return guessed_side

#%%
# ##################################################
# OPENSIM ##########################################
# ##################################################
# 
def writeModelGeometriesFolder(aTriGeomBoneSet, aGeomFolder = '.', aFileFormat = 'obj', coeffFaceReduc = 0.3):
    # -------------------------------------------------------------------------
    # Write bone geometry for an automated model in the specified geometry 
    # folder using a user-defined file format.
    #
    # Inputs:
    #    aTriGeomBoneSet - a set of triangulation dict generated using the
    #        createTriGeomSet function.
    #
    #    aGeomFolder - the folder where to save the geometries of the automatic
    #        OpenSim model.
    #
    #    aFileFormat - the format to use when writing the bone geometry files.
    #       Currently 'stl' files and Waterfront 'obj' files are supported.
    #       Note that both formats are ASCII, as required by the OpenSim
    #       visualiser.
    #
    #   coeffFaceReduc - number between 0 and 1 indicating the ratio of faces
    #       that the final geometry will have. Default value 0.3, meaning that
    #       the geometry files will have 30% of the faces of the original bone
    #       geometries. This is required for faster visualization in OpenSim.
    # 
    # Outputs:
    #   none - the bone geometries are saved in the specified folder using the
    #       specified file format.
    # -------------------------------------------------------------------------
        
    # create geometry file if not existing
    dst = Path(aGeomFolder)
    dst.mkdir(parents=True, exist_ok=True)
    dst = str(dst)
        
    # ensure lower case fileformat
    aFileFormat = aFileFormat.lower()
    
    # how many bones to print?
    bone_names = list(aTriGeomBoneSet.keys())
    
    print('-------------------------------------')
    print('Writing geometries for visualization ')
    print('-------------------------------------')
    
    for curr_bone_name in bone_names:
        
        curr_tri = aTriGeomBoneSet[curr_bone_name]
    
        # reduce number of faces in geometry
        points_out, faces_out = fast_simplification.simplify(curr_tri['Points'], curr_tri['ConnectivityList'], 1-coeffFaceReduc) 
        
        if aFileFormat == 'obj':
            
            logging.exception('writeModelGeometriesFolder Error: Please write model geometries as .STL files (preferred).')
            
        elif aFileFormat == 'stl':
            
            new_mesh = mesh.Mesh(np.zeros(faces_out.shape[0], dtype=mesh.Mesh.dtype))
            for i, f in enumerate(faces_out):
                for j in range(3):
                    new_mesh.vectors[i][j] = points_out[f[j],:]

            # Write the mesh to file 
            new_mesh.save(os.path.join(aGeomFolder, curr_bone_name + '.stl'))
            
        else:
            
            logging.exception('writeModelGeometriesFolder Error: Please specify a file format to write the model geometries between ''stl'' and ''obj''.')
    
    # inform the user
    print('Stored ' + aFileFormat + ' files in folder ' + aGeomFolder)
    
    return 0

# -----------------------------------------------------------------------------
def computeXYZAngleSeq(aRotMat):
    # -------------------------------------------------------------------------
    # Convert a rotation matrix in the orientation vector used in OpenSim 
    # (X-Y-Z axes rotation order).
    # 
    # Inputs:
    # aRotMat - a rotation matrix, normally obtained writing as columns the
    # axes of the body reference system, expressed in global reference
    # system.
    # 
    # Outputs:
    # orientation - the sequence of angles used in OpenSim to define the
    # joint orientation. Sequence of rotation is X-Y-Z.
    # -------------------------------------------------------------------------    
    orientation = np.zeros((1,3))
    
    # fixed body sequence of angles from rot mat usable for orientation in OpenSim
    beta = np.arctan2(aRotMat[0,2], np.sqrt(aRotMat[0,0]**2 + aRotMat[0,1]**2))
    alpha = np.arctan2(-aRotMat[1,2]/np.cos(beta), aRotMat[2,2]/np.cos(beta))
    gamma = np.arctan2(-aRotMat[0,1]/np.cos(beta), aRotMat[0,0]/np.cos(beta))
    
    # build a vector
    orientation[0,0] = beta
    orientation[0,1] = alpha
    orientation[0,2] = gamma
    
    return orientation




#%% OpenSim functions

# -----------------------------------------------------------------------------
def initializeOpenSimModel(aModelNameString):
    # -------------------------------------------------------------------------
    # Create barebone of OpenSim model adding name, gravity and credits for the
    # automatic models.
    # 
    # Inputs:
    # aModelNameString - a string that will be set as name of the automatic model.
    # 
    # Outputs:
    # osimModel - an OpenSim model to use as basis for the automatic modelling.
    # It has only a name, includes gravity and has credits.
    # -------------------------------------------------------------------------
    
    print('-------------------------------------')
    print('Initializing automatic OpenSim model:')
    
    # create the model
    osimModel = opensim.Model()
    
    # set gravity
    osimModel.setGravity(opensim.Vec3(0, -9.8081, 0))
    
    # set model name
    osimModel.setName(aModelNameString)
    
    # set credits
    osimModel.set_credits('Luca Modenese, Jean-Baptiste Renault 2020. Model created using the STAPLE (Shared Tools for Automatic Personalised Lower Extremity) modelling toolbox. GitHub page: https://github.com/modenaxe/msk-STAPLE.')
    
    return osimModel

# -----------------------------------------------------------------------------
def addBodiesFromTriGeomBoneSet(osimModel = '', geom_set = {}, vis_geom_folder = '', vis_geom_format = 'obj', body_density = 1420*0.001**3, in_mm = 1):
    # -------------------------------------------------------------------------
    # Create a body for each triangulation object in the provided geom_set 
    # structure. These bodies are added to the specified OpenSim model. A 
    # density for computing the mass properties and a details of the 
    # visualization folder can also be provided.
    # NOTE: the added bodies are not yet connected by appropriate joints, so
    # unless this function is used in a workflow including 
    # createLowerLimbJoints or joint definition, the resulting OpenSim model
    # will consist of bodies connected to ground.
    # 
    # Inputs:
    # osimModel - an OpenSim model to which the bodies created from
    # triangulation objects will be added.
    # 
    # geom_set - a set of Dict triangulation objects, normally created
    # using the function createTriGeomSet. See that function for more details.
    # 
    # vis_geom_folder - the folder where the geometry files used to
    # visualised the OpenSim model will be stored.
    # 
    # vis_geom_format - the format used to write the geometry files employed
    # by the OpenSim model for visualization.
    # 
    # body_density - (optional) the density assigned to the triangulation
    # objects when computing the mass properties. Default value is 1420
    # Kg/m^3, which is the density assigned to bone in Dumas et al. 
    # IEEE Transactions on Biomedical engineering (2005). In the
    # generation of automatic lower extremity models this value is
    # overwritten by mass properties estimated through regression
    # equations. The purpose of computing them is to provide a reasonable
    # first estimate
    # 
    # in_mm - (optional) indicates if the provided geometries are given in mm
    # (value: 1) or m (value: 0). Please note that all tests and analyses
    # done so far were performed on geometries expressed in mm, so this
    # option is more a placeholder for future adjustments.
    # 
    # Outputs:
    # osimModel - the OpenSim model provided as input updated to include the
    # bodies defined from the triangulation objects.
    # -------------------------------------------------------------------------
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1
    
    # bone density by default (Dumas 2005)
    body_density = 1420*dim_fact**3
    
    # add the individual bodies to the model
    print('-------------------------------------')
    print('Adding ' + str(len(geom_set)) + ' bodies to the OpenSim model')
    
    for cur_body_name, cur_geom in geom_set.items():
        # geometry file used for visualisation
        cur_vis_geom_file = os.path.join(vis_geom_folder, cur_body_name + '.' + vis_geom_format)
        
        if cur_body_name == 'pelvis_no_sacrum':
            cur_body_name == 'pelvis'
        
        print('     ' + 'i) ' + cur_body_name)
        
        # creating the body and adding it to the OpenSim model
        addBodyFromTriGeomObj(osimModel, cur_geom, cur_body_name, cur_vis_geom_file, body_density, in_mm)
        print('      ADDED')
        print('      -----')

    return osimModel

# -----------------------------------------------------------------------------
def addBodyFromTriGeomObj(osimModel = '', triGeom = {}, body_name = '', vis_mesh_file = '', body_density = 1420*0.001**3, in_mm = 1):
    # -------------------------------------------------------------------------
    # Create an OpenSim Body from a MATLAB triangulation object. 
    # Requires a name.
    # 
    # Inputs:
    # osimModel - an OpenSim model to which the bodies created from
    # triangulation objects will be added.
    # 
    # triGeom - a set of Dict triangulation objects.
    # 
    # body_name - a string indicating the name to assign to the OpenSim body.
    # 
    # vis_mesh_file - the path of the visualisation file that will be assigned 
    # to the OpenSim body. For automatic models, this will be an object file 
    # generated by writeModelGeometriesFolder.
    # 
    # body_density - (optional) the density assigned to the triangulation
    # objects when computing the mass properties. Default value is 1420
    # Kg/m^3, which is the density assigned to bone in Dumas et al. 
    # IEEE Transactions on Biomedical engineering (2005). In the
    # generation of automatic lower extremity models this value is
    # overwritten by mass properties estimated through regression
    # equations. The purpose of computing them is to provide a reasonable
    # first estimate
    # 
    # in_mm - (optional) indicates if the provided geometries are given in mm
    # (value: 1) or m (value: 0). Please note that all tests and analyses
    # done so far were performed on geometries expressed in mm, so this
    # option is more a placeholder for future adjustments.
    # 
    # Outputs:
    # osimModel - the OpenSim model provided as input updated to include the
    # bodies defined from the triangulation objects.
    # -------------------------------------------------------------------------
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1
    
    # bone density by default (Dumas 2005)
    body_density = 1420*dim_fact**3
    
    # compute mass properties
    # NOTE: when creating automated models this initial mass properties are
    # overwritten by mapping proper segment inertial properties and scaling
    # them.
    boneMassProps = computeMassProperties_Mirtich1996(triGeom)
    bone_mass = boneMassProps['mass']*body_density # vol [mm^3] * density [kg/mm^3] 
    bone_COP = boneMassProps['COM']*dim_fact 
    bone_inertia = boneMassProps['Ivec']*body_density*dim_fact**2 # Ivec [mm^3 * mm^2] * density [kg/mm^3] 
    
    # create opensim body
    osim_body = opensim.Body(body_name, bone_mass, \
                             opensim.ArrayDouble.createVec3(bone_COP[0][0], bone_COP[1][0], bone_COP[2][0]), \
                             opensim.Inertia(bone_inertia[0], bone_inertia[1], bone_inertia[2], \
                                             bone_inertia[3], bone_inertia[4], bone_inertia[5]))
    # add body to model
    osimModel.addBody(osim_body)
    
    # add visualization mesh
    if vis_mesh_file != '':
        # vis_geom = opensim.Mesh(vis_mesh_file)
        vis_geom = opensim.Mesh(body_name + '_geom')
        vis_geom.set_scale_factors(opensim.Vec3(dim_fact))
        vis_geom.set_mesh_file(vis_mesh_file)
        vis_geom.setOpacity(1)
        vis_geom.setColor(opensim.Vec3(1, 1, 1))
        osimModel.getBodySet().get(body_name).attachGeometry(vis_geom)
        # osimModel.attachGeometry(vis_geom)
       
    return osimModel

# -----------------------------------------------------------------------------
def getJointParams(joint_name, root_body = 'root_body'):
    # -------------------------------------------------------------------------
    # Assemble a structure with all the information required to create a 
    # CustomJoint of a specified lower limb joint. Normally this function is 
    # used after the geometrical analyses on the bones and before generating 
    # the joint for the automatic OpenSim model using createCustomJointFromStruct. 
    # It is assumed that the inputs will contain enough information 
    # (location and orientation) to define the joint reference system.
    # NOTE: 
    # A body is connected to ground with a free_to_ground joint if no other 
    # specifics are provided, please see examples on partial models for 
    # practical examples.
    # IMPORTANT: modifying the values of the fields of JointParamsStruct output
    # structure allows to modify the joint model according to the preferences
    # of the researcher. See advanced examples.
    # 
    # Inputs:
    # joint_name - name of the lower limb joint for which we want to create
    # the structure containing all parameters (string).
    # 
    # root_body - a string specifying the name of the body attached to ground
    # in the case it is not the default (pelvis).
    # 
    # Outputs:
    # JointParamsStruct - a structure collecting all the information required
    # to define an OpenSim CustomJoint. The typical fields of this
    # structure are the following: name, parent, child, coordsNames,
    # coordsTypes, ROM and rotationAxes. An example of JointParamsStruct is
    # the following:
    # JointParamsStruct['jointName'] = 'hip_r'
    # JointParamsStruct['parentName'] = 'pelvis'
    # JointParamsStruct['childName'] = 'femur_r'
    # JointParamsStruct['coordsNames'] = ['hip_flexion_r','hip_adduction_r','hip_rotation_r']
    # JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational']
    # JointParamsStruct['coordRanges'] = [[-120, 120], [-120, 120], [-120, 120]] # in degrees 
    # JointParamsStruct['rotationAxes'] = 'zxy'
    # -------------------------------------------------------------------------
    
    # detect side from bone names
    if joint_name[-2:] == '_r' or joint_name[-2:] == '_l':
        side = joint_name[-1]
        
    # assign the parameters required to create a CustomJoint
    JointParamsStruct = {}
    if joint_name == 'ground_pelvis':
        JointParamsStruct['jointName'] = 'ground_pelvis'
        JointParamsStruct['parentName'] = 'ground'
        JointParamsStruct['childName'] = 'pelvis'
        JointParamsStruct['coordsNames'] = ['pelvis_tilt','pelvis_list','pelvis_rotation', 'pelvis_tx','pelvis_ty', 'pelvis_tz']
        JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational', 'translational', 'translational','translational']
        JointParamsStruct['coordRanges'] = [[-90, 90], [-90, 90] , [-90, 90], [-10, 10] , [-10, 10] , [-10, 10]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'free_to_ground':
        cb = root_body # cb = current bone (for brevity)
        JointParamsStruct['jointName'] = 'ground_' + cb
        JointParamsStruct['parentName'] = 'ground'
        JointParamsStruct['childName'] = cb
        JointParamsStruct['coordsNames'] = ['ground_'+ cb + '_rz', 'ground_' + cb +'_rx', 'ground_' + cb +'_ry', 'ground_'+ cb + '_tx', 'ground_' + cb +'_ty', 'ground_' + cb +'_tz']
        JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational', 'translational', 'translational','translational']
        JointParamsStruct['coordRanges'] = [[-120, 120], [-120, 120] , [-120, 120], [-10, 10] , [-10, 10] , [-10, 10]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'hip_' + side:
        JointParamsStruct['jointName'] = 'hip_' + side
        JointParamsStruct['parentName'] = 'pelvis'
        JointParamsStruct['childName'] = 'femur_' + side
        JointParamsStruct['coordsNames'] = ['hip_flexion_' + side, 'hip_adduction_' + side, 'hip_rotation_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational']
        JointParamsStruct['coordRanges'] = [[-120, 120], [-120, 120] , [-120, 120]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'knee_' + side:
        JointParamsStruct['jointName'] = 'knee_' + side
        JointParamsStruct['parentName'] = 'femur_' + side
        JointParamsStruct['childName'] = 'tibia_' + side
        JointParamsStruct['coordsNames'] = ['knee_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-120, 10]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'ankle_' + side:
        JointParamsStruct['jointName'] = 'ankle_' + side
        JointParamsStruct['parentName'] = 'tibia_' + side
        JointParamsStruct['childName'] = 'talus_' + side
        JointParamsStruct['coordsNames'] = ['ankle_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'subtalar_' + side:
        JointParamsStruct['jointName'] = 'subtalar_' + side
        JointParamsStruct['parentName'] = 'talus_' + side
        JointParamsStruct['childName'] = 'calcn_' + side
        JointParamsStruct['coordsNames'] = ['subtalar_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'subtalar_' + side:
        JointParamsStruct['jointName'] = 'subtalar_' + side
        JointParamsStruct['parentName'] = 'talus_' + side
        JointParamsStruct['childName'] = 'calcn_' + side
        JointParamsStruct['coordsNames'] = ['subtalar_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'patellofemoral_' + side:
        JointParamsStruct['jointName'] = 'patellofemoral_' + side
        JointParamsStruct['parentName'] = 'femur_' + side
        JointParamsStruct['childName'] = 'patella_' + side
        JointParamsStruct['coordsNames'] = ['knee_angle_' + side + '_beta']
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'mtp_' + side:
        JointParamsStruct['jointName'] = 'toes_' + side
        JointParamsStruct['parentName'] = 'calcn_' + side
        JointParamsStruct['childName'] = 'toes_' + side
        JointParamsStruct['coordsNames'] = ['mtp_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    else:
        print('getJointParams.py Unsupported joint ' + joint_name + '.')
        # logging.error('getJointParams.py Unsupported joint ' + joint_name + '.')
        
    return JointParamsStruct

# -----------------------------------------------------------------------------
def getJointParams3DoFKnee(joint_name, root_body = 'root_body'):
    # -------------------------------------------------------------------------
    # Custom function that implements a lower limb model with a 3 degrees of 
    # freedom knee joint. 
    # This script is an example of advanced use of STAPLE.
    # 
    # Inputs:
    # joint_name - name of the lower limb joint for which we want to create
    # the structure containing all parameters (string).
    # 
    # root_body - a string specifying the name of the body attached to ground
    # in the case it is not the default (pelvis).
    # 
    # Outputs:
    # JointParamsStruct - a structure collecting all the information required
    # to define an OpenSim CustomJoint. The typical fields of this
    # structure are the following: name, parent, child, coordsNames,
    # coordsTypes, ROM and rotationAxes. An example of JointParamsStruct is
    # the following:
    # JointParamsStruct['jointName'] = 'hip_r'
    # JointParamsStruct['parentName'] = 'pelvis'
    # JointParamsStruct['childName'] = 'femur_r'
    # JointParamsStruct['coordsNames'] = ['hip_flexion_r','hip_adduction_r','hip_rotation_r']
    # JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational']
    # JointParamsStruct['coordRanges'] = [[-120, 120], [-120, 120], [-120, 120]] # in degrees 
    # JointParamsStruct['rotationAxes'] = 'zxy'
    # -------------------------------------------------------------------------
    
    # detect side from bone names
    if joint_name[-2:] == '_r' or joint_name[-2:] == '_l':
        side = joint_name[-1]
        
    # assign the parameters required to create a CustomJoint
    JointParamsStruct = {}
    if joint_name == 'ground_pelvis':
        JointParamsStruct['jointName'] = 'ground_pelvis'
        JointParamsStruct['parentName'] = 'ground'
        JointParamsStruct['childName'] = 'pelvis'
        JointParamsStruct['coordsNames'] = ['pelvis_tilt','pelvis_list','pelvis_rotation', 'pelvis_tx','pelvis_ty', 'pelvis_tz']
        JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational', 'translational', 'translational','translational']
        JointParamsStruct['coordRanges'] = [[-90, 90], [-90, 90] , [-90, 90], [-10, 10] , [-10, 10] , [-10, 10]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'free_to_ground':
        cb = root_body # cb = current bone (for brevity)
        JointParamsStruct['jointName'] = 'ground_' + cb
        JointParamsStruct['parentName'] = 'ground'
        JointParamsStruct['childName'] = cb
        JointParamsStruct['coordsNames'] = ['ground_'+ cb + '_rz', 'ground_' + cb +'_rx', 'ground_' + cb +'_ry', 'ground_'+ cb + '_tx', 'ground_' + cb +'_ty', 'ground_' + cb +'_tz']
        JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational', 'translational', 'translational','translational']
        JointParamsStruct['coordRanges'] = [[-120, 120], [-120, 120] , [-120, 120], [-10, 10] , [-10, 10] , [-10, 10]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'hip_' + side:
        JointParamsStruct['jointName'] = 'hip_' + side
        JointParamsStruct['parentName'] = 'pelvis'
        JointParamsStruct['childName'] = 'femur_' + side
        JointParamsStruct['coordsNames'] = ['hip_flexion_' + side, 'hip_adduction_' + side, 'hip_rotation_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational']
        JointParamsStruct['coordRanges'] = [[-120, 120], [-120, 120] , [-120, 120]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    # --------------------------------------------------
    # SPECIALSETTING FOR ALTERING JOINT
    # --------------------------------------------------
    elif joint_name == 'knee_' + side:
        JointParamsStruct['jointName'] = 'knee_' + side
        JointParamsStruct['parentName'] = 'femur_' + side
        JointParamsStruct['childName'] = 'tibia_' + side
        JointParamsStruct['coordsNames'] = ['knee_angle_' + side, 'knee_varus_' + side, 'knee_rotation_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational']
        JointParamsStruct['coordRanges'] = [[-120, 10], [-20, 20], [-30, 30]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    # --------------------------------------------------
    elif joint_name == 'ankle_' + side:
        JointParamsStruct['jointName'] = 'ankle_' + side
        JointParamsStruct['parentName'] = 'tibia_' + side
        JointParamsStruct['childName'] = 'talus_' + side
        JointParamsStruct['coordsNames'] = ['ankle_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'subtalar_' + side:
        JointParamsStruct['jointName'] = 'subtalar_' + side
        JointParamsStruct['parentName'] = 'talus_' + side
        JointParamsStruct['childName'] = 'calcn_' + side
        JointParamsStruct['coordsNames'] = ['subtalar_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'subtalar_' + side:
        JointParamsStruct['jointName'] = 'subtalar_' + side
        JointParamsStruct['parentName'] = 'talus_' + side
        JointParamsStruct['childName'] = 'calcn_' + side
        JointParamsStruct['coordsNames'] = ['subtalar_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'patellofemoral_' + side:
        JointParamsStruct['jointName'] = 'patellofemoral_' + side
        JointParamsStruct['parentName'] = 'femur_' + side
        JointParamsStruct['childName'] = 'patella_' + side
        JointParamsStruct['coordsNames'] = ['knee_angle_' + side + '_beta']
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    elif joint_name == 'mtp_' + side:
        JointParamsStruct['jointName'] = 'toes_' + side
        JointParamsStruct['parentName'] = 'calcn_' + side
        JointParamsStruct['childName'] = 'toes_' + side
        JointParamsStruct['coordsNames'] = ['mtp_angle_' + side]
        JointParamsStruct['coordsTypes'] = ['rotational']
        JointParamsStruct['coordRanges'] = [[-90, 90]] # in degrees 
        JointParamsStruct['rotationAxes'] = 'zxy'
    else:
        print('getJointParams.py Unsupported joint ' + joint_name + '.')
        # logging.error('getJointParams.py Unsupported joint ' + joint_name + '.')
        
    return JointParamsStruct

# -----------------------------------------------------------------------------
def assembleJointStruct(jointStruct = {}):
    # -------------------------------------------------------------------------
    # Fill the MATLAB structure that will be used for creating the OpenSim 
    # joints using the available information on the joint parameters. It makes 
    # the simple assumption that when a parameter is missing 
    # (child/parent_location/orientation), it is appropriate to copy the 
    # information from the corresponding parent/child_location/orientation 
    # parameter. This works when using geometries segmented from consistent 
    # medical images and might not be appropriate for all intended uses of 
    # STAPLE (although it is for most uses).
    # 
    # Inputs:
    # jointStruct - Dictionary including the reference parameters
    # that will be used to generate an OpenSim JointSet. It might be
    # incomplete, with joints missing some of the required parameters.
    # 
    # Outputs:
    # updJointStruct - updated jointStruct with the joints with fields 
    # completed as discussed above.
    # -------------------------------------------------------------------------
    
    updJointStruct = jointStruct.copy()
    
    fields_to_check = ['parent_location', 'parent_orientation', \
                       'child_location',  'child_orientation']
    
    print('Finalizing joints:')
    
    for cur_joint_name in jointStruct:
        
        print(' *' + cur_joint_name)
        complete_fields = [True if key in jointStruct[cur_joint_name] else False for key in fields_to_check]
        
        if all(complete_fields):
            print('   - already complete.')
            continue
        
        if np.sum(complete_fields[0::2]) == 0:
            print('   - WARNING: ' + cur_joint_name + ' cannot be finalized: no joint locations available on neither bodies. Please read log and fix it.')
            continue
        
        if np.sum(complete_fields[1::2]) == 0:
            print('   - WARNING: Joint ' + cur_joint_name + ' cannot be finalized: no joint orientation available on neither bodies. Please read log and fix it.')
            continue
        
        # if wither child or parent location/orientation is available, copy on 
        # the missing field
        for pos, key in enumerate(fields_to_check):
            if complete_fields[pos] == False:
                if key == 'parent_location':
                    updJointStruct[cur_joint_name][key] = jointStruct[cur_joint_name]['child_location']
                    print('   - parent_location missing: copied from "child_location".')
                elif key == 'child_location':
                    updJointStruct[cur_joint_name][key] = jointStruct[cur_joint_name]['parent_location']
                    print('   - child_location missing: copied from "parent_location".')
                elif key == 'parent_orientation':
                    updJointStruct[cur_joint_name][key] = jointStruct[cur_joint_name]['child_orientation']
                    print('   - parent_orientation missing: copied from "child_orientation".')
                elif key == 'child_orientation':
                    updJointStruct[cur_joint_name][key] = jointStruct[cur_joint_name]['parent_orientation']
                    print('   - child_orientation missing: copied from "parent_orientation".')
        
    return updJointStruct

# -----------------------------------------------------------------------------
def verifyJointStructCompleteness(jointStruct = {}):
    # -------------------------------------------------------------------------
    # Check that the MATLAB structure that will be used for creating the 
    # OpenSim joints includes all the required parameters, so that the joint 
    # generation will not failed when the OpenSim API are called. This function
    # should be called after the joint definitions have been applied, as a 
    # "last check" before running the OpenSim API. 
    # It checked that the following fields are defined:
    # - child/parent_location/orientation
    # - joint/parent/child_name
    # - coordsNames/Types
    # - rotationAxes
    # 
    # Inputs:
    # jointStruct - Dictionary including the reference parameters
    # that will be used to generate an OpenSim JointSet. It might be
    # incomplete, with joints missing some of the required parameters.
    # 
    # Outputs:
    # none
    # -------------------------------------------------------------------------
        
    fields_to_check = ['jointName', 'parentName', 'parent_location',\
                       'parent_orientation', 'childName', 'child_location',\
                       'child_orientation', 'coordsNames', 'coordsTypes',\
                       'rotationAxes']
    
    throw_error = False
    
    for cur_joint_name in jointStruct:
        
        defined_joint_params = [True if key in jointStruct[cur_joint_name] else False for key in fields_to_check]
        # if error will be thrown (not all fields are present), printout 
        # where the issue is
        if all(defined_joint_params) != True:
            
            throw_error = True
            print(cur_joint_name + ' definition incomplete. Missing fields:')
            
            for pos, key in enumerate(fields_to_check):
                if defined_joint_params[pos] == False:
                    print('   -> ' + key)
    
    # if flag is now positive, throw the error
    if throw_error:
        print('createOpenSimModelJoints.py Incomplete joint(s) definition(s): the joint(s) cannot be generated. See printout above.')
        # loggin.error('createOpenSimModelJoints.py Incomplete joint(s) definition(s): the joint(s) cannot be generated. See printout above.')
    else:
        print('All joints are complete!')
        
    return 0

# -----------------------------------------------------------------------------
def getAxisVecFromStringLabel(axisLabel):
    # -------------------------------------------------------------------------
    # Take one character as input ('x', 'y' or 'z') and returns the 
    # corresponding vector (as a row).
    # 
    # Inputs:
    # axisLabel - a string indicating an axis. Valid values are: 'x', 'y' or 
    # 'z'.
    # 
    # Outputs:
    # v - a row vector corresponding to the axis label specified as input.
    # -------------------------------------------------------------------------
    
    # make it case independent
    axisLabel = axisLabel.lower()
    
    # switch axisLabel
    if axisLabel == 'x':
        v = np.array([1, 0, 0])
        # v = np.reshape(v,(v.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    elif axisLabel == 'y':
        v = np.array([0, 1, 0])
        # v = np.reshape(v,(v.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    elif axisLabel == 'z':
        v = np.array([0, 0, 1])
        # v = np.reshape(v,(v.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    
    return v

# -----------------------------------------------------------------------------
def createSpatialTransformFromStruct(jointStruct = {}):
    # -------------------------------------------------------------------------
    # Create a SpatialTransform from the provided structure. Intended to be 
    # used to create a CustomJoint in OpenSim >4.0.
    # 
    # Inputs:
    # jointStruct - Dictionary with the typical fields of an OpenSim 
    # CustomJoint: name, parent (name), child (name), parent location, 
    # parent orientation, child location, child orientation.
    # 
    # Outputs:
    # jointSpatialTransf - a SpatialTranform (object), to be included in an OpenSim 
    # CustomJoint.
    # 
    # Example of input structure:
    # JointParamsStruct['coordsNames'] = ['hip_flexion_r','hip_adduction_r','hip_rotation_r']
    # JointParamsStruct['coordsTypes'] = ['rotational', 'rotational', 'rotational']
    # JointParamsStruct['rotationAxes'] = 'zxy'
    # -------------------------------------------------------------------------
    
    # creating coordinates
    coordsNames = jointStruct['coordsNames']
    coordsTypes = jointStruct['coordsTypes']
    
    # rotational coordinates
    rot_coords_names = [name for pos, name in enumerate(coordsNames) if coordsTypes[pos] == 'rotational']
    
    # translational coordinates
    trans_coords_names = [name for pos, name in enumerate(coordsNames) if coordsTypes[pos] == 'translational']
    
    # list of coordinate names
    coords_names = rot_coords_names + trans_coords_names
    
    # check of consistency of dimentions

    if len(coordsNames) != len(coords_names):
        print('ERROR: createSpatialTrasnformFromStruct.py The sum of translational and rotational coordinates does not match the coordinates names. Please double check.')
        # loggin.error('createSpatialTrasnformFromStruct.py The sum of translational and rotational coordinates does not match the coordinates names. Please double check.')
    
    # extracting the vectors associated with the order of rotation 
    # if rotationAxes is specified they will be updated otherwise will stay
    v = np.zeros((6,3))
    v[:3,:] = np.eye(3)
    if 'rotationAxes' in jointStruct:
        rotationAxes = jointStruct['rotationAxes']
        if type(rotationAxes) is str:
            for ind, axe in enumerate(rotationAxes):
                v[ind,:] = getAxisVecFromStringLabel(axe)
        else:
            for ind in range(len(rot_coords_names)):
                v[ind,:] = rotationAxes[ind,:]
    
    # translations are always along the axes XYZ (in this order)
    v[3:,:] = np.eye(3)
    if 'translationAxes' in jointStruct:
        translationAxes = jointStruct['translationAxes']
        if type(translationAxes) is str:
            for ind, axe in enumerate(translationAxes):
                v[ind,:] = getAxisVecFromStringLabel(axe)
        else:
            for ind in range(len(trans_coords_names)):
                v[ind+3,:] = translationAxes[ind,:]
    
    # create spatial transform
    jointSpatialTransf = opensim.SpatialTransform()
    
    # ================= ROTATIONS ===================
    # create a linear function and a constant function
    lin_fun = opensim.LinearFunction(1,0)
    const_fun = opensim.Constant(0)
    
    # looping through axes (3 rotations, 3 translations)
    for n, name in enumerate(coords_names):
        if 'tilt' in name or 'flexion' in name:
            rot1 = jointSpatialTransf.get_rotation1()
            rot1.append_coordinates(name)
            rot1.set_axis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
            rot1.set_function(lin_fun)
        if 'list' in name or 'adduction' in name:
            rot2 = jointSpatialTransf.get_rotation2()
            rot2.append_coordinates(name)
            rot2.set_axis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
            rot2.set_function(lin_fun)
        if 'rotation' in name:
            rot3 = jointSpatialTransf.get_rotation3()
            rot3.append_coordinates(name)
            rot3.set_axis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
            rot3.set_function(lin_fun)
        if 'tx' in name:
            trans1 = jointSpatialTransf.get_translation1()
            trans1.append_coordinates(name)
            trans1.set_axis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
            trans1.set_function(lin_fun)
        if 'ty' in name:
            trans2 = jointSpatialTransf.get_translation2()
            trans2.append_coordinates(name)
            trans2.set_axis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
            trans2.set_function(lin_fun)
        if 'tz' in name:
            trans3 = jointSpatialTransf.get_translation3()
            trans3.append_coordinates(name)
            trans3.set_axis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
            trans3.set_function(lin_fun)
        if 'angle' in name:
            rot1 = jointSpatialTransf.get_rotation1()
            rot1.append_coordinates(name)
            rot1.set_axis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
            rot1.set_function(lin_fun)
            rot2 = jointSpatialTransf.get_rotation2()
            rot2.append_coordinates('')
            rot2.set_axis(opensim.ArrayDouble.createVec3(0, 1, 0))
            rot2.set_function(const_fun)
            rot3 = jointSpatialTransf.get_rotation3()
            rot3.append_coordinates('')
            rot3.set_axis(opensim.ArrayDouble.createVec3(1, 0, 0))
            rot3.set_function(const_fun)
        
        jointSpatialTransf.updTransformAxis(n)
    # for n, name in enumerate(coords_names):
    #     # get modifiable transform axis (upd..)
    #     TransAxis = jointSpatialTransf.updTransformAxis(n)
        
    #     # applying specified rotation order
    #     # TransAxis = jointSpatialTransf.updTransformAxis(TransAxis, v[n,:])
    #     TransAxis.setAxis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
    #     # jointSpatialTransf.updTransformAxis(n).setAxis(opensim.ArrayDouble.createVec3(v[n,0], v[n,1], v[n,2]))
        
    #     # this will update the coordinate names and assign a linear
    #     # function to those axes with a coordinate associated with.
    #     # the axis without a coordinate associated will be assigned a constant
    #     # zero function (they will not move).
    #     # TransAxis = jointSpatialTransf.updTransformAxisCoordName(TransAxis, name)
    #     TransAxis.setName(name)
    #     # jointSpatialTransf.updTransformAxis(n).setName(name)
        
    #     # assign appropriate function
    #     if name != '':
    #         # TransAxis.set_function(lin_fun)
    #         jointSpatialTransf.updTransformAxis(n).set_function(lin_fun)
    #     else:
    #         # TransAxis.set_function(const_fun)
    #         jointSpatialTransf.updTransformAxis(n).set_function(const_fun)
            
    #     # jointSpatialTransf.updTransformAxis(n)
    
    # this will take care of having 3 independent axis
    jointSpatialTransf.constructIndependentAxes(len(rot_coords_names), 0)
    
    return jointSpatialTransf
    
# -----------------------------------------------------------------------------
def createCustomJointFromStruct(osimModel, jointStruct = {}):
    # -------------------------------------------------------------------------
    # Create and add to model a CustomJoint using the parameters defined in the
    # dictionary given as input.
    # 
    # Inputs:
    # model - an OpenSim model (object)
    # 
    # jointStruct - Dictionary with the typical fields of an OpenSim 
    # CustomJoint: name, parent (name), child (name), parent location, 
    # parent orientation, child location, child orientation.
    # 
    # Outputs:
    # myCustomJoint - a CustomJoint (object), that can be used outside this
    # function to add it to the OpenSim model.
    # 
    # Example of dictionary to provide as input:
    # JointParamsStruct['jointName'] = 'knee_r'
    # JointParamsStruct['parentName'] = 'femur_r'
    # JointParamsStruct['childName'] = 'tibia_r'
    # JointParamsStruct['coordsNames'] = ['knee_angle_r', 'knee_tx_r']
    # JointParamsStruct['coordsTypes'] = ['rotational', 'translational']
    # JointParamsStruct['rotationAxes'] = 'zxy' # OPTIONAL: default xyz; ALWAYS 3 COMP 
    # JointParamsStruct['translationAxes'] = 'xyz' # OPTIONAL: default xyz; ALWAYS 3 COMP
    # JointParamsStruct['coordRanges'] = [[-10 90], [-3 3]] # OPTIONAL (deg/metres)
    # JointParamsStruct['parent_location'] = np.array([x y z])
    # JointParamsStruct['parent_orientation'] = np.array([x y z])
    # JointParamsStruct['child_location'] = np.array([x y z])
    # JointParamsStruct['child_orientation'] = np.array([x y z])
    # -------------------------------------------------------------------------
    
    # extract names
    jointName = jointStruct['jointName']
    parentName = jointStruct['parentName']
    childName = jointStruct['childName']
    
    # transform offsets in Vec3
    location_in_parent = opensim.ArrayDouble.createVec3(jointStruct['parent_location'][0][0], jointStruct['parent_location'][1][0], jointStruct['parent_location'][2][0])
    orientation_in_parent = opensim.ArrayDouble.createVec3(jointStruct['parent_orientation'][0][0], jointStruct['parent_orientation'][0][1], jointStruct['parent_orientation'][0][2])
    location_in_child = opensim.ArrayDouble.createVec3(jointStruct['child_location'][0][0], jointStruct['child_location'][1][0], jointStruct['child_location'][2][0])
    orientation_in_child = opensim.ArrayDouble.createVec3(jointStruct['child_orientation'][0][0], jointStruct['child_orientation'][0][1], jointStruct['child_orientation'][0][2])
    
    # get the Physical Frames to connect with the CustomJoint
    if parentName == 'ground':
        parent_frame = osimModel.getGround()
    else:
        parent_frame = osimModel.getBodySet().get(parentName)
        
    child_frame = osimModel.getBodySet().get(childName)
    
    # create the spatialTransform from the assigned structure
    # openSim 3.3
    # OSJoint = setCustomJointSpatialTransform(OSJoint, jointStruct);
    # OpenSim 4.1
    jointSpatialTransf = createSpatialTransformFromStruct(jointStruct)
    
    # create the CustomJoint
    myCustomJoint = opensim.CustomJoint(jointName,\
                                        parent_frame, location_in_parent, orientation_in_parent,\
                                        child_frame, location_in_child, orientation_in_child,\
                                        jointSpatialTransf)
    
    # add joint to model
    osimModel.addJoint(myCustomJoint)
    
    # update coordinates range of motion, if specified
    if 'coordRanges' in jointStruct:
        for n_coord in range(len(jointStruct['coordsNames'])):
            curr_coord = myCustomJoint.get_coordinates(int(n_coord))
            curr_ROM = jointStruct['coordRanges'][n_coord]
            if jointStruct['coordsTypes'][n_coord] == 'rotational':
                # curr_ROM /= 180*np.pi
                curr_ROM[0] /= 180*np.pi
                curr_ROM[1] /= 180*np.pi
            # set the range of motion for the coordinate
            curr_coord.setRangeMin(curr_ROM[0])
            curr_coord.setRangeMax(curr_ROM[1])
            
    # state = osimModel.initSystem
    
    return myCustomJoint

# -----------------------------------------------------------------------------
def createOpenSimModelJoints(osimModel, JCS, joint_defs = 'auto2020', jointParamFile = 'getJointParams'):
    # -------------------------------------------------------------------------
    # Create the lower limb joints based on assigned joint coordinate systems
    # stored in a structure and adds them to an existing OpenSim model.
    # 
    # Inputs:
    # osimModel - an OpenSim model of the lower limb to which we want to add
    # the lower limb joints.
    # 
    # JCS - a dictionary created using the function createLowerLimbJoints(). 
    # This structure includes as fields the elements to generate a 
    # CustomJoint using the createCustomJointFromStruct function. See these 
    # functions for details.
    # 
    # joint_defs - optional input specifying the joint definitions used for
    # arranging and finalizing the partial reference systems obtained
    # from morphological analysis of the bones. Valid values:
    # - 'Modenese2018': define the same reference systems described in 
    # Modenese et al. J Biomech (2018).
    # - 'auto2020': use the reference systems from morphological analysis as 
    # much as possible. See Modenese and Renault, JBiomech 2020 for a comparison.
    # - any definition you want to add. You just need to write a function and 
    # include it in the "switch" structure where joints are defined. Your 
    # implementation will be check for completeness by the 
    # verifyJointStructCompleteness.m function.
    # 
    # jointParamFile - optional input specifying the name of a function 
    # including the parameters of the joints to build. Default value is:
    # - 'getJointParams.py': same joint parameters as the gait2392
    # standard model.
    # - 'getJointParams3DoFKnee.py': file available from the advanced examples, 
    # shows how to create a 3 dof knee joint.
    # - any other joint parameters you want to implement. Be careful because 
    # joint definitions and input bone will have to match:
    # for example you cannot create a 2 dof ankle joint and
    # exclude a subtalar joint if you are providing a talus and
    # calcn segments, as otherwise they would not have any joint.
    # 
    # Outputs:
    # none - the joints are added to the input OpenSim model.
    # -------------------------------------------------------------------------
    
    # printout
    print('---------------------')
    print('   CREATING JOINTS   ')
    print('---------------------')

    # add ground body to JCS together with standard ground_pelvis joint.
    # if model is partial, it will be modified.
    JCS['ground'] = {}
    JCS['ground']['ground_pelvis'] = {'parentName': 'ground', \
                                      'parent_location': np.zeros((3,1)), \
                                      'parent_orientation': np.zeros((1,3))}

    # based on JCS make a list of bodies and joints
    joint_list = compileListOfJointsInJCSStruct(JCS)

    # TRANSFORM THE JCS FROM MORPHOLOGYCAL ANALYSIS IN JOINT DEFINITION
    # complete the joints parameters
    print('Checking parameters from morphological analysis:')

    # useful list
    fields_v = ['parent_location','parent_orientation','child_location', 'child_orientation']

    if jointParamFile != 'getJointParams':
        jointParamFuncName = jointParamFile
    else:
        print('WARNING: Specified function ' + jointParamFile + 'for joint parameters was not found. Using default "getJointParams.py"')
        jointParamFuncName = 'getJointParams'

    jointStruct = {}
    for cur_joint_name in joint_list:
        # getting joint parameters using the desired joint param function
        if jointParamFuncName == 'getJointParams':
            jointStructTemp = getJointParams(cur_joint_name)
        elif jointParamFuncName == 'getJointParams3DoFKnee':
            jointStructTemp = getJointParams3DoFKnee(cur_joint_name)
        
        # STEP1: check if parent and child body are available
        parent_name = jointStructTemp['parentName']
        child_name  = jointStructTemp['childName']

        # the assumption is that if, given a joint from the analysis, parent is
        # missing, that's because the model is partial proximally and will be
        # connected to ground. If child is missing, instead, the model if
        # partial distally and the chain will be interrupted there.
        if parent_name not in JCS:
            if child_name in JCS:
                print('Partial model detected proximally:')
                
                # get appropriate parameters for the new joint
                jointStructTemp = getJointParams('free_to_ground', child_name)
                
                # adjusting joint parameters
                old_joint_name = cur_joint_name
                new_cur_joint_name = jointStructTemp['jointName']
                parent_name = jointStructTemp['parentName']
                print('   * Connecting ' + child_name + ' to ground with ' + new_cur_joint_name + ' free joint.')
                
                # defines the new joints for parent/child location and orientation
                JCS['ground'][new_cur_joint_name] = JCS['ground']['ground_pelvis']
                JCS['child_name'][new_cur_joint_name] = JCS['child_name'][old_joint_name]
            else:
                new_cur_joint_name = cur_joint_name
                print('ERROR: Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
                # loggin.error('Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
        else:
            new_cur_joint_name = cur_joint_name
        
        # if there is a parent but not a child body then the model is partial
        # distally, i.e. it is missing some distal body/bodies.
        if child_name not in JCS:
            if parent_name in JCS:
                print('Partial model detected distally...')
                print('* Deleting incomplete joint "' + new_cur_joint_name + '"')
                continue
            else:
                print('ERROR: Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
                # loggin.error('Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
        
        # display joint details
        print('* ' + new_cur_joint_name)
        print('   - parent: ' + parent_name)
        print('   - child: ' + child_name)
        
        # create an appropriate jointStructTemp from the info available in JCS
        # body_list = list(JCS.keys())
        for cur_body_name in JCS:
            if new_cur_joint_name in JCS[cur_body_name]:
                joint_info = JCS[cur_body_name][new_cur_joint_name]
                for key in fields_v:
                    if key in joint_info:
                        jointStructTemp[key] = joint_info[key]
            else:
                continue
        
        # store the resulting parameters for each joint to the final dictionary
        jointStruct[new_cur_joint_name] = jointStructTemp
                    
    # JOINT DEFINITIONS
    print('Applying joint definitions: ' + joint_defs)

    if joint_defs == 'auto2020':
        jointStruct = jointDefinitions_auto2020(JCS, jointStruct)
    elif joint_defs == 'Modenese2018':
        # joint definitions of Modenese et al.
        jointStruct = jointDefinitions_Modenese2018(JCS, jointStruct)
    else:
        print('createOpenSimModelJoints.py You need to define joint definitions')
        # loggin.error('createOpenSimModelJoints.py You need to define joint definitions')

    # completeJoints(jointStruct)
    jointStruct = assembleJointStruct(jointStruct)

    # check that all joints are completed
    verifyJointStructCompleteness(jointStruct)

    # after the verification joints can be added
    print('Adding joints to model:')

    # for cur_joint_name in jointStruct:
    #     # create the joint
        # _ = createCustomJointFromStruct(osimModel, jointStruct[cur_joint_name])
    #     # display what has been created
    #     print('   * ' + cur_joint_name)
    #     # state = osimModel.initSystem
    
    # ----------------------------------------------
    for cur_joint_name in jointStruct:
        # create the joint
        
        # ------------------------------------------
        # extract names
        jointName = jointStruct[cur_joint_name]['jointName']
        parentName = jointStruct[cur_joint_name]['parentName']
        childName = jointStruct[cur_joint_name]['childName']
        
        # transform offsets in Vec3
        location_in_parent = opensim.ArrayDouble.createVec3(jointStruct[cur_joint_name]['parent_location'][0][0], jointStruct[cur_joint_name]['parent_location'][1][0], jointStruct[cur_joint_name]['parent_location'][2][0])
        orientation_in_parent = opensim.ArrayDouble.createVec3(jointStruct[cur_joint_name]['parent_orientation'][0][0], jointStruct[cur_joint_name]['parent_orientation'][0][1], jointStruct[cur_joint_name]['parent_orientation'][0][2])
        location_in_child = opensim.ArrayDouble.createVec3(jointStruct[cur_joint_name]['child_location'][0][0], jointStruct[cur_joint_name]['child_location'][1][0], jointStruct[cur_joint_name]['child_location'][2][0])
        orientation_in_child = opensim.ArrayDouble.createVec3(jointStruct[cur_joint_name]['child_orientation'][0][0], jointStruct[cur_joint_name]['child_orientation'][0][1], jointStruct[cur_joint_name]['child_orientation'][0][2])
        
        # get the Physical Frames to connect with the CustomJoint
        if parentName == 'ground':
            parent_frame = osimModel.getGround()
        else:
            parent_frame = osimModel.getBodySet().get(parentName)
            
        child_frame = osimModel.getBodySet().get(childName)
        
        # create the spatialTransform from the assigned structure
        # openSim 3.3
        # OSJoint = setCustomJointSpatialTransform(OSJoint, jointStruct);
        # OpenSim 4.1
        jointSpatialTransf = createSpatialTransformFromStruct(jointStruct[cur_joint_name])
        
        # create the CustomJoint
        myCustomJoint = opensim.CustomJoint(jointName,\
                                            parent_frame, location_in_parent, orientation_in_parent,\
                                            child_frame, location_in_child, orientation_in_child,\
                                            jointSpatialTransf)
        
        # add joint to model
        osimModel.addJoint(myCustomJoint)
        
        # update coordinates range of motion, if specified
        if 'coordRanges' in jointStruct:
            for n_coord in range(len(jointStruct['coordsNames'])):
                curr_coord = myCustomJoint.get_coordinates(int(n_coord))
                curr_ROM = jointStruct['coordRanges'][n_coord]
                if jointStruct['coordsTypes'][n_coord] == 'rotational':
                    # curr_ROM /= 180*np.pi
                    curr_ROM[0] /= 180*np.pi
                    curr_ROM[1] /= 180*np.pi
                # set the range of motion for the coordinate
                curr_coord.setRangeMin(curr_ROM[0])
                curr_coord.setRangeMax(curr_ROM[1])
                # osimModel.getJointSet().get(curr_coord).setRangeMin(curr_ROM[0])
                # osimModel.getJointSet().get(curr_coord).setRangeMax(curr_ROM[1])
                
        # state = osimModel.initSystem
    

    print('Done.')
    
    return osimModel

# -----------------------------------------------------------------------------
def addBoneLandmarksAsMarkers(osimModel, BLStruct, in_mm = 1):
    # -------------------------------------------------------------------------
    # Add the bone landmarks listed in the input structure as Markers in the 
    # OpenSim model.
    # 
    # Inputs:
    # osimModel - an OpenSim model (object) to which to add the bony
    # landmarks as markers.
    # 
    # BLStruct - a Dictionary with two layers. The external layer has
    # fields named as the bones, the internal layer as fields named as
    # the bone landmarks to add. The value of the latter fields is a
    # [1x3] vector of the coordinate of the bone landmark. For example:
    # BLStruct['femur_r']['RFE'] = [xp, yp, zp].
    # 
    # in_mm - if all computations are performed in mm or m. Valid values: 1
    # or 0.
    # 
    # Outputs:
    # none - the OpenSim model in the scope of the calling function will
    # include the specified markers.
    # 
    # -------------------------------------------------------------------------
    
    if in_mm == 1:
        dim_fact = 0.001
    else:
        dim_fact = 1

    print('------------------------')
    print('     ADDING MARKERS     ')
    print('------------------------')
    print('Attaching bony landmarks to model bodies:')

    # loop through the bodies specified in BLStruct
    # myState = osimModel.initSystem
    # newMarkerSet = osimModel.getMarkerSet()
    newMarkerSet = opensim.MarkerSet()
    
    for cur_body_name in BLStruct:
        # body name
        print('  ' + cur_body_name + ':')
        
        # check that cur_body_name actually corresponds to a body
        if osimModel.getBodySet().getIndex(cur_body_name) < 0:
            # loggin.warning('Markers assigned to body ' + cur_body_name + ' cannot be added to the model. Body is not in BodySet.')
            print('Markers assigned to body ' + cur_body_name + ' cannot be added to the model. Body is not in BodySet.')
            continue
        
        # loop through the markers
        cur_body_markers = list(BLStruct[cur_body_name].keys())
        # skip markers if the structure is empty, otherwise process it
        if cur_body_markers == []:
            print('    NO LANDMARKS AVAILABLE')
            continue
        else:
            # the actual markers are fields of the cur_body_markers variable
            for cur_marker_name in BLStruct[cur_body_name]:
                # get body
                # cur_phys_frame = osimModel.getBodySet().get(cur_body_name)
                Loc = BLStruct[cur_body_name][cur_marker_name]*dim_fact
                # marker = opensim.Marker(cur_marker_name, \
                #                         cur_phys_frame,\
                #                         opensim.Vec3(float(Loc[0][0]), float(Loc[1][0]), float(Loc[2][0])))
                marker = opensim.Marker()
                marker.setName(cur_marker_name)
                marker.setParentFrameName('/bodyset/' + cur_body_name)
                marker.set_location(opensim.Vec3(float(Loc[0][0]), float(Loc[1][0]), float(Loc[2][0])))
                # add current marker to model
                newMarkerSet.addComponent(marker)
                # newMarkerSet.addMarker(marker)
                # osimModel.addMarker(marker)
                
                
                # clear coordinates as precaution
                del Loc
                print('    * ' + cur_marker_name)
    
    # myState = osimModel.initSystem
    osimModel.set_MarkerSet(newMarkerSet)
    # myState = osimModel.initSystem
    # osimModel.updMarkerSet()
    # myState = osimModel.initSystem
    # osimModel.addComponent(newMarkerSet)

    print('Done.')

    return 0


