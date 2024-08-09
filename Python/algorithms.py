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

from geometry import bodySide2Sign, \
                      landmarkBoneGeom

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
                                            lsplane, \
                                             TriErodeMesh, \
                                              TriCurvature, \
                                               TriConnectedPatch, \
                                                TriDifferenceMesh, \
                                                 TriVertexNormal, \
                                                  TriCloseMesh, \
                                                   PtsOnCondylesFemur, \
                                                    cylinderFitting, \
                                                     plotCylinder

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
            # loggin.error('GIBOC_femur.m ''method'' input has value: ''spheres'', ''cylinder'' or ''ellipsoids''.')
            print('GIBOC_femur.m ''method'' input has value: ''spheres'', ''cylinder'' or ''ellipsoids''.')
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
    # np.array([est_p[0], Center0[1][0], est_p[1]])
    rn = est_p[4]
    an = np.zeros((3,1))
    an[x1] = rn*np.cos(est_p[2])
    an[PoP] = -rn*np.cos(est_p[3])
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
    
    
    
    
    
    
    
    
    