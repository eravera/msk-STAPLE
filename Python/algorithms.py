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
                            TriReduceMesh

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
    
    #%% Get the Post-Ant direction by finding the largest triangle of the pelvis
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
def STAPLE_pelvis(Pelvis, side_raw = 'right', result_plots = 1, debug_plots = 0, in_mm = 1):
    # -------------------------------------------------------------------------
    
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
    RASIS = LargestTriangle['Points'][ASIS_inds[ind_RASIS]]
    LASIS = LargestTriangle['Points'][ASIS_inds[ind_LASIS]]
    
    # Get the Posterior, Superior, Right eigth of the pelvis
    Nodes_RPSIS = list(np.where((PelvisPseudoISB['Points'][:,0] < 0) & \
                           (PelvisPseudoISB['Points'][:,1] > 0) & \
                           (PelvisPseudoISB['Points'][:,2] > 0))[0])
    
    Pelvis_RPSIS = TriReduceMesh(Pelvis, [], Nodes_RPSIS)
    
    
    
    
    
    return 0

