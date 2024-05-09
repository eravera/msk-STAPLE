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
                         TriChangeCS

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
    tmp_LargestTriangle = {}
    BL = {}
    
    # inertial axes
    V_all, CenterVol, _, D, _ =  TriInertiaPpties(pelvisTri)
    
    # smaller moment of inertia is normally the medio/lateral axis. It will be
    # updated anyway. It can be checked using D from TriInertiaPpties
    Z0 = V_all[:,0]
    
    # compute convex hull
    hull = ConvexHull(pelvisTri['Points'])
    # transform it in triangulation
    PelvisConvHull = {'Points': hull.points[hull.vertices], 'ConnectivityList': hull.simplices}
    
    #%% Get the Post-Ant direction by finding the largest triangle of the pelvis
    # and checking the inertial axis that more closely aligns with it
    
    # Find the largest triangle on the projected Convex Hull
    PelvisConvHull_Ppties = TriMesh2DProperties(PelvisConvHull)
    I = np.argmax(PelvisConvHull_Ppties['Area'])
    
    # Get the triangle center and normal
    tmp_LargestTriangle['Points'] = PelvisConvHull['Points'][PelvisConvHull['ConnectivityList'][I]]
    tmp_LargestTriangle['ConnectivityList'] = np.array([0, 1, 2])
    
    # Convert tiangulation dict to mesh object
    LargestTriangle = mesh.Mesh(np.zeros(tmp_LargestTriangle['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(tmp_LargestTriangle['ConnectivityList']):
        for j in range(3):
            LargestTriangle.vectors[i][j] = tmp_LargestTriangle['Points'][f[j],:]
    
    # NOTE that we are working using a GIBOC reference system until where the 
    # rotation matrix is assembled using ISB conventions(specified in comments)
    
    # vector pointing forward is X
    ind_X = np.argmax(np.abs(np.dot(V_all.T, LargestTriangle.get_unit_normals())))
    X0 = V_all[:, ind_X]
    
    # Reorient X0 to point posterior to anterior
    anterior_v = LargestTriangle.centroids - CenterVol
    X0 = preprocessing.normalize(np.dot(np.sign(np.dot(anterior_v,X0)),X0),axis=0)
    
    # Y0 is just normal to X0 and Y0 (direction non inportant for now)
    # NOTE: Z normally points medio-laterally, Y will be cranio-caudal.
    # Directions not established yet
    Y0 = preprocessing.normalize(np.cross(Z0, X0))
    
    # transform the pelvis to the new set of inertial axes
    Rot = np.transpose(np.array([X0, Y0, Z0]))
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
    upw_ini = preprocessing.normalize(P3 - CenterVol)
    upw = upw_ini - (np.dot(np.dot(upw_ini, X0), X0))
    
    # vector pointing upward is Z
    ind_Z = np.argmax(np.abs(np.dot(V_all.T, upw)))
    Z0 = V_all[:, ind_Z]
    Z0 = preprocessing.normalize(np.dot(np.sign(np.dot(upw,Z0)),Z0),axis=0)
    
    # Until now I have used GIBOC convention, now I build the ISB one!
    # X0 = X0_ISB, Z0 = Y_ISB
    RotPseudoISB2Glob[:,0] = X0
    RotPseudoISB2Glob[:,1] = Z0
    RotPseudoISB2Glob[:,2] = np.cross(X0, Z0)
    
    # export markers
    BL['ICT1'] = P1
    BL['ICT2'] = P2
    
    # # debugs Plots
    # if debug_plots:
    #     plt.figure()
        
    
    
    
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
    _, CenterVol, InertiaMatrix, D =  TriInertiaPpties(Pelvis)
    
    # Modification of initial guess of CS direction [JB]
    print('Analyzing pelvis geometry...')
    
    # RotPseudoISB2Glob, LargestTriangle = pelvis_guess_CS(Pelvis, debug_plots)
    
    
    
    
    return 0

