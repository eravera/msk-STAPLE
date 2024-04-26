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
import pandas as pd
import os, shutil
from pathlib import Path
import sys
import time
import logging

from geometry import bodySide2Sign

from GIBOC_core import TriInertiaPpties

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
    V_all, CenterVol, _, D =  TriInertiaPpties(pelvisTri)
    
    # smaller moment of inertia is normally the medio/lateral axis. It will be
    # updated anyway. It can be checked using D from TriInertiaPpties
    Z0 = V_all[:,0]
    
    # compute convex hull
    
    
    
    
    
    
    
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

