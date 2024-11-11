#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 12:23:56 2024

@author: emi
"""
# ----------- import packages --------------
import numpy as np
import pandas as pd
import os, shutil
from pathlib import Path
import sys

from Public_functions_MTL import getVSKstatparam
                               
from Prive_functions_MTL import N2GetActiveSubject, \
                                WriteSetupScaleFile, \
                                WriteSetupMTLFile, \
                                WriteSetupIKFile, \
                                runOpenSim

# ------------------------------------------
# MTL main
# TO DO: header
# INPUT:
# OUTPUT:

# Initialized

# Set to Executables directory if running in Python directly
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


TrialName = ['2425~aa~Standing', '2425~aa~Walking 03']

TrialTypes = ['Static' if 'Standing' in name else 'Dynamic' for name in TrialName]
TrialTypes.sort(reverse=True)

TRCPathName = '/home/emi/Documents/Codigos MATLAB_PYTHON/msk-STAPLE/Python/MTL Basic files/'

TrunkFlag = True

Mass = 44.4
FootLengthR = 0
FootLengthL = 0

# if body mass is not in the c3d file, look for it in the vsk file
# if np.isnan(Mass):
#     Mass = getVSKstatparam(EclipsePath + subname + '.vsk','Bodymass')

if 'Dynamic' not in TrialTypes and 'Static' not in TrialTypes:
    print('ERROR reading c3d.')
    print(TrialTypes)
    quit()

print(' done.')

# tend = cend['time'] - cstart['time']
# t1 = 0

## Select muscles to include
# Select 'all', 'none', or any of the following muscles
# Include as one string, with spaces in between muscles
# For left side muscles use _l instead of _r

# AVAILABLE MUSCLES:
# glut_med1_r glut_med2_r glut_med3_r glut_min1_r glut_min2_r glut_min3_r
# semimem_r semiten_r bifemlh_r bifemsh_r sar_r
# add_long_r add_brev_r add_mag1_r add_mag2_r add_mag3_r
# tfl_r pect_r grac_r glut_max1_r glut_max2_r glut_max3_r
# iliacus_r psoas_r quad_fem_r gem_r peri_r rect_fem_r
# vas_med_r vas_int_r vas_lat_r med_gas_r lat_gas_r soleus_r
# tib_post_r flex_dig_r flex_hal_r tib_ant_r per_brev_r
# per_long_r per_tert_r ext_dig_r ext_hal_r
# ercspn_r intobl_r extobl_r

muscles = 'all'

# Run process
print('Calling OpenSim')

for pos, TrialType in enumerate(TrialTypes):
        
    TRCFileName = TrialName[pos] + '.trc'
    tmp_file = pd.read_csv(TRCPathName + TRCFileName, sep='\t', header=4)
    time = list(tmp_file['Unnamed: 1'])
    t1 = time[0]
    tend = time[-1]
    
    if TrialType == 'Static':
    # Run static trial pipeline
    # Run Scale
        if TrunkFlag:
            Model = 'GAIT2392_WithMarkers.osim'
            MarkerSet = 'Scale_MarkerSet.xml'
            MeasurementSet = 'Scale_MeasurementSet.xml'
            IKTaskSet_file = 'Scale_Tasks.xml'
            
            WriteSetupScaleFile(Mass,TRCPathName, TRCFileName, t1, tend, FootLengthR, FootLengthL, Model, MarkerSet, MeasurementSet, IKTaskSet_file)
        else:
            Model = 'GAIT2392_NoTrunk.osim'
            MarkerSet = 'No_trunk_Scale_MarkerSet.xml'
            MeasurementSet = 'No_trunk_Scale_MeasurementSet.xml'
            IKTaskSet_file = 'No_trunk_Scale_Tasks.xml'
            
            WriteSetupScaleFile(Mass,TRCPathName, TRCFileName, t1, tend, FootLengthR, FootLengthL, Model, MarkerSet, MeasurementSet, IKTaskSet_file)
        
        Setup_Scale = TRCPathName + 'Scale/Setup_Scale_' + TRCFileName[:-4] + '.xml'
        cmd_msg = runOpenSim(Setup_Scale)
        
        if cmd_msg.stderr:
            print(' ERROR. ')
            print('Scaling the static model failed')
            print(cmd_msg.stdout)
            quit()
            
        # Run MTLneutral
        muscle = 'all'
        coordinates = 'neutral_coordinates.mot'
        Model = 'GAIT2392_SCALED.osim'
        
        WriteSetupMTLFile(TRCPathName, TRCFileName, muscle, coordinates, t1, 0.005, Model)
        
        Setup_MTL = TRCPathName + '\MTL\Setup_MTL_' + TRCFileName[:-4] + '.xml'
        cmd_msg = runOpenSim(Setup_MTL)
        
        if cmd_msg.stderr:
            print(' ERROR. ')
            print('Calculating neutral lengths failed')
            print(cmd_msg.stdout)
            quit()
    
    elif TrialType == 'Dynamic':
        
        # Run dynamic trial pipeline
    
        # Run IK
        if TrunkFlag:
            IKTaskSet_file = 'IK_Tasks.xml'
            
            WriteSetupIKFile(TRCPathName, TRCFileName, t1, tend, IKTaskSet_file)
        else:
            IKTaskSet_file = 'No_trunk_IK_Tasks.xml'
            
            WriteSetupIKFile(TRCPathName, TRCFileName, t1, tend, IKTaskSet_file)
            
        Setup_IK = TRCPathName + '\IK\Setup_IK_' + TRCFileName[:-4] + '.xml'
        cmd_msg = runOpenSim(Setup_IK)
        
        if cmd_msg.stderr:
            print(' ERROR. ')
            print('Inverse kinematics tool failed')
            print(cmd_msg.stdout)
            quit()
        
        # Run MTL
        muscle = 'all'
        coordinates = 'IK_Output.mot'
        Model = 'GAIT2392_SCALED.osim'
        
        WriteSetupMTLFile(TRCPathName, TRCFileName, muscle, coordinates, t1, tend, Model)
        
        Setup_MTL = TRCPathName + '\MTL\Setup_MTL_' + TRCFileName[:-4] + '.xml'
        cmd_msg = runOpenSim(Setup_MTL)
        
        if cmd_msg.stderr:
            print(' ERROR. ')
            print('Calculating neutral lengths failed')
            print(cmd_msg.stdout)
            quit()
    
        shutil.copyfile( TRCPathName + '\MTL\MTL_output.sto' , TRCPathName + '\MTL\MTL_output' + TrialName[pos] + '.sto')
    
        MTL = pd.read_csv(TRCPathName + '\MTL\MTL_output' + TrialName[pos] + '.sto', sep='\t', header=9)
        MTLdata = [MTL[x].values.tolist() for x in MTL.columns]
        MTLLabel = list(MTL.keys())
        MTLneutral = pd.read_csv(TRCPathName + '\MTL\MTLneutral_output.sto', sep='\t', header=9)
        MTLneutraldata = [MTLneutral[x].values.tolist() for x in MTLneutral.columns]

