# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 09:31:12 2021

@author: EmilianoPRavera
"""
# ----------- import packages --------------
import numpy as np
# import struct
import ctypes
from lxml import etree
import pandas as pd
from pathlib import Path
import os, shutil
import subprocess
import gc

# from Public_functions import readC3D_mhs, \
#                                 getparam, \
#                                 AssignVbl, \
#                                 getEVENT

# General functions ------------------------
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))
# ------------------------------------------

# ------------------------------------------
def getCyclesMTL(event):
    # identify complete cycles (L and R)
    Lcyccont = ['Left', 'Right', 'Right', 'Left', 'Left']
    Rcyccont = ['Right', 'Left', 'Left', 'Right', 'Right']
    cyclabel = ['Foot Strike', 'Foot Off', 'Foot Strike', 'Foot Off', 'Foot Strike']
    
    Rcycstart = []
    Lcycstart = []
        
    nLcyc = 0
    nRcyc = 0
    
    numevents = len(event)
 
    for k in range(numevents-4):
        cont = [event[key]['context'] for key in list(event.keys())[k:k+5]]     # get actual context
        label = [event[key]['label'] for key in list(event.keys())[k:k+5]]      # get actual label
        
        if cont == Lcyccont:
            if label == cyclabel:
                nLcyc += 1
                Lcycstart.append(k)
        elif cont == Rcyccont:
            if label == cyclabel:
                nRcyc += 1
                Rcycstart.append(k)

    return nLcyc, Lcycstart, nRcyc, Rcycstart


##############################################################################
def N2c3d2trc(PathName, c3dfile, trcfile):
    
    # list of all potential markers (only the ones that are available are
    # written to trc file)
    # see Scale_markerset for required markers
    MLabels = ['RPSI', 'LPSI', 'RASI', 'LASI', \
               'RTHI', 'RUL1', 'RKNE', 'RUL2', \
               'RTIB', 'RLL1','RLL2', \
               'RANK', 'RMM', 'RMMv', 'RTOE', \
               'LTHI', 'LUL1', 'LKNE', 'LUL2', \
               'LTIB', 'LLL1','LLL2', \
               'LANK', 'LMM','LMMv', 'LTOE', \
               'RAC', 'LAC', 'N', 'XYPH','STRN', \
               'HIPL', 'HIPR', 'KNEL', 'KNER','RMFC','LMFC', \
               'LHD', 'RHD', 'LKD', 'RKD','LKAl', 'LKAm', \
               'RKAl','RKAm','LKAld','LKAmd', 'RKAld', 'RKAmd', 'RFEP','RFEO', \
               'RTIO', 'LFEP', 'LFEO', 'LTIO']
        
    # Coordinate (joint angle) labels specific to OpenSim
    CLabels = ['pelvis_tilt','pelvis_list','pelvis_rotation', \
               'pelvis_tx','pelvis_ty','pelvis_tz', \
               'hip_flexion_r','hip_adduction_r','hip_rotation_r', \
               'knee_angle_r','ankle_angle_r', \
               'hip_flexion_l','hip_adduction_l','hip_rotation_l', \
               'knee_angle_l','ankle_angle_l', \
               'lumbar_extension','lumbar_bending','lumbar_rotation', \
               'subtalar_angle_r','mtp_angle_r','subtalar_angle_l','mtp_angle_l']
    
    # time in ms before and after the gait cycle to be used
    # this will be useful for forward simulation where the begining and end of
    # the simulation has the most trouble
    # TimeLag = 0     # currently the function is set to read in the entire trial, ignoring events
    cstart = {}
    cend = {}
    # Read in c3d data
    POINTdat, VideoFrameRate, ANALOGdat, AnalogFrameRate, _, ParameterGroup, _, _, _, _ = readC3D_mhs(c3dfile)
    
    # Assign POINT/ANALOG data to variables (dictionary)
    POINTdata, ANALOGdata = AssignVbl(ParameterGroup, POINTdat, ANALOGdat)
    
    # Find CAMERA_RATE and VIDEO_RATE_DIVIDER
    CAMERA_RATE = getparam(ParameterGroup,'TRIAL','CAMERA_RATE')
    VIDEO_RATE_DIVIDER = getparam(ParameterGroup,'TRIAL','VIDEO_RATE_DIVIDER')
    if not VIDEO_RATE_DIVIDER:
        VIDEO_RATE_DIVIDER = 1
    
    # get subject data
    SubjMass = getparam(ParameterGroup, 'PROCESSING', 'Bodymass')
    if not SubjMass:
        SubjMass = np.nan
        
    # GET GAIT CYCLE DATA
    # get event dictionaty
    # event['contect'], event['label'], event['time'], event['frame']
    event, numevents, _ = getEVENT(ParameterGroup, CAMERA_RATE/VIDEO_RATE_DIVIDER)
    
    # identify complete cycle (L and R)
    nLcyc, Lcycstart, nRcyc, Rcycstart = getCyclesMTL(event)
    
    TrunkFlag = True
    
    if all(marker in POINTdata for marker in ['N', 'RAC', 'LAC']):
        if nRcyc > 0 and nLcyc > 0:
            eventstartframe = [event[str(pos+1)]['frame'] for pos in Lcycstart] + \
                                [event[str(pos+1)]['frame'] for pos in Rcycstart]
            eventendframe = [event[str(pos+5)]['frame'] for pos in Lcycstart] + \
                                [event[str(pos+5)]['frame'] for pos in Rcycstart]
            startframe = int(np.min(eventstartframe))
            endframe = int(np.max(eventendframe))
            
            if (~POINTdata['N'][startframe:endframe, :].any(axis=0)).any():
                TrunkFlag = False
            if (~POINTdata['RAC'][startframe:endframe, :].any(axis=0)).any():
                TrunkFlag = False    
            if (~POINTdata['LAC'][startframe:endframe, :].any(axis=0)).any():
                TrunkFlag = False
    else:
        TrunkFlag = False
    
    if nRcyc == 0 and nLcyc == 0:
        
        TrialType = 'Static'
        if TrunkFlag:
            MArkerSetFile = PathName + 'Scale\\Scale_MarkerSet.xml'
        else:
            MArkerSetFile = PathName + 'Scale\\No_trunk_Scale_MarkerSet.xml'
            
        sms = etree.parse(MArkerSetFile)
        reqmks = [child.get('name') for child in sms.findall('.//Marker')]
    else:
        
        TrialType = 'Dynamic'
        if TrunkFlag:
            MArkerSetFile = PathName + 'IK\\IK_Tasks.xml'
        else:
            MArkerSetFile = PathName + 'IK\\No_trunk_IK_Tasks.xml'
            
        sms = etree.parse(MArkerSetFile)
        reqmks = [child.get('name') for child in sms.findall('.//IKMarkerTask')]
        
    # Actaul Foot Length Only Needed for Static Trial Scaling
    if TrialType == 'Static':
        
        RFootLength = getparam(ParameterGroup, 'PROCESSING', 'RFootLength')
        LFootLength = getparam(ParameterGroup, 'PROCESSING', 'LFootLength')
        if not RFootLength and not LFootLength:
            RFootLength = np.nan
            LFootLength = np.nan
        if np.isnan(RFootLength) or np.isnan(LFootLength):
            if all(marker in POINTdata for marker in ['LTOE', 'LHEE', 'RTOE', 'RHEE']):
                RFootLength = np.mean(np.sqrt(np.diagonal( np.dot((POINTdata['RHEE'] - POINTdata['RTOE']), (POINTdata['RHEE'] - POINTdata['RTOE']).T) )))
                LFootLength = np.mean(np.sqrt(np.diagonal( np.dot((POINTdata['LHEE'] - POINTdata['LTOE']), (POINTdata['LHEE'] - POINTdata['LTOE']).T) )))
            else:
                # Modified 08Oct2015 AJR - can continue if heel marker is
                # missing in static (will scale using TIO to TOE distance)
                RFootLength = 0
                LFootLength = 0
                
                print('FtLenL         FtLenR')
                print(RFootLength, '      ', LFootLength)
                TrialType = 'Foot length not found.'
    else:
        # Modified 27Oct2015 AJR - Set foot length at 0 for dynamic trials
        # (don't need for processing, but need to assign for c3d2trc function)
        RFootLength = 0
        LFootLength = 0
        
    # Marker to offset relative position (O of body coordinate frame)
    Omkr = 'RPSI'
    
    # check for required markers
    for marker in reqmks:
        if marker not in POINTdata:
            print('Marker ' + marker + ' not found.')
            TrialType = 'Marker ' + marker + ' not found.'
            
            SubjMass = -99
            cstart = {}
            cend = {}
            RFootLength = -99
            LFootLength = -99
            return 0
    
    # check for required kinematics
    # Note: only the ankle angles are required for the model but if all of the
    # required markers are visible and the kinematics are calculated, the rest
    # of the kinematics will also be available.
    jointKinematics = ['RPelvisAngles', 'RHipAngles', 'RKneeAngles', \
                       'RAnkleAngles', 'LHipAngles', 'LKneeAngles', 'LAnkleAngles']
    if not all(k in POINTdata for k in jointKinematics):
        print('Kinematics not found.')
        TrialType = 'Kinematics not found.'
        
        SubjMass = -99
        cstart = {}
        cend = {}
        RFootLength = -99
        LFootLength = -99
        return 0
    
    nframes = len(POINTdata[Omkr])

    KneedifL = POINTdata['LFEO'] - POINTdata['KNEL']
    KneedifmagL = np.nanmean(np.sqrt(np.diagonal( np.dot(KneedifL, KneedifL.T) )))                 
    KneedifR = POINTdata['RFEO'] - POINTdata['KNER']
    KneedifmagR = np.nanmean(np.sqrt(np.diagonal( np.dot(KneedifR, KneedifR.T) )))
    if KneedifmagL > 50 or KneedifmagR > 50:
        print('Functional knee axis likely flipped.')
        print('KDifL      KDifR')
        print(KneedifmagL, '      ', KneedifmagR)
        TrialType = 'Fuctional knee axis likely flipped.'
        
        SubjMass = -99
        cstart = {}
        cend = {}
        RFootLength = -99
        LFootLength = -99
        return 0
    
    # Rotation from lab coordinate system to OpenSim coordinate system
    R = np.array([[1, 0, 0],[0, 0, -1],[0, 1, 0]])

    if np.nanmean(np.diff(POINTdata[Omkr][:,0])) < 0 and TrialType == 'Dynamic':
        # walking "backwards" (negative x direction in lab)
        R = np.array([[-1, 0, 0],[0, 0, 1],[0, 1, 0]])

    # Check for data
    mkrindex = np.zeros(len(MLabels))
    frameindex = np.ones(nframes)
    startfield = [ind for ind in range(nframes) if all(POINTdata[Omkr][ind] != 0)][0]
    
    # translate all markers for better visualization in OpenSim
    tvec = np.array([POINTdata[Omkr][startfield,0], POINTdata[Omkr][startfield,1], 0])
    tvecmat = np.tile(tvec.T, (nframes, 1))
    R1mat = np.tile(R[:,0].T, (nframes, 1))
    R2mat = np.tile(R[:,1].T, (nframes, 1))
    R3mat = np.tile(R[:,2].T, (nframes, 1))
    
    data = []
    frameindex = []
    for pos, marker in enumerate(MLabels):
        
        if marker in POINTdata:
            
            tmpd1 = np.diag(np.dot(POINTdata[marker] - tvecmat, R1mat.T))
            tmpd2 = np.diag(np.dot(POINTdata[marker] - tvecmat, R2mat.T))
            tmpd3 = np.diag(np.dot(POINTdata[marker] - tvecmat, R3mat.T))

            data.append(tmpd1)
            data.append(tmpd2)
            data.append(tmpd3)
            
            mkrindex[pos] = 1
            
        if marker in reqmks:
            frameindex.append(list(np.any(POINTdata[marker], axis=1)))
    
    # Select coordinate data; convert from C3D to opensim format
    # nsamples = len(POINTdata[Omkr])
    pos_Omkr = list(POINTdata.keys()).index('RPSI')
    Cdata = [-POINTdata['RPelvisAngles'][:,0], -POINTdata['RPelvisAngles'][:,1], POINTdata['RPelvisAngles'][:,2], \
             data[pos_Omkr]/1000, data[pos_Omkr+1]/1000, data[pos_Omkr+2]/1000, \
             POINTdata['RHipAngles'][:,0], POINTdata['RHipAngles'][:,1], POINTdata['RHipAngles'][:,2], \
             -POINTdata['RKneeAngles'][:,0], POINTdata['RAnkleAngles'][:,0], \
             POINTdata['LHipAngles'][:,0], POINTdata['LHipAngles'][:,1], POINTdata['LHipAngles'][:,2], \
             -POINTdata['LKneeAngles'][:,0], POINTdata['LAnkleAngles'][:,0], \
             POINTdata['LAnkleAngles'][:,0]*0, POINTdata['LAnkleAngles'][:,0]*0, \
             POINTdata['LAnkleAngles'][:,0]*0, POINTdata['LAnkleAngles'][:,0]*0, \
             POINTdata['LAnkleAngles'][:,0]*0, POINTdata['LAnkleAngles'][:,0]*0, \
             POINTdata['LAnkleAngles'][:,0]*0] # last seven correspond to lumbar, subtalar and mtp angles
    
    # use all available frames   
    gapindex = [(i, list(np.where(np.asarray(frame) == False)[0])) for i, frame in enumerate(frameindex) if False in frame]
    if gapindex:
        
        # define gap interval/s for each reqmks
        gapintervals = [(values[0], ranges(values[1])) for values in gapindex]
        
        # if dyanmic trial, chech to make sure a complete gait cycle in included
        if TrialType == 'Dynamic':
            
            eventstartframe = [event[str(pos+1)]['frame'] for pos in Lcycstart] + \
                                [event[str(pos+1)]['frame'] for pos in Rcycstart]
            eventendframe = [event[str(pos+5)]['frame'] for pos in Lcycstart] + \
                                [event[str(pos+5)]['frame'] for pos in Rcycstart]
            startframe = int(np.min(eventstartframe))
            endframe = int(np.max(eventendframe))
            
            tmpsf = []
            tmpef = []
            for mkrgap in gapintervals:
                for interval in mkrgap[1]:
                    size_gap = interval[1] - interval[0]
                    if size_gap > 0:
                        if (interval[1] > startframe and interval[0] < startframe) or \
                            (interval[1] > endframe and interval[0] < startframe) or \
                            (interval[1] > startframe and interval[0] < endframe):
                            print('Gap detected in gait cycle.')
                            TrialType = 'Gap detected in gait cycle.'
            
                            SubjMass = -99
                            cstart = {}
                            cend = {}
                            RFootLength = -99
                            LFootLength = -99
                            return 0
                        elif interval[1] < startframe:
                            tmpsf.append(interval[1] + 1)
                        elif interval[0] > endframe:
                            tmpef.append(interval[0] + 1)

            if tmpsf:
                startfield = max(tmpsf)
            else:
                startfield = 1
            if tmpef:
                endfield = min(tmpef)
            else:
                endfield = len(frameindex[0]) 
    else:
        startfield = 1
        endfield = len(frameindex[0])
                               
    cstart['field'] = startfield
    cend['field'] = endfield
    cstart['time'] = startfield/CAMERA_RATE
    cend['time'] = endfield/CAMERA_RATE
    Ostart = startfield
    Oend = endfield
    
    data = [value[Ostart-1:Oend] for value in data]
    Cdata = [value[Ostart-1:Oend] for value in Cdata]
    
    #
    nvF = Oend - Ostart +1
    Rate = CAMERA_RATE
    Units = 'mm'
    
    Frames = np.arange(1, nvF+1)
    timestamp = (Frames - 1)/Rate
    
    nM = int(np.sum(mkrindex))
    
    # create a dataframe to write with pandas
    DATA = {}
    DATA['Unnamed'] = Frames 
    DATA['Unnamed1'] = timestamp
    
    coord = ['X', 'Y', 'Z']
    coordlabels = [ x + str(y+1) for y in list(range(nM)) for x in coord]
    for pos, label in enumerate(coordlabels):
        DATA[label] = data[pos]
    DATA = pd.DataFrame.from_dict(DATA)
    
    # Write .trc file
    # Generate the header for the .trc file
    line1 = ['PathFileType', '4', '(X/Y/Z)', trcfile] 
    line1 = line1 + ['']*(len(coordlabels) - len(line1) + 2)
    line2 = ['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', 'OrigDataRate', 'OrigDataStartFrame', 'OrigNumFrames'] 
    line2 = line2 + ['']*(len(coordlabels) - len(line2) + 2)
    line3 = [str(Rate), str(Rate), str(nvF), str(nM), Units, str(Rate), str(Ostart), str(nvF)] 
    line3 = line3 + ['']*(len(coordlabels) - len(line3) + 2)
    line4 = ['Frame#', 'Time']
    for pos in range(len(mkrindex)):
        if mkrindex[pos] == True:
            line4.append(MLabels[pos])
            line4.append('')
            line4.append('')
    DATA.columns = ['' if 'Unnamed' in i else i for i in DATA.columns]
    DATA.columns = pd.MultiIndex.from_tuples(zip(line1, line2, line3, line4, DATA.columns))
    
    DATA.to_csv(trcfile, index=False, sep='\t', encoding='utf-8', float_format='%.6f', line_terminator='\t\n')
    
    # Write coordinates.mot file
    
    CDATA = {}
    CDATA['time'] = timestamp
    
    for pos, label in enumerate(CLabels):
        CDATA[label] = Cdata[pos]
    CDATA = pd.DataFrame.from_dict(CDATA)
    
    
    motfile = trcfile[:-4] + '_coordinates.mot'
    motname = Path(motfile).name
    # write header
    # line1 = [motname]
    # line1 = line1 + ['']*(len(Cdata) - len(line1) + 1)
    # line2 = ['nRows='+str(len(Cdata[0]))]
    # line2 = line2 + ['']*(len(Cdata) - len(line2) + 1)
    # line3 = ['nColumns='+str(len(Cdata))]
    # line3 = line3 + ['']*(len(Cdata) - len(line3) + 1)
    # line4 = ['']*(len(Cdata) + 1)
    line5 = ['name', motname]
    line5 = line5 + ['']*(len(Cdata) - len(line5) + 1) 
    line6 = ['datacolumns', str(len(Cdata) + 1)]
    line6 = line6 + ['']*(len(Cdata) - len(line6) + 1) 
    line7 = ['datarows', str(len(Cdata[0]))]
    line7 = line7 + ['']*(len(Cdata) - len(line7) + 1) 
    line8 = ['range', timestamp[0], timestamp[-1]]
    line8 = line8 + ['']*(len(Cdata) - len(line8) + 1) 
    line9 = ['endheader']
    line9 = line9 + ['']*(len(Cdata) - len(line9) + 1)
 
    # CDATA.columns = pd.MultiIndex.from_tuples(zip(line1, line2, line3, line4, \
    #                                               line5, line6, line7, line8, \
    #                                               line9, CDATA.columns))
    CDATA.columns = pd.MultiIndex.from_tuples(zip(line5, line6, line7, line8, \
                                                  line9, CDATA.columns))
    
    CDATA.to_csv(motfile, index=False, sep='\t', encoding='utf-8', float_format='%.6f', line_terminator='\t\n')
    
        
    return SubjMass, TrialType, cstart, cend, RFootLength, LFootLength, TrunkFlag

############################################################################## 
def N2GetActiveSubject(vicon):
    # N2GETACTIVESUBJECT finds the active subject in a Nexus Session
    
    # Apparently one of the only ways to find if a subject is active is to see
    # the subject has any marker trajectories
    
    subnames = vicon.GetSubjectNames()   # List with all subjects
    
    markers = ['RTHI','RTIB','LTHI','LTIB','RPSI','LPSI', \
                'RCA','LCA','RUL1','LUL1','RLL1','LLL1']
    
    for subj in subnames:
        SubjMarkers = vicon.GetMarkerNames(subj)
        
        if set(markers).intersection(set(SubjMarkers)):
            subject = subj
            return subject
        else:
            print('ERROR: No active subject found.')
            quit()

##############################################################################
def WriteSetupScaleFile(mass = 0, StaticPathName = [], StaticFileName = [],\
                        tstart = 0, tend = 1.3, FootLengthR = 0, FootLengthL = 0,\
                        Model = 'GAIT2392_WithMarkers.osim', MarkerSet = 'Scale_MarkerSet.xml',\
                        MeasurementSet = 'Scale_MeasurementSet.xml', IKTaskSet_file = 'Scale_Tasks'):
    
    if not mass:
        print('No mass defined')
        return 0
    if not StaticFileName:
        print('No file name defined')
        return 0
    if not StaticPathName:
        print('No path name defined')
        return 0
    
    # Calculate Foot Scale factors
    if FootLengthR and FootLengthL: # both have a value other than 0
        FootScaleFactorR = (FootLengthR/1000) / 0.1623 # 0.1623 is foot length of OpenSim model
        FootScaleFactorL = (FootLengthL/1000) / 0.1623   
    
    # Load generic setup scale file
    ScaleSetup = etree.parse(StaticPathName + 'Scale/Setup_Scale.xml')
    element = ScaleSetup.find("ScaleTool")
    
    element.set('name', StaticFileName[:-7])
    
    ScaleSetup.find('ScaleTool/mass').text = str(mass)
    ScaleSetup.find('ScaleTool/GenericModelMaker/model_file').text = StaticPathName + 'Scale/' + Model
    ScaleSetup.find('ScaleTool/GenericModelMaker/marker_set_file').text = StaticPathName + 'Scale/' + MarkerSet
    ScaleSetup.find('ScaleTool/ModelScaler/marker_file').text = StaticPathName + StaticFileName
    ScaleSetup.find('ScaleTool/ModelScaler/time_range').text = str(tstart) + '    ' + str(tend)
    
    ScaleSetup.find('ScaleTool/ModelScaler/MeasurementSet').set('file', MeasurementSet)
    
    if FootLengthL and FootLengthR:
        
        ScaleSetup.find('ScaleTool/ModelScaler/ScaleSet').set('name', 'Feet')
        ScaleSet = ScaleSetup.find('ScaleTool/ModelScaler/ScaleSet/objects')
        
        for segm in ['calcn_r', 'toes_r']:
            ScaleFoot = etree.SubElement(ScaleSet, 'Scale', name='foot_r')
            scales = etree.SubElement(ScaleFoot, 'scales')
            scales.text = str(FootScaleFactorR) + '    '  + str(FootScaleFactorR) + '    ' + str(FootScaleFactorR)
            segment = etree.SubElement(ScaleFoot, 'segment')
            segment.text = segm 
            apply = etree.SubElement(ScaleFoot, 'apply')
            apply.text = 'true' 
        
        for segm in ['calcn_l', 'toes_l']:
            ScaleFoot = etree.SubElement(ScaleSet, 'Scale', name='foot_l')
            scales = etree.SubElement(ScaleFoot, 'scales')
            scales.text = str(FootScaleFactorL) + '    '  + str(FootScaleFactorL) + '    ' + str(FootScaleFactorL)
            segment = etree.SubElement(ScaleFoot, 'segment')
            segment.text = segm 
            apply = etree.SubElement(ScaleFoot, 'apply')
            apply.text = 'true'
    
    ScaleSetup.find('ScaleTool/MarkerPlacer/IKTaskSet').set('file', IKTaskSet_file)
    ScaleSetup.find('ScaleTool/MarkerPlacer/marker_file').text = StaticPathName + StaticFileName
    ScaleSetup.find('ScaleTool/MarkerPlacer/coordinate_file').text = StaticPathName + StaticFileName[:-4] + '_coordinates.mot'
    ScaleSetup.find('ScaleTool/MarkerPlacer/time_range').text = str(tstart) + '    ' + str(tend)
    
    # Set output
    ScaleSetup.find('ScaleTool/ModelScaler/output_scale_file').text = StaticPathName + 'Scale/ScaleSet_Applied.xml'
    ScaleSetup.find('ScaleTool/ModelScaler/output_model_file').text = StaticPathName + 'Scale/Model_scaledOnly.osim'
    ScaleSetup.find('ScaleTool/MarkerPlacer/output_model_file').text = StaticPathName + 'Scale/GAIT2392_SCALED.osim'
    ScaleSetup.find('ScaleTool/MarkerPlacer/output_motion_file').text = StaticPathName + 'Scale/Static_Output.mot'
            
    ScaleSetup.write(StaticPathName + 'Scale/Setup_Scale_' + StaticFileName[:-4] + '.xml', encoding='utf-8', xml_declaration=True)
    
    return 0

##############################################################################
def WriteSetupMTLFile(TrialPathName = [], TrailFileName = [], muscle = 'all',\
                      coordinates = 'Unassigned', tstart = 0, tend = 0.005, \
                      Model = 'GAIT2392_WithMarkers.osim'):
    if not TrailFileName:
        print('No file name defined')
        return 0
    if not TrialPathName:
        print('No path name defined')
        return 0
    if coordinates == 'Unassigned':
        print('No coordinates file defined')
        return 0
    
    # Load generic setup MTL file
    MTLSetup = etree.parse(TrialPathName + 'MTL\Setup_MTL.xml')
    element = MTLSetup.find("AnalyzeTool")
    
    if 'neutral_coordinates.mot' in coordinates:
        element.set('name', 'MTLneutral')
    else:
        element.set('name', 'MTL')
    
    MTLSetup.find('AnalyzeTool/model_file').text = TrialPathName + 'Scale\GAIT2392_SCALED.osim'
    if 'neutral' in coordinates:
        MTLSetup.find('AnalyzeTool/coordinates_file').text = TrialPathName + 'MTL\\' + coordinates
    else:
        MTLSetup.find('AnalyzeTool/coordinates_file').text = TrialPathName + 'IK\\' + coordinates
    MTLSetup.find('AnalyzeTool/initial_time').text = str(tstart)
    MTLSetup.find('AnalyzeTool/final_time').text = str(tend)
    MTLSetup.find('AnalyzeTool/AnalysisSet/objects/MuscleAnalysis/muscle_list').text = muscle
    
    # Set output
    MTLSetup.find('AnalyzeTool/results_directory').text = TrialPathName + 'MTL\Results'
    
    MTLSetup.write(TrialPathName + 'MTL\Setup_MTL_' + TrailFileName[:-4] + '.xml', encoding='utf-8', xml_declaration=True)
    
    return 0

##############################################################################
def WriteSetupIKFile(TrialPathName = [], TrailFileName = [], tstart = 0, tend = 1.3, \
                     IKTaskSet_file = 'IK_Tasks.xml'):
    
    if not TrailFileName:
        print('No file name defined')
        return 0
    if not TrialPathName:
        print('No path name defined')
        return 0
    
    # Load generic setup scale file
    IKSetup = etree.parse(TrialPathName + 'IK\Setup_IK.xml')
    element = IKSetup.find("InverseKinematicsTool")
    
    element.set('name', TrailFileName[:-7])
    
    IKSetup.find('InverseKinematicsTool/results_directory').text = TrialPathName + 'IK'
    IKSetup.find('InverseKinematicsTool/model_file').text = TrialPathName + 'Scale\GAIT2392_SCALED.osim'
    IKSetup.find('InverseKinematicsTool/IKTaskSet').set('file', TrialPathName + 'IK\\' + IKTaskSet_file)
    IKSetup.find('InverseKinematicsTool/marker_file').text = TrialPathName + TrailFileName
    IKSetup.find('InverseKinematicsTool/coordinate_file').text = TrialPathName + TrailFileName[:-4] + '_coordinates.mot'
    IKSetup.find('InverseKinematicsTool/time_range').text = str(tstart) + '    ' + str(tend)
    
    # Set output
    IKSetup.find('InverseKinematicsTool/output_motion_file').text = TrialPathName + 'IK\IK_Output.mot'
    
    IKSetup.write(TrialPathName + 'IK\Setup_IK_' + TrailFileName[:-4] + '.xml', encoding='utf-8', xml_declaration=True)
    
    return 0

##############################################################################
def runOpenSim(SetupFile = 'None.xml'):
    # runOpenSim calls tools in OpenSim (secified with 'SetupFile.xml'), 
    # which can be each of the following:
    
    # 1) 'Scale': Scales the generic model (Gait2394) to the subject's size 
    #     using a static file
      
    # 2) 'IK': Runs inverse kinematics, tracking the motion of the subject 
    
    # 3) 'MTL': Runs an analysis on the IK outcomes to calculate muscle-tendon
    #    lengths (MTLs)
    
    # 4) 'MTLneutral': Runs another MTL analysis to find the neutral lengths
    #    of the muscles at anatomical posture (with all joints angles zero)
    
    # INPUTS:   - GAIT2392.osim file for the generic model, 
    #             containing Gillette markerset
    #           - .trc files for static and dynamic trials
    #           - .xml setup files for SCALE (Setup_scale, Scale_Tasks,
    #             Scale_MarkerSet, Scale_MeasurementSet)
    #           - .xml setup files for IK (Setup_IK, IK_Tasks)
    #           - .xml setup file for MTL (Setup_MTL)
    
    # OUTPUTS:  - GAIT2394_SCALED.osim file containing the scaled model
    #           - .mot and .sto file containing coordinates for the static and dynamic
    #             trials
    #           - .sto file containing absolute MTLs
    #           - .sto file containing neutral MTLs
    SetupPath = Path(SetupFile).resolve()
    SetupPath = SetupPath.parent
    SetupName = Path(SetupFile).name
         
    if 'Scale' in SetupName:
        # Run Scale tool    
        logfile = str(SetupPath) + '/Scale_Output.log'
        if os.path.isfile(logfile):
            os.remove(logfile)

        print('     Running scale tool...')

        results = subprocess.run(['opensim-cmd', 'run-tool', SetupFile], shell=True, capture_output=True)
        gc.collect()
        
        if results.stderr:
            print('WARNING: Call to OpenSim "Scale" failed.')
            print(results.stdout)
        else:
            print('     Scale tool succesfully run')
            print('  ')
            shutil.copyfile(str(SetupPath) + '/opensim.log', logfile)   # out is saved as *_Output.log 
            os.remove(str(SetupPath) + '/opensim.log')  # out is empty
    
    if 'MTL' in SetupName:
        # Run MTL tool
        MTLSetup = etree.parse(SetupFile)
 
        if 'neutral_coordinates.mot' in MTLSetup.find('AnalyzeTool/coordinates_file').text:
            MTLcase = 'MTLneutral'
        else:
            MTLcase = 'MTL'
        
        logfile = str(SetupPath) + '\\' + MTLcase + '_Outpu.log'
        if os.path.isfile(logfile):
            os.remove(logfile)
                    
        print('     Running ' + MTLcase + ' tool...')

        results = subprocess.run(['opensim-cmd', 'run-tool', SetupFile], shell=True, capture_output=True)
        gc.collect()
        
        if results.stderr:
            print('WARNING: Call to OpenSim "MTL" failed.')
            print(results.stdout)
        else:
            print('     ' + MTLcase + ' tool succesfully run')
            print('  ')
            # only save MTL file (no other muscle params calculated)
            shutil.copyfile(str(SetupPath) + '\Results\\' + MTLcase + '_MTL_Length.sto', str(SetupPath) + '\\' + MTLcase + '_output.sto')
            shutil.rmtree(str(SetupPath) + '\Results')
            
            # Move opensim.log to MTL folder
            abspath = os.path.abspath(__file__)
            dname = os.path.dirname(abspath)
            shutil.copyfile(dname + '\opensim.log', logfile)   # out is saved as *_Output.log 
            os.remove(dname + '\opensim.log')  # out is empty

    if 'IK' in SetupName:
        # Run Scale tool    
        logfile = str(SetupPath) + '\IK_Output.log'
        if os.path.isfile(logfile):
            os.remove(logfile)

        print('     Running IK tool...')

        results = subprocess.run(['opensim-cmd', 'run-tool', SetupFile], shell=True, capture_output=True)
        gc.collect()
        
        if results.stderr:
            print('WARNING: Call to OpenSim "IK" failed.')
            print(results.stdout)
        else:
            print('     IK tool succesfully run')
            print('  ')
            shutil.copyfile(str(SetupPath) + '\opensim.log', logfile)   # out is saved as *_Output.log 
            os.remove(str(SetupPath) + '\opensim.log')  # out is empty

    return results



























