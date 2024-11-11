# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 13:39:25 2021

@author: EmilianoPRavera
"""
# ----------- import packages --------------
import numpy as np
import struct
import ctypes
from lxml import etree

# General functions ------------------------
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
# ------------------------------------------

# ------------------------------------------

def readC3D_mhs(FullFileName):
    # GetC3D:	Getting 3D coordinate/analog data from a C3D file
    
    # Input:	FullFileName - file (including path) to be read
    
    # Output:
    # POINTdat            3D-marker data [Nmarkers x NvideoFrames x Ndim(=3)]
    # VideoFrameRate     Frames/sec
    # ANALOGdat      Analog signals [Nsignals x NanalogSamples ]
    # AnalogFrameRate    Samples/sec
    # Event              Event(Nevents).time ..value  ..name
    # ParameterGroup     ParameterGroup(Ngroups).Parameters(Nparameters).data ..etc.
    # CameraInfo         MarkerRelated CameraInfo [Nmarkers x NvideoFrames]
    # ResidualError      MarkerRelated ErrorInfo  [Nmarkers x NvideoFrames]
    
    # AUTHOR(S) AND VERSION-HISTORY
    # MatLab Version:
    # Ver. 1.0 Creation (Alan Morris, Toronto, October 1998) [originally named "getc3d.m"]
    # Ver. 2.0 Revision (Jaap Harlaar, Amsterdam, april 2002)
    
    # Modified by May Liu, Dec 2004. Added the HeaderGroup, timeVector, and
    # changed some of the type parameters (e.g., int8, float32, etc)
    
    # Modified by Michael Schwartz, Dec 2004. Changed data reading; eliminated
    # loops (read in blocks with skip) - dramatically faster (150x - 500x).
    
    # Pyhton Version:
    # Coding by Emiliano Ravera, Sep 2021. Adapted to Python 3.7
    
    POINTdat = []
    VideoFrameRate = 0
    ANALOGdat = []
    AnalogFrameRate = 0
    Event = {}
    ParameterGroup = {}
    CameraInfo = []
    ResidualError = []
    HeaderGroup = {}
    
    # |------------------------------------------|
    # |                                          |
    # |   open the file and get general info     |
    # |                                          |
    # |------------------------------------------|
     
    f = open(FullFileName, 'rb')
    if f.readable() == False:
        return 0
    
    NrecordFirstParameterblock = list(f.read(1)) # Reading record number of parameter section
    key = list(f.read(1)) # key = 80
    
    if key[0] != 80:
        f.close()
        return 0
    
    if not key or not NrecordFirstParameterblock:
        f.close()
        return 0
    
    f.seek(512*(NrecordFirstParameterblock[0] - 1) + 3, 0) # jump to processortype - field
    proctype = list(f.read(1))[0] - 83 # proctype: 1(INTEL-PC); 2(DEC-VAX); 3(MIPS-SUN/SGI)

    # ############################################ #
    # |------------------------------------------| #
    # |                                          | #
    # |               READ HEADER                | #
    # |                                          | #
    # |------------------------------------------| #
    # ############################################ #
    
    f.seek(2, 0)                                                    # set pointer just before word 2
    Nmarkers = struct.unpack('h', f.read(2))[0]                     # word 2: number of markers
    NanalogSamplesPerVideoFrame = struct.unpack('h', f.read(2))[0]  # word 3: number of analog mesurements = chann x #anl frames per video frame
    StartFrame = struct.unpack('h', f.read(2))[0]                   # word 4: # of first video frame
    EndFrame = struct.unpack('h', f.read(2))[0]                     # word 5: # of last video frame
    MaxInterpolationGap = struct.unpack('h', f.read(2))[0]          # word 6: maximun interpolation gap allowed (in frame)                                
    Scale = struct.unpack('f', f.read(4))[0]                        # word 7-8: floating-point scale factor to convert 3D-integers to ref system units
    NrecordDataBlock = struct.unpack('h', f.read(2))[0]             # word 9: starting record number for 3D point and analog data
    NanalogFramesPerVideoFrame = struct.unpack('h', f.read(2))[0]   # word 10: number of analog samples per 3d frame
    VideoFrameRate = struct.unpack('f', f.read(4))[0]               # word 11-12: 3D frame rate
        
    if NanalogFramesPerVideoFrame > 0:
        NanalogChannels = NanalogSamplesPerVideoFrame/NanalogFramesPerVideoFrame
    else:
        NanalogChannels = 0
    
    if Scale < 0: # if the scale value is positive the data type is integer
        DataTypeFloat = True
    else:
        DataTypeFloat = False
    
    AnalogFrameRate = VideoFrameRate*NanalogFramesPerVideoFrame
    
    # ############################################ #
    # |------------------------------------------| #
    # |                                          | #
    # |               READ EVENTS                | #
    # |                                          | #
    # |------------------------------------------| #
    # ############################################ #
    
    # NO ANDA ESTA LECTURA DE EVENTOS, SE USA PARAMETER GROUP (VER DE BORRAR AL FINAL)
    f.seek(298, 0)                                          # place pointer before 150th word (bytes 299 and 300)
    EventIndicator = struct.unpack('h', f.read(2))[0]       # word 150: key value (12345 decimal) indicates 4 char event labels
    
    if EventIndicator == 12345:
        
        Nevents = struct.unpack('h', f.read(2))[0]         # word 151: number of events
        f.seek(2, 1)                                       # skip one word (2 bytes)

        if Nevents > 0:
            Event = {str(i) : {'time' : [], 'value' : [], 'name' : [] } for i in range(Nevents)} 
            
            for key in Event.keys():
                Event[key]['time'] = struct.unpack('d', f.read(8))[0]  # read in event times
            
            f.seek(188*2, 0)
            for key in Event.keys():
                Event[key]['value'] = struct.unpack('h', f.read(2))[0]  # read in event values: 0x00 = ON, 0x01 = OFF
                print(Event[key]['value'])    

            f.seek(198*2, 0)
            for key in Event.keys():
                Event[key]['name'] = struct.unpack('4s', f.read(4))[0]  # read in event names
    
    # ############################################ #
    # |------------------------------------------| #
    # |                                          | #
    # |         READ 1st PARAMETER BLOCK         | #
    # |                                          | #
    # |------------------------------------------| #
    # ############################################ #            
    
    f.seek(512*(NrecordFirstParameterblock[0]-1), 0)
    
    # varios parametrso que no se usan -------------------
    dat1 = struct.unpack('B', f.read(1))[0]
    key2 = struct.unpack('B', f.read(1))[0]   # key = 80
    NparameterRecords =  struct.unpack('B', f.read(1))[0]  # number of parameter blocks to follow
    proctype = struct.unpack('B', f.read(1))[0] - 83 # proctype: 1(INTEL-PC); 2(DEC-VAX); 3(MIPS-SUN/SGI)

    # This is the initial read of Nchar... and Grou... Subsequently, these are
    # read from within the while loop (below).
    
    Ncharacters = struct.unpack('b', f.read(1))[0]    # characters in ground/parameter name
    GroupNumber = struct.unpack('b', f.read(1))[0]    # id number -ve = group / =ve = parameter
    
    ParameterNumberIndex = {}
    
    while Ncharacters > 0:      # While loop to read in parameter section...
        # The end of the parameter record is indicated by <0 characters for group/parameter name
        
        if GroupNumber < 0:

            GroupNumber = np.abs(GroupNumber)
            
            ParameterGroup[str(GroupNumber)] = {}
            
            ParameterGroup[str(GroupNumber)] = {'name': [], 'description': []}
            GroupName = struct.unpack(str(Ncharacters) + 's', f.read(Ncharacters))[0].decode('utf-8')
            ParameterGroup[str(GroupNumber)]['name'] = GroupName    # group name
            
            filepos = f.tell()      # present file position
            offset = struct.unpack('h', f.read(2))[0]   # offset in bytes
            nextrec = filepos + offset      # position of beginning of next record
            
            deschars = struct.unpack('b', f.read(1))[0]     # description characters
            GroupDescription = struct.unpack(str(deschars) + 's', f.read(deschars))[0].decode('utf-8')
            ParameterGroup[str(GroupNumber)]['description'] = GroupDescription    # group description
            
            ParameterNumberIndex[str(GroupNumber)] = 0
            
            f.seek(nextrec, 0)      # pointer to next group
            
        else:
            
            dimension = []
            
            ParameterNumberIndex[str(GroupNumber)] += 1
            
            ParameterNumber = ParameterNumberIndex[str(GroupNumber)]    # index all parameters within a group
                        
            ParameterName = struct.unpack(str(Ncharacters) + 's', f.read(Ncharacters))[0].decode('utf-8')   # name of parameter
            
            if len(ParameterName) > 0:
                
                if 'Parameter' in ParameterGroup[str(GroupNumber)]:
                    new_parameter = { str(ParameterNumber) : {'name': [], 'datatype': [], 'dim': {}, 'data' : {}, 'description' : []}}
                    ParameterGroup[str(GroupNumber)]['Parameter'] = {**ParameterGroup[str(GroupNumber)]['Parameter'] , **new_parameter}
                else:
                    ParameterGroup[str(GroupNumber)]['Parameter'] = {}
                    ParameterGroup[str(GroupNumber)]['Parameter'] = \
                        { str(ParameterNumber) : {'name': [], 'datatype': [], 'dim': {}, 'data' : {}, 'description' : []}}
                        
                ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['name'] = ParameterName   # save parameter name
                
            # read offset
            filepos = f.tell()      # present file position
            offset = struct.unpack('h', f.read(2))[0]   # offset in bytes
            nextrec = filepos + offset      # position of beginning of next record

            # read type
            type_of_data = struct.unpack('b', f.read(1))[0]     # type of data: -1=char/1=byte/2=integer*2/4=real*4
            ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['datatype'] = type_of_data 
            
            # read number of dimensions
            dimnum = struct.unpack('b', f.read(1))[0]
            
            if dimnum == 0:
                datalength = np.abs(type_of_data)
            else:
                mult = 1
                dimension = []
                for j in range(dimnum):
                    dimension.append(struct.unpack('B', f.read(1))[0])
                    mult *= dimension[-1]
                    ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['dim'][str(j)] = dimension[-1]    # save parameter dimension data                
                datalength = np.abs(type_of_data)*mult     # length of data record for multi-dimensional array

            # Read in the data
            # ==================================================================
            #                        CHARACTER
            # ==================================================================
            if type_of_data == -1:      # datatype = char
                
                wordlength = dimension[0]       # length of character word
                
                if dimnum == 2 and datalength > 0:      # parameter(idnumber,index,2).dim>0

                    ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = \
                        {str(j) : struct.unpack(str(wordlength) + 's', f.read(int(wordlength)))[0].decode('utf-8') for j in range(dimension[1])}  # character word data record for 2-D array
                        
                elif dimnum == 1 and datalength > 0:
                    
                    data = struct.unpack(str(wordlength) + 's', f.read(int(wordlength)))[0].decode('utf-8')  # character word data record for 1-D array
                    ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = data
            
            # ==================================================================
            #                        BOOLEAN
            # ==================================================================
            elif type_of_data == 1:         # datatype = 1-byte for boolean
                
                Nparameters = int(datalength/np.abs(type_of_data))
                data = struct.unpack(str(Nparameters) + 'b', f.read(int(Nparameters)))[0]  
                ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = data
            
            # ==================================================================
            #                        INTEGER
            # ==================================================================
            elif type_of_data == 2 and datalength > 0:
                
                Nparameters = int(datalength/np.abs(type_of_data))

                data = struct.unpack(str(Nparameters) + 'h', f.read(int(Nparameters*2)))
                                
                if dimnum > 1:
                    ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = np.reshape(data, tuple(dimension))
                else:
                    if len(data) == 1:
                        ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = data[0]
                    else:
                        ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = np.asarray(data)
            
            # ==================================================================
            #                      FLOATING POINT
            # ==================================================================
            elif type_of_data == 4 and datalength > 0:
                
                Nparameters = int(datalength/np.abs(type_of_data))
                data = struct.unpack(str(Nparameters) + 'f', f.read(int(Nparameters*4)))
                                
                if dimnum > 1:
                    ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = np.reshape(data, tuple(dimension[::-1]))
                else:
                    if len(data) == 1:
                        ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = data[0]
                    else:
                        ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['data'] = np.asarray(data)
            
            deschars = struct.unpack('b', f.read(1))[0]     # description characters
            
            if deschars > 0:

                description = []
                for i in range(deschars):
                    description.append( struct.unpack(str(deschars) + 's', f.read(int(deschars)))[0] )
                ParameterGroup[str(GroupNumber)]['Parameter'][str(ParameterNumber)]['description'] = description
                
            f.seek(nextrec, 0)          # moving ahead to next record
            
        # check group/parameter characters and idnumber to see if more records present
        Ncharacters = struct.unpack('b', f.read(1))[0]      # characters in next group/parameter name
        GroupNumber = struct.unpack('b', f.read(1))[0]      # id number -ve=group / +ve=parameter
    # en of while loop reading parameter section
        
    # check for empty groups
    for key in ParameterGroup:
        if not ParameterGroup[key]['Parameter']:
            ParameterGroup[key]['Parameter']['name'] = 'empty'
            
    # ############################################ #
    # |------------------------------------------| #
    # |                                          | #
    # |                READ DATA                 | #
    # |                                          | #
    # |------------------------------------------| #
    # ############################################ #
    
    NvideoFrames = EndFrame - StartFrame + 1 
    # print(NvideoFrames)
    # Emi's code for reading in the data follows
    # The code assumes Scale < 0, this can be easily modified    
    
    if Nmarkers == 0:
        return 0
    
    # Read in data (marker and analog) as repeated blocks 
    f.seek((NrecordDataBlock - 1)*512, 0)
    if DataTypeFloat:
        tmpf = str(int(4*Nmarkers*NvideoFrames)) + 'f' + str(int(NvideoFrames*NanalogFramesPerVideoFrame*NanalogChannels)) + 'f'   # note four bytes per analog channel data skipped
        tmpb = int((4*Nmarkers*NvideoFrames)*4 + (NvideoFrames*NanalogFramesPerVideoFrame*NanalogChannels)*4)
        tmpDATA = struct.unpack(tmpf, f.read(tmpb))
    else:
        tmpf = str(int(4*Nmarkers*NvideoFrames)) + 'h' + str(int(NvideoFrames*NanalogFramesPerVideoFrame*NanalogChannels)) + 'h'   # note four bytes per analog channel data skipped
        tmpb = int((4*Nmarkers*NvideoFrames)*2 + (NvideoFrames*NanalogFramesPerVideoFrame*NanalogChannels)*2)
        tmpDATA = struct.unpack(tmpf, f.read(tmpb))
    
    # Divide beteewn marker and analog data. For each video frame -> 4 corrd (x,y,z,res) * Nmarkers + NanalogChanels * NanalogFramesPerVideoFrame   
    tmpMKR = []
    tmpANL = []
    if NanalogChannels > 0:
        for frame in range(NvideoFrames):
            # marker
            tm1 = int((4*Nmarkers+NanalogFramesPerVideoFrame*NanalogChannels)*frame)
            tm2 = int((4*Nmarkers+NanalogFramesPerVideoFrame*NanalogChannels)*frame + 4*Nmarkers)
            tmpMKR += list(tmpDATA[tm1:tm2])
            # analog
            ta1 = int((4*Nmarkers+NanalogFramesPerVideoFrame*NanalogChannels)*frame + 4*Nmarkers)
            ta2 = int((4*Nmarkers+NanalogFramesPerVideoFrame*NanalogChannels)*frame + 4*Nmarkers + NanalogFramesPerVideoFrame*NanalogChannels)
            tmpANL += list(tmpDATA[ta1:ta2])
    else:
          tmpMKR = list(tmpDATA)   
    
    # Reshape/resize/reorder data
    # First do the markers
    if tmpMKR:
        # go from one long column to a 4xNmarkersxNvideoFrames matrix
        tmpMKR1 = np.reshape(tmpMKR, (NvideoFrames, Nmarkers, 4))
         
        POINTdat = tmpMKR1[:,:,:3]  # trim off the residual/camera contribution stuff%
    else:
        f.close()
        return 0

    # Get camera contribution/residual
    a = np.fix(tmpMKR1[:,:,3])
    highbyte = np.fix(a/256)
    lowbyte = a - highbyte*256*256
    CameraInfo = highbyte
    ResidualError = lowbyte*np.abs(Scale)
    
    # Reshape the analog data
    if NanalogChannels > 0:

        ANALOGdat = np.reshape(tmpANL, (int(NanalogFramesPerVideoFrame*NvideoFrames), int(NanalogChannels)))
        
        # Scale the analog data
        ANL_offset = np.tile(np.mean(ANALOGdat[0:29,:], axis=0), (int(NanalogFramesPerVideoFrame*NvideoFrames), 1))
        
        # Find  GEN_SCALE (scalar) and SCALE (vector)
        GEN_SCALE = getparam(ParameterGroup, 'ANALOG', 'GEN_SCALE')
        SCALE = getparam(ParameterGroup, 'ANALOG', 'SCALE')
        SCALEmtx = np.tile(SCALE, (int(NanalogFramesPerVideoFrame*NvideoFrames), 1))  
        
        ANALOGdat = GEN_SCALE*((ANALOGdat - ANL_offset)*SCALEmtx)
    
    # close the c3d file
    f.close()
    
    HeaderGroup = { str(key): {'name': [], 'data': [] } for key in range(1,9) }
    
    HeaderGroup['1']['name'] = 'nMarkers'
    HeaderGroup['1']['data'] = Nmarkers
    HeaderGroup['2']['name'] = 'nAnalogChannels'
    HeaderGroup['2']['data'] = int(NanalogChannels)
    HeaderGroup['3']['name'] = 'startFrame'
    HeaderGroup['3']['data'] = StartFrame
    HeaderGroup['4']['name'] = 'endFrame'
    HeaderGroup['4']['data'] = EndFrame
    HeaderGroup['5']['name'] = 'videoSampleRate'
    HeaderGroup['5']['data'] = int(VideoFrameRate)
    HeaderGroup['6']['name'] = 'analogSampleRate'
    HeaderGroup['6']['data'] = int(AnalogFrameRate)
    HeaderGroup['7']['name'] = 'startRecord'
    HeaderGroup['7']['data'] = NrecordDataBlock
    HeaderGroup['8']['name'] = 'maxInterpolationGap'
    HeaderGroup['8']['data'] = MaxInterpolationGap
    
    
    
    return POINTdat, VideoFrameRate, ANALOGdat, AnalogFrameRate, Event, ParameterGroup, CameraInfo, ResidualError, HeaderGroup, tmpMKR

############################################################################## 
def getparam(ParameterGroup, GROUP_NAME, PARAMETER_NAME):
    
    OUTPUT = []
    
    for group in ParameterGroup.keys():
        if ParameterGroup[group]['name'] == GROUP_NAME:
            for param in ParameterGroup[group]['Parameter'].keys():
                if ParameterGroup[group]['Parameter'][param]['name'] == PARAMETER_NAME:
                    if isinstance(ParameterGroup[group]['Parameter'][param]['data'], np.ndarray):
                        OUTPUT = ParameterGroup[group]['Parameter'][param]['data']
                    else:
                        if ParameterGroup[group]['Parameter'][param]['data']:
                            OUTPUT = ParameterGroup[group]['Parameter'][param]['data']                      
    return OUTPUT

##############################################################################

def getlabel(ParameterGroup, GROUP_NAME): 
    
    OUTPUT = []
    
    for group in ParameterGroup.keys():
        if ParameterGroup[group]['name'] == GROUP_NAME:
            for param in ParameterGroup[group]['Parameter'].keys():
                if 'LABELS' in ParameterGroup[group]['Parameter'][param]['name']:
                    if ParameterGroup[group]['Parameter'][param]['data']:
                        tmp = [v.rstrip() for v in ParameterGroup[group]['Parameter'][param]['data'].values()] 
                        OUTPUT += tmp
            
    return OUTPUT


##############################################################################      
def AssignVbl(ParameterGroup, POINTdat, ANALOGdat):
    
    _, nPOINT, _ = POINTdat.shape
    
    # Get the LABELS and count POINTs and ANALOGs
    tmpLabels = getlabel(ParameterGroup, 'POINT')
    Plabels = tmpLabels[:nPOINT] # in case there are more labels than actual points
    
    Pdata = {key: POINTdat[:,pos,:] for pos, key in enumerate(Plabels)}
    
    Alabels = getlabel(ParameterGroup, 'ANALOG')
    
    if not Alabels:
        Adata = []
        return Pdata, Adata
    if all(i != i for i in Alabels):
        Adata = []
        return Pdata, Adata
    
    nANALOG, _ = ANALOGdat.shape
    
    Adata = {key: ANALOGdat[:,pos] for pos, key in enumerate(Alabels)}

    return Pdata, Adata


##############################################################################
def getEVENT(ParameterGroup, VideoFrameRate):
    
    # this function returns the EVENT data as a dictionary with key contex, label, time_in_seconds, frame
    # as well as the number of events
    # example event, numevt = getEVENT(ParameterGroup). The events are chronologically
    # sorted (earliest to latest).
    event = {}
    numevt = 0 
    isAuto = False
    
    # Find the group number for the 'EVENT' group
    nameid = [key for key in ParameterGroup.keys() if ParameterGroup[key]['name'] == 'EVENT'] 
    
    # If there are no EVENTs then return
    if not nameid:
        event = {}
        numevt = 0
        isAuto = np.nan
        return event, numevt, isAuto
    
    # Find the parameter number for the 'CONTEXT' parameter
    munberid = [key for key in ParameterGroup[nameid[0]]['Parameter'].keys() if ParameterGroup[nameid[0]]['Parameter'][key]['name'] == 'CONTEXTS']
    
    # If there are no Contexts for Event then return
    if not munberid:
        event = {}
        numevt = 0
        isAuto = np.nan
        return event, numevt, isAuto
    
    # Get the CONTEXTS
    contexts = [v.rstrip() for v in ParameterGroup[nameid[0]]['Parameter'][munberid[0]]['data'].values()]
    
    # Find the parameter number for the 'LABELS' parameter
    munberid = [key for key in ParameterGroup[nameid[0]]['Parameter'].keys() if ParameterGroup[nameid[0]]['Parameter'][key]['name'] == 'LABELS']
    
    # Get the LABELS
    labels = [v.rstrip() for v in ParameterGroup[nameid[0]]['Parameter'][munberid[0]]['data'].values()]
    
    # Find the parameter number for the 'TIMES' parameter
    munberid = [key for key in ParameterGroup[nameid[0]]['Parameter'].keys() if ParameterGroup[nameid[0]]['Parameter'][key]['name'] == 'TIMES']
    
    # Get the TIMES
    evttimes = ParameterGroup[nameid[0]]['Parameter'][munberid[0]]['data']

    # Find the number of events
    numevt, _ = evttimes.shape
    
    if numevt == 0:
        event['1'] = {'time': -99}
        isAuto = np.nan
        return event, numevt, isAuto
    
    # Convert times to seconds
    timessec = [(evttimes[k,0]*60 + evttimes[k,1]) for k in range(numevt)]
    
    # Find the temporal order of events
    ii = np.argsort(np.array(timessec))
    
    event = { str(k+1): {'context': contexts[ii[k]], \
                         'label': labels[ii[k]], \
                         'time': timessec[ii[k]], \
                         'frame': (int(timessec[ii[k]]*VideoFrameRate) + 1)} \
             for k in range(numevt)}
            
    # AJR Modified for Auto Event Detection
    events_full = event.copy()
           
    if 'Auto Events' in [event[key]['label'] for key in event.keys()]:

        # Get Event Data
        context = [events_full[key]['context'] for key in events_full.keys()]
        label = [events_full[key]['label'] for key in events_full.keys()]
                
        # Get 'Auto Events' events
        ixBauto = [pos for pos, key in enumerate(label) if key == 'Auto Events']
        
        # Get general 'Event' and frame(marks individual cycle we want to keep)
        ixLgen = [pos for pos, key in enumerate(label) if key == 'Event' and context[pos] == 'Left']
        ixRgen = [pos for pos, key in enumerate(label) if key == 'Event' and context[pos] == 'Right']
                        
        # Clear 'Auto Events' and general 'Event' events
        ixClear = ixBauto + ixLgen + ixRgen
        
        for pos in ixClear:
            del events_full[str(pos+1)]
 
        # Find gait cycle nearest the identified general event
        # Get Updated Event Data
        context2 = [events_full[key]['context'] for key in events_full.keys()]
        label2 = [events_full[key]['label'] for key in events_full.keys()]
                
        # Left Foot Strike Nearest General Event (Add 10000 to Right events
        # and Foot Off events to force full index reference)
        tmpl = [pos for pos, key in enumerate(label2) if key == 'Foot Strike' and context2[pos] == 'Left']
        ixLstart = min(tmpl, key=lambda x:abs(x - ixLgen[0]))
                
        # Left Foot Strike Nearest General Event (Add 10000 to Right events
        # and Foot Off events to force full index reference)
        tmpr = [pos for pos, key in enumerate(label2) if key == 'Foot Strike' and context2[pos] == 'Right']
        ixRstart = min(tmpr, key=lambda x:abs(x - ixRgen[0]))
       
        if ixLstart < ixRstart:
            # L first, R second
            ixFirst = ixLstart
            ixSecond = ixRstart
        else:
            ixFirst = ixRstart
            ixSecond = ixLstart
            
        # Events needed for First and Second Cycles
        # Works for both Consecutive/Non-Consecutive Cycles
        ixFirstFull = [*range(ixFirst, ixFirst+6)] # First cycle foot plus next 6 events
        ixSecondFull = [*range(ixSecond-2, ixSecond+4)] # Two events before Second cycle foot strike and next 4 after
        
        # Get full list of events needed
        ixFull = ixFirstFull + list(set(ixSecondFull) - set(ixFirstFull))
        ixFull.sort()
                       
        # Check if we have enough events
        if ixFull[-1] > len(events_full):
            subject = list(getparam(ParameterGroup, 'SUBJECTS', 'NAMES').values())
            Mbox('Error', 'Gait Cycle info Incomplete: ' + subject[0].rstrip(), 0)
            event = np.nan
            numevt = 0
            isAuto = np.nan
            return event, numevt, isAuto
        
        # Events to pass along
        key_event_full = list(events_full.keys())
                
        tmp1 = {}
        for pos, value in enumerate(ixFull):
            tmp1[str(pos+1)] = events_full[key_event_full[value]]
        
        event = tmp1.copy()
        isAuto = True
        numevt = len(event)
        
    else:
    
        event = events_full.copy()
        isAuto = False
        numevt = len(event)
    
    return event, numevt, isAuto


##############################################################################
def getVSKstatparam(vskfile,targetparam):
    
    xDoc = etree.parse(vskfile)
    statparam = [child.get('VALUE') for child in xDoc.findall('.//StaticParameter') if child.get('NAME') == targetparam]
    
    return float(statparam[0])























