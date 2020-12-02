%-------------------------------------------------------------------------%
% Copyright (c) 2020 Modenese L.                                          %
%                                                                         %
% Licensed under the Apache License, Version 2.0 (the "License");         %
% you may not use this file except in compliance with the License.        %
% You may obtain a copy of the License at                                 %
% http://www.apache.org/licenses/LICENSE-2.0.                             %
%                                                                         %
% Unless required by applicable law or agreed to in writing, software     %
% distributed under the License is distributed on an "AS IS" BASIS,       %
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or         %
% implied. See the License for the specific language governing            %
% permissions and limitations under the License.                          %
%                                                                         %
%    Author:   Luca Modenese,  2020                                       %
%    email:    l.modenese@imperial.ac.uk                                  %
% ----------------------------------------------------------------------- %
% This example demonstrates how to setup a STAPLE workflow to
% automatically generate a complete kinematic model of the right legs of
% the anatomical datasets used in the manuscript:
% Modenese, Luca, and Jean-Baptiste Renault. "Automatic Generation of
% Personalised Skeletal Models of the Lower Limb from Three-Dimensional
% Bone Geometries." bioRxiv (2020).
% https://www.biorxiv.org/content/10.1101/2020.06.23.162727v2
% and included in the bone_dataset folder.
% ----------------------------------------------------------------------- %
clear; clc; close all
addpath(genpath('STAPLE'));

%----------%
% SETTINGS %
%----------%
output_models_folder = 'opensim_models';
% folder where the various datasets (and their geometries) are located.
datasets_folder = 'bone_datasets';
% datasets that you would like to process
datasets = {'TLEM2_MRI', 'JIA_MRI'};
% format of input geometries
input_geom_format = 'tri';
% visualization geometry format (options: 'stl' or 'obj')
vis_geom_format = 'obj';
% body sides
sides = {'r', 'l'};
% choose the definition of the joint coordinate systems (see documentation)
joint_defs = 'Modenese2018';
%--------------------------------------

tic

% create model folder if required
if ~isfolder(output_models_folder); mkdir(output_models_folder); end

for n_d = 1:numel(datasets)
    
    % current dataset being processed
    cur_dataset = datasets{n_d};
    
    % folder from which triangulations will be read
    tri_folder = fullfile(datasets_folder, cur_dataset, input_geom_format);
    
    for n_side = 1:2
        
        % get current body side
        [sign_side , cur_side] = bodySide2Sign(sides{n_side});
        
        % cell array with the bone geometries that you would like to process
        bones_list = {'pelvis_no_sacrum',  ['femur_', cur_side],...
                     ['tibia_', cur_side],['talus_', cur_side],...
                     ['calcn_', cur_side]};

        % model and model file naming
        cur_model_name = ['auto_',datasets{n_d},'_',upper(cur_side)];
        model_file_name = [cur_model_name, '.osim'];
        
        % log printout
        log_file = fullfile(output_models_folder, [cur_model_name, '.log']);
        logConsolePrintout('on', log_file);
        
        % create geometry set structure for all 3D bone geometries in the dataset
        triGeom_set = createTriGeomSet(bones_list, tri_folder);
        
        % create bone geometry folder for visualization
        geometry_folder_name = [cur_model_name, '_Geometry'];
        geometry_folder_path = fullfile(output_models_folder,geometry_folder_name);
        
        % convert geometries in chosen format (30% of faces for faster visualization)
        writeModelGeometriesFolder(triGeom_set, geometry_folder_path, vis_geom_format,0.3);
        
        % initialize OpenSim model
        osimModel = initializeOpenSimModel(cur_model_name);
        
        % create bodies
        osimModel = addBodiesFromTriGeomBoneSet(osimModel, triGeom_set, geometry_folder_name, vis_geom_format);
        
        % process bone geometries (compute joint parameters and identify markers)
        [JCS, BL, CS] = processTriGeomBoneSet(triGeom_set, cur_side);
        
        % create joints
        createLowerLimbJoints(osimModel, JCS, joint_defs);
        
        % add markers to the bones
        addBoneLandmarksAsMarkers(osimModel, BL);
        
        % finalize connections
        osimModel.finalizeConnections();
        
        % print
        osim_model_file = fullfile(output_models_folder, model_file_name);
        osimModel.print(osim_model_file);
        
        % inform the user about time employed to create the model
        disp('-------------------------')
        disp(['Model generated in ', sprintf('%.1f', toc), ' s']);
        disp(['Saved as ', osim_model_file,'.']);
        disp(['Model geometries saved in folder: ', geometry_folder_path,'.'])
        disp('-------------------------')
        clear triGeom_set JCS BL CS
        logConsolePrintout('off');
        % store model file (with path) for merging
        osim_model_set(n_side) = {osim_model_file}; %#ok<SAGROW>
    end
    % merge the two sides
    merged_model_file = fullfile(output_models_folder,[cur_dataset,'_bilateral.osim']);
    mergeOpenSimModels(osim_model_set{1}, osim_model_set{2}, merged_model_file);

end
% remove paths
rmpath(genpath('STAPLE'));