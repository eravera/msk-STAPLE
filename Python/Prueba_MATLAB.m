close all; clear all;

mesh = '/home/eravera/Documentos/Investigacion/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_new.stl';

% pelvisTri = load_mesh(mesh);
pelvisTri = load('/home/eravera/Documentos/Investigacion/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_Tri.mat');
pelvisTri = pelvisTri.pelvisTri;
% [RotPseudoISB2Glob, LargestTriangle, BL] = pelvis_guess_CS(pelvisTri, 0);

%   [CS, JCS, FemurBL] = GIBOC_femur(femurTri,...
%                                    side,...
%                                    fit_method,...
%                                    result_plots,...
%                                    debug_plots,...
%                                    in_mm)

femurTri = load('/home/eravera/Documentos/Investigacion/Codigos MATLAB_PYTHON/STAPLE/Python/femur_Tri.mat');
femurTri = femurTri.femurTri;

U_DistToProx = femur_guess_CS( femurTri , 0 )
[CS, JCS, FemurBL] = GIBOC_femur(femurTri);

%figure()
%quickPlotTriang(femurTri)