close all; clear all;

mesh = '/home/eravera/Documentos/Investigacion/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_new.stl';

% pelvisTri = load_mesh(mesh);
pelvisTri = load('/home/eravera/Documentos/Investigacion/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_Tri.mat');
pelvisTri = pelvisTri.pelvisTri;
[RotPseudoISB2Glob, LargestTriangle, BL] = pelvis_guess_CS(pelvisTri, 0);