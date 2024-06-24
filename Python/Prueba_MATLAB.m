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

%U_DistToProx = femur_guess_CS( femurTri , 0 )
%[CS, JCS, FemurBL] = GIBOC_femur(femurTri);

%figure()
%quickPlotTriang(femurTri)



debug_plots = 0;

[ U_DistToProx ] = femur_guess_CS( femurTri, debug_plots);
[ProxFemTri, DistFemTri] = cutLongBoneMesh(femurTri, U_DistToProx);

% Compute the coefficient for morphology operations
CoeffMorpho = computeTriCoeffMorpho(femurTri);

% Get inertial principal vectors V_all of the femur geometry & volum center
[ V_all, CenterVol, InertiaMatrix] = TriInertiaPpties( femurTri );

%-------------------------------------
% Initial Coordinate system (from inertial axes and femoral head):
% * Z0: points upwards (inertial axis) 
% * Y0: points medio-lat (from OT and Z0 in findFemoralHead.m)
%-------------------------------------
% coordinate system structure to store coordinate system's info
AuxCSInfo = struct();
AuxCSInfo.CenterVol = CenterVol;
AuxCSInfo.V_all = V_all;

% Check that the distal femur is 'below' the proximal femur or invert Z0
Z0 = V_all(:,1);
Z0 = sign((mean(ProxFemTri.Points)-mean(DistFemTri.Points))*Z0)*Z0;
AuxCSInfo.Z0 = Z0;

[AuxCSInfo, ~] = Kai2014_femur_fitSphere2FemHead(ProxFemTri, AuxCSInfo, debug_plots);




% -------------------
TriObj = ProxFemTri;
Axis = Z0;
cut_offset = 0.5;
step = 0.5;
min_coord = min(TriObj.Points*Axis)+cut_offset;
max_coord = max(TriObj.Points*Axis)-cut_offset;
Alt = min_coord:step:max_coord;

Areas=[];

it = 1;
for d = -Alt
    [ Curves , Areas(it), ~ ] = TriPlanIntersect(TriObj, Axis, d);
    it = it + 1;
end







