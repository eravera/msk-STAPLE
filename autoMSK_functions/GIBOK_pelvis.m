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
%    Author: Luca Modenese                                                %
%    email:    l.modenese@imperial.ac.uk                                  % 
% ----------------------------------------------------------------------- %
function [ SCS, JCS, BoneLandmarks] = GIBOK_pelvis(Pelvis, in_mm)

% check units
if nargin<2;     in_mm = 1;  end
if in_mm == 1;     dim_fact = 0.001;  else;  dim_fact = 1; end

% guess of direction of axes on medical images (not always correct)
% Z : pointing cranially
% Y : pointing posteriorly
% X : pointing medio-laterally
% translating this direction in ISB reference system:
x_pelvis_in_global = [0 -1 0];
y_pelvis_in_global = [0 0 1];
z_pelvis_in_global = [-1 0 0];

% building the rot mat from global to pelvis ISB (roughly)
% RGlob2Pelvis = [x_pelvis_in_global; y_pelvis_in_global; z_pelvis_in_global];

% Get eigen vectors V_all and volumetric center
[eigVctrs, CenterVol, InertiaMatrix ] =  TriInertiaPpties(Pelvis);

% clarifying that this rotation goes inertial to global
RInert2Glob = eigVctrs;

% aligning pelvis ref system to ISB one provided
[~, ind_x_pelvis] = max(abs(RInert2Glob'*x_pelvis_in_global'));
[~, ind_y_pelvis] = max(abs(RInert2Glob'*y_pelvis_in_global'));
[~, ind_z_pelvis] = max(abs(RInert2Glob'*z_pelvis_in_global'));

% signs of axes for the largest component
sign_x_pelvis = sign(RInert2Glob'*x_pelvis_in_global');
sign_y_pelvis = sign(RInert2Glob'*y_pelvis_in_global');
sign_z_pelvis = sign(RInert2Glob'*z_pelvis_in_global');
sign_x_pelvis = sign_x_pelvis(ind_x_pelvis);
sign_y_pelvis = sign_y_pelvis(ind_y_pelvis);
sign_z_pelvis = sign_z_pelvis(ind_z_pelvis);

RotISB2Glob = [sign_x_pelvis*eigVctrs(:, ind_x_pelvis),...  
               sign_y_pelvis*eigVctrs(:, ind_y_pelvis),...
               sign_z_pelvis*eigVctrs(:, ind_z_pelvis)];
           
% RInert2ISB = RInert2Glob*RotISB2Glob';

% generating a triangulated pelvis with coordinate system ISB (see comment
% in function). Pseudo because there are still possible errors (see check
% below).
[ PelvisPseudoISB, ~ , ~ ] = TriChangeCS( Pelvis, RotISB2Glob, CenterVol);

% In ISB reference system, points at the right have positive z coordinates
% max z should be ASIS
R_side_ind = PelvisPseudoISB.Points(:,3)>0;
[~, RASIS_ind] = max(PelvisPseudoISB.Points(:,1).*R_side_ind);
[~, LASIS_ind] = max(PelvisPseudoISB.Points(:,1).*~R_side_ind);
[~, RPSIS_ind] = min(PelvisPseudoISB.Points(:,1).*R_side_ind);
[~, LPSIS_ind] = min(PelvisPseudoISB.Points(:,1).*~R_side_ind);

% extract points on bone
RASIS = [Pelvis.Points(RASIS_ind,1),Pelvis.Points(RASIS_ind,2), Pelvis.Points(RASIS_ind,3)];
LASIS = [Pelvis.Points(LASIS_ind,1),Pelvis.Points(LASIS_ind,2), Pelvis.Points(LASIS_ind,3)];
RPSIS = [Pelvis.Points(RPSIS_ind,1),Pelvis.Points(RPSIS_ind,2), Pelvis.Points(RPSIS_ind,3)];
LPSIS = [Pelvis.Points(LPSIS_ind,1),Pelvis.Points(LPSIS_ind,2), Pelvis.Points(LPSIS_ind,3)];

% check if bone landmarks are correctly identified or axes were incorrect
if norm(RASIS-LASIS)<norm(RPSIS-LPSIS)
    % inform user
    disp('GIBOK_pelvis.')
    disp('Inter-ASIS distance is shorter than inter-PSIS distance.')
    disp('Likely error in guessing medical image axes. Flipping X-axis.')
    % switch ASIS and PSIS
    % temp variables
    LPSIS_temp = RASIS;
    RPSIS_temp = LASIS;
    % assign asis
    LASIS = RPSIS;
    RASIS = LPSIS;
    % update psis
    LPSIS = LPSIS_temp;
    RPSIS = RPSIS_temp;
end

% defining the ref system (global)
PelvisOr = (RASIS+LASIS)'/2.0;
Z = normalizeV(RASIS-LASIS);
temp_X = ((RASIS+LASIS)/2.0) - ((RPSIS+LPSIS)/2.0);
pseudo_X = temp_X/norm(temp_X);
Y = normalizeV(cross(Z, pseudo_X));
X = normalizeV(cross(Y, Z));

% segment reference system
SCS.CenterVol = CenterVol;
SCS.InertiaMatrix = InertiaMatrix;
% ISB reference system
SCS.Origin = PelvisOr;
SCS.X = X;
SCS.Y = Y;
SCS.Z = Z;
SCS.V = [X Y Z];

% debug plot
quickPlotTriang(Pelvis, [], 1); hold on
quickPlotRefSystem(SCS)
plotDot(RASIS, 'k', 7)
plotDot(LASIS, 'k', 7)
plotDot(LPSIS, 'r', 7)
plotDot(RPSIS, 'r', 7)

% storing joint details
JCS.ground_pelvis.child_location    = PelvisOr*dim_fact;
JCS.ground_pelvis.child_orientation = computeZXYAngleSeq(SCS.V);
JCS.hip_r.parent_orientation        = computeZXYAngleSeq(SCS.V);

% for debugging purposes
% PlotPelvis_ISB( CSs.ISB, Pelvis); hold on
% plot3(RASIS(1), RASIS(2), RASIS(3),'o')
% plot3(RPSIS(1), RPSIS(2), RPSIS(3),'o')
% plot3(LASIS(1), LASIS(2), LASIS(3),'o')
% plot3(LPSIS(1), LPSIS(2), LPSIS(3),'o')

% defining the ref system (ISB) - TO VERIFY THAT THE TRANSFORMATION WORKED
% EXPECTED: PelvisISB will be aligned with PelvisISBRS (global ISB axes)
% figure
% PelvisISBRS.Origin = [0 0 0]';
% PelvisISBRS.X = [1 0 0];
% PelvisISBRS.Y = [0 1 0];
% PelvisISBRS.Z = [0 0 1];
% PlotPelvis_ISB( PelvisISBRS, PelvisISB )

%% Export identified objects of interest
if nargout > 2
    BoneLandmarks.RASIS     = RASIS; % in Pelvis ref 
    BoneLandmarks.LASIS     = LASIS; % in Pelvis ref 
    BoneLandmarks.RPSIS     = RPSIS; % in Pelvis ref 
    BoneLandmarks.LPSIS     = LPSIS; % in Pelvis ref 
end

end