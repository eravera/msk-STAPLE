#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:58:04 2024

@author: emi
"""
import numpy as np
from stl import mesh
import fast_simplification
import os, shutil
import scipy.spatial as spatial
from scipy.spatial import ConvexHull
from pykdtree.kdtree import KDTree
from pathlib import Path
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import path as mpl_path
# from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.optimize import nnls
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA


from Public_functions import load_mesh, freeBoundary, PolyArea, inpolygon

from algorithms import pelvis_guess_CS, STAPLE_pelvis, femur_guess_CS, GIBOC_femur_fitSphere2FemHead, \
    Kai2014_femur_fitSphere2FemHead, GIBOC_isolate_epiphysis, GIBOC_femur_processEpiPhysis, \
    GIBOC_femur_getCondyleMostProxPoint, GIBOC_femur_smoothCondyles, GIBOC_femur_filterCondyleSurf, \
    GIBOC_femur_ArticSurf, CS_femur_SpheresOnCondyles, CS_femur_CylinderOnCondyles, \
    GIBOC_femur, tibia_guess_CS, tibia_identify_lateral_direction, Kai2014_tibia

from GIBOC_core import plotDot, TriInertiaPpties, TriReduceMesh, TriFillPlanarHoles,\
    TriDilateMesh, cutLongBoneMesh, computeTriCoeffMorpho, TriUnite, sphere_fit, \
    TriErodeMesh, TriKeepLargestPatch, TriOpenMesh, TriPlanIntersect, quickPlotRefSystem, \
    TriSliceObjAlongAxis, fitCSA, LargestEdgeConvHull, PCRegionGrowing, lsplane, \
    fit_ellipse, PtsOnCondylesFemur, TriVertexNormal, TriCurvature, TriConnectedPatch, \
    TriCloseMesh, TriDifferenceMesh, cylinderFitting, TriMesh2DProperties, plotCylinder, \
    TriChangeCS, plotTriangLight, plotBoneLandmarks, PlanPolygonCentroid3D, \
    getLargerPlanarSect

from opensim_tools import computeXYZAngleSeq, getJointParams, getJointParams3DoFKnee, \
    assembleJointStruct, verifyJointStructCompleteness, createCustomJointFromStruct

from geometry import bodySide2Sign, landmarkBoneGeom, compileListOfJointsInJCSStruct, \
    jointDefinitions_auto2020, jointDefinitions_Modenese2018, inferBodySideFromAnatomicStruct
    
from anthropometry import mapGait2392MassPropToModel, gait2392MassProps

# np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


path = Path(abspath)
print(path.parent.absolute())

tmp_path = path._parts
i = [ind for ind, val in enumerate(tmp_path) if val == 'Codigos MATLAB_PYTHON'][0]

ruta = ''
for pos in range(i+2):
    ruta += tmp_path[pos]
    if tmp_path[pos] != '/':
        ruta += '/'

# triangulation en matlab (https://www.mathworks.com/help/matlab/math/triangulation-representations.html)

# P = [ 2.5    8.0
#       6.5    8.0
#       2.5    5.0
#       6.5    5.0
#       1.0    6.5
#       8.0    6.5];

# Define the connectivity, T.

# T = [5  3  1;
#      3  2  1;
#      3  4  2;
#      4  6  2];

# Create a triangulation from this data.

# TR = triangulation(T,P)

# TR = 
#   triangulation with properties:

#               Points: [6x2 double]
#               ConnectivityList: [4x3 double]
# --------------------------------------------------
# Access the properties in a triangulation in the same way you access the fields of a struct. For example, examine the Points property, which contains the coordinates of the vertices.

# TR.Points

# ans = 6×2

#     2.5000    8.0000
#     6.5000    8.0000
#     2.5000    5.0000
#     6.5000    5.0000
#     1.0000    6.5000
#     8.0000    6.5000

# Next, examine the connectivity.

# TR.ConnectivityList

# ans = 4×3

#      5     3     1
#      3     2     1
#      3     4     2
#      4     6     2

# tri_geom = mesh.Mesh.from_file(ruta + 'bone_datasets/TLEM2/stl/pelvis.stl')

# Faces = tri_geom.vectors
# P = Faces.reshape(-1, 3)
# Vertex = np.zeros(np.shape(tri_geom.v0), dtype=np.int64)
# # Vertex1 = np.zeros(np.shape(tri_geom.v0), dtype=np.int64)

# _, idx = np.unique(P, axis=0, return_index=True)
# Points = P[np.sort(idx)]

# # Vertex1[:,0] = [np.where(Points == elem)[0][0] for elem in tri_geom.v0]
# # Vertex1[:,1] = [np.where(Points == elem)[0][0] for elem in tri_geom.v1]
# # Vertex1[:,2] = [np.where(Points == elem)[0][0] for elem in tri_geom.v2]

# for pos, elem in enumerate(tri_geom.v0):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,0] = tmp1[0]
#     else:
#         Vertex[pos,0] = tmp[0]
        
# for pos, elem in enumerate(tri_geom.v1):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,1] = tmp1[0]
#     else:
#         Vertex[pos,1] = tmp[0]

# for pos, elem in enumerate(tri_geom.v2):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,2] = tmp1[0]
#     else:
#         Vertex[pos,2] = tmp[0]

# # tmp_t0 = [np.where(Points == elem) for elem in tri_geom.v0]
# # tmp_t1 = [np.where(Points == elem) for elem in tri_geom.v1]
# # tmp_t2 = [np.where(Points == elem) for elem in tri_geom.v2]

# triangle = {'Points': Points, 'ConnectivityList': Vertex}


# # Creating Mesh objects from a list of vertices and faces
# # Create the mesh
# new_mesh = mesh.Mesh(np.zeros(Vertex.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(Vertex):
#     # print(f)
#     for j in range(3):
#         new_mesh.vectors[i][j] = Points[f[j],:]

# aux = new_mesh.vectors

# # Write the mesh to file "pelvis_new.stl"
# new_mesh.save(ruta + 'Python/pelvis_new.stl')

# # # check differences
# # tri_geom1 = mesh.Mesh.from_file('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/pelvis_new.stl')
# # Faces1 = tri_geom1.vectors
# # l0 = []
# # l1 = []
# # l2 = []
# # for i in range(len(Faces)):
# #     tmp_norm = np.linalg.norm(Faces[i] - Faces1[i])
# #     if tmp_norm > 0:
# #         print(i, tmp_norm)
# #         l0 = list(tmp_t0[i][0])
# #         print('t0', tmp_t0[i], [x for x in l0 if l0.count(x) > 1])
# #         l1 = list(tmp_t1[i][0])
# #         print('t1', tmp_t1[i], [x for x in l1 if l1.count(x) > 1])
# #         l2 = list(tmp_t2[i][0])
# #         print('t2', tmp_t2[i], [x for x in l2 if l2.count(x) > 1])


# # reduce number of triangles

# points_out, faces_out = fast_simplification.simplify(Points, Vertex, 0.7) # 30%

# new_mesh1 = mesh.Mesh(np.zeros(faces_out.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces_out):
#     for j in range(3):
#         new_mesh1.vectors[i][j] = points_out[f[j],:]

# aux = new_mesh1.vectors

# # Write the mesh to file "pelvis_new.stl"
# new_mesh1.save(ruta + 'Python/pelvis_new_simplify.stl')


# # triangle = []
# # for vert in range(0,len(Faces),3):
# #     vert1 = []
# #     vert2 = []
# #     vert3 = []
    
# #     for pos, elem in enumerate(P):
# #         if np.array_equal(elem, Faces[vert,0,:]):
# #             vert1.append(pos)
# #         if np.array_equal(elem, Faces[vert+1,0,:]):
# #             vert2.append(pos)
# #         if np.array_equal(elem, Faces[vert+2,0,:]):
# #             vert3.append(pos)
# #     triangle.append([vert1[0], vert2[0], vert3[0]])
    

# # print('Triangle 2: \n')
# # print(np.where(P == Faces[1,0,:]))
# # print(np.where(P == Faces[1,1,:]))
# # print(np.where(P == Faces[1,2,:]))

# # print('Triangle 3: \n')
# # print(np.where(P == Faces[2,0,:]))
# # print(np.where(P == Faces[2,1,:]))
# # print(np.where(P == Faces[2,2,:]))

# # print('Triangle 4: \n')
# # print(np.where(P == Faces[3,0,:]))
# # print(np.where(P == Faces[3,1,:]))
# # print(np.where(P == Faces[3,2,:]))

# # print('Triangle 5: \n')
# # print(np.where(P == Faces[4,0,:]))
# # print(np.where(P == Faces[4,1,:]))
# # print(np.where(P == Faces[4,2,:]))

# # print('Triangle 6: \n')
# # print(np.where(P == Faces[5,0,:]))
# # print(np.where(P == Faces[5,1,:]))
# # print(np.where(P == Faces[5,2,:]))

# # for pos, elem in enumerate(P):
# #     pr

# # vertex = Faces[0]

# # triangle_index = 0
# # vertex_indices = Faces[triangle_index]

# # vertices_xyz = Points.transpose()


# # # import numpy as np

# # points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])

# # from scipy.spatial import Delaunay

# # tri = Delaunay(points)
# # # tri = Delaunay(Points)

# # import matplotlib.pyplot as plt

# # plt.triplot(points[:,0], points[:,1], tri.simplices)

# # plt.plot(points[:,0], points[:,1], 'o')

# # plt.show()

# # print('polyhedron(faces = [')
# # #for vert in tri.triangles:
# # for vert in tri.simplices:
# #     print('[%d,%d,%d],' % (vert[0],vert[1],vert[2]), '], points = [')
# # for i in range(x.shape[0]):
# #     print('[%f,%f,%f],' % (x[i], y[i], z[i]), ']);')


# # fig = plt.figure()
# # ax = fig.add_subplot(1, 1, 1, projection='3d')

# # # The triangles in parameter space determine which x, y, z points are
# # # connected by an edge
# # #ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
# # ax.plot_trisurf(aux[:,0], aux[:,1], aux[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)


# # plt.show()





# # import open3d as o3d

# # mesh = o3d.io.read_triangle_mesh('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/bone_datasets/TLEM2/stl/pelvis.stl')
# # mesh = mesh.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh], window_name="STL", left=1000, top=200, width=800, height=650, mesh_show_wireframe=True)

# # mesh1 = o3d.io.read_triangle_mesh('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_new.stl')
# # mesh1 = mesh1.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh1], window_name="STL NEW", left=1000, top=200, width=800, height=650, mesh_show_wireframe=True)

# # mesh1 = o3d.io.read_triangle_mesh('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_new_simplify.stl')
# # mesh1 = mesh1.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh1], window_name="STL NEW", left=1000, top=200, width=800, height=650, mesh_show_wireframe=True)

# # from mpl_toolkits import mplot3d
# # from matplotlib import pyplot
# # # Create a new plot
# # figure = pyplot.figure()
# # axes = mplot3d.Axes3D(figure)

# # # Load the STL files and add the vectors to the plot
# # your_mesh = mesh.Mesh.from_file('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/pelvis.stl')
# # axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# # # Auto scale to the mesh size
# # scale = your_mesh.points.flatten()
# # # axes.auto_scale_xyz(scale, scale, scale)

# # # Show the plot to the screen
# # pyplot.show()
# #%%
# # A = np.array([[1, 2, 10], [3, 4, 20], [9, 6, 15]])

# # triangle['Points']

# hull = ConvexHull(triangle['Points'])

# HullPoints = hull.points[hull.vertices]
# HullConect = hull.simplices

# # hull object doesn't remove unreferenced vertices
# # create a mask to re-index faces for only referenced vertices
# vid = np.sort(hull.vertices)
# mask = np.zeros(len(hull.points), dtype=np.int64)
# mask[vid] = np.arange(len(vid))
# # remove unreferenced vertices here
# faces = mask[hull.simplices].copy()
# # rescale vertices back to original size
# vertices = hull.points[vid].copy()



# # Hulltriangle = {'Points': HullPoints, 'ConnectivityList': HullConect}
# Hulltriangle = {'Points': vertices, 'ConnectivityList': faces}






# # Convert tiangulation dict to mesh object
# Tr_mesh = mesh.Mesh(np.zeros(Hulltriangle['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(Hulltriangle['ConnectivityList']):
#     for j in range(3):
#         Tr_mesh.vectors[i][j] = Hulltriangle['Points'][f[j],:]


# # aux = sum(TR.incenter.*repmat(Properties.Area,1,3),1)/Properties.TotalArea;

# tmp1 = new_mesh1.areas
# tmp2 = new_mesh1.centroids

# tmp_meanNormal = np.sum(new_mesh1.get_unit_normals()*tmp1/np.sum(tmp1),0)

# center = np.sum(tmp2*tmp1/np.sum(tmp2), 0)

# # print(np.were(tmp2, np.min(tmp2-center)))

# tree = KDTree(triangle['Points'])
# # pts = np.array([[0, 0.2, 0.2]])
# pts = np.array([center])
# dist, idx = tree.query(pts)
# print(triangle['Points'][idx])


# TMPtriangle = {}
# TMPtriangle['Points'] = points_out
# TMPtriangle['ConnectivityList'] = faces_out

# I = np.argmax(tmp1)
# aux1 = TMPtriangle['Points'][TMPtriangle['ConnectivityList'][I]]


# eigenvalues, eigenvectors = np.linalg.eig(np.array([[1, 1, 2], [1, 2, 2], [1, 3, 0]]))

# aux2 = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])

# normalized_arr = preprocessing.normalize(aux2,axis=0)
# print(normalized_arr)

# aux3 = np.dot(TMPtriangle['Points'],aux2)


# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.plot_trisurf(TMPtriangle['Points'][:,0], TMPtriangle['Points'][:,1], TMPtriangle['Points'][:,2], triangles = TMPtriangle['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'red')
# ax.plot_trisurf(Hulltriangle['Points'][:,0], Hulltriangle['Points'][:,1], Hulltriangle['Points'][:,2], triangles = Hulltriangle['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.2, shade=False, color = 'b')

# plt.show()



# soa = np.array([[0, 0, 1, 1, -2, 0], [0, 0, 2, 1, 1, 0],
#                 [0, 0, 3, 2, 1, 0], [0, 0, 4, 0.5, 0.7, 0]])

# X, Y, Z, U, V, W = zip(*soa)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.quiver(X, Y, Z, U, V, W)
# ax.set_xlim([-1, 0.5])
# ax.set_ylim([-1, 1.5])
# ax.set_zlim([-1, 8])
# plt.show()

#%% prueba de funcion pelvis_gess_CS
BCS = {}
JCS = {}
BL = {}
# pelvisTri = load_mesh(ruta + 'bone_datasets/TLEM2/stl/pelvis.stl')
pelvisTri = load_mesh(ruta + 'Python/pelvis_new_simplify.stl')
# pelvisTri = load_mesh(ruta + 'Python/pelvis_new.stl')

# RotPseudoISB2Glob, LargestTriangle, BL = pelvis_guess_CS(pelvisTri, 0)

BCS['pelvis'], JCS['pelvis'], BL['pelvis'] = STAPLE_pelvis(pelvisTri,'right', 0)


# plotDot(TMPtriangle['Points'][10,:], 'k', 7)


#%% prueba de funcion femur_guess_CS

# # reduce number of triangles

# tri_geom = mesh.Mesh.from_file(ruta + 'bone_datasets/TLEM2/stl/femur_r.stl')

# Faces = tri_geom.vectors
# P = Faces.reshape(-1, 3)
# Vertex = np.zeros(np.shape(tri_geom.v0), dtype=np.int64)

# _, idx = np.unique(P, axis=0, return_index=True)
# Points = P[np.sort(idx)]

# for pos, elem in enumerate(tri_geom.v0):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,0] = tmp1[0]
#     else:
#         Vertex[pos,0] = tmp[0]
        
# for pos, elem in enumerate(tri_geom.v1):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,1] = tmp1[0]
#     else:
#         Vertex[pos,1] = tmp[0]

# for pos, elem in enumerate(tri_geom.v2):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,2] = tmp1[0]
#     else:
#         Vertex[pos,2] = tmp[0]


# points_out, faces_out = fast_simplification.simplify(Points, Vertex, 0.9) # 30%

# new_mesh1 = mesh.Mesh(np.zeros(faces_out.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces_out):
#     for j in range(3):
#         new_mesh1.vectors[i][j] = points_out[f[j],:]

# aux = new_mesh1.vectors

# # Write the mesh to file "pelvis_new.stl"
# new_mesh1.save(ruta + 'Python/femur_new_simplify.stl')


# aca aranca el codigo:
femurTri = load_mesh(ruta + 'Python/femur_new_simplify.stl')
# # femurTri = load_mesh(ruta + 'Python/Femur_predicted.stl')
# femur_name = 'femur_r'
# BCS = {}
# JCS = {}
# BL = {}
AuxCSInfo = {}
BCS['femur_r'], JCS['femur_r'], BL['femur_r'], _, AuxCSInfo['femur_r'] = GIBOC_femur(femurTri, 'r', 'cylinder', 0, 0)

#%% prueba de funcion femur_guess_CS

# # reduce number of triangles

# tri_geom = mesh.Mesh.from_file(ruta + 'bone_datasets/TLEM2/stl/tibia_r.stl')

# Faces = tri_geom.vectors
# P = Faces.reshape(-1, 3)
# Vertex = np.zeros(np.shape(tri_geom.v0), dtype=np.int64)

# _, idx = np.unique(P, axis=0, return_index=True)
# Points = P[np.sort(idx)]

# for pos, elem in enumerate(tri_geom.v0):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,0] = tmp1[0]
#     else:
#         Vertex[pos,0] = tmp[0]
        
# for pos, elem in enumerate(tri_geom.v1):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,1] = tmp1[0]
#     else:
#         Vertex[pos,1] = tmp[0]

# for pos, elem in enumerate(tri_geom.v2):
#     tmp = np.where(Points == elem)[0]
#     if len(tmp) > 3:
#         l0 = []
#         l0 = list(tmp)
#         tmp1 = [x for x in l0 if l0.count(x) > 1]
#         Vertex[pos,2] = tmp1[0]
#     else:
#         Vertex[pos,2] = tmp[0]


# points_out, faces_out = fast_simplification.simplify(Points, Vertex, 0.9) # 30%

# new_mesh1 = mesh.Mesh(np.zeros(faces_out.shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(faces_out):
#     for j in range(3):
#         new_mesh1.vectors[i][j] = points_out[f[j],:]

# aux = new_mesh1.vectors

# # Write the mesh to file "tibia_new_simplify.stl"
# new_mesh1.save(ruta + 'Python/tibia_new_simplify.stl')


# aca aranca el codigo:
tibiaTri = load_mesh(ruta + 'Python/tibia_new_simplify.stl')

BCS['tibia_r'], JCS['tibia_r'], BL['tibia_r'], AuxCSInfo['tibia_r'] = Kai2014_tibia(tibiaTri, 'r', 0)


#%% ----------------------------------------------------------------------

# # def createOpenSimModelJoints(osimModel, JCS, joint_defs = 'auto2020', jointParamFile = 'getJointParams'):
#     # -------------------------------------------------------------------------
#     # Create the lower limb joints based on assigned joint coordinate systems
#     # stored in a structure and adds them to an existing OpenSim model.
#     # 
#     # Inputs:
#     # osimModel - an OpenSim model of the lower limb to which we want to add
#     # the lower limb joints.
#     # 
#     # JCS - a dictionary created using the function createLowerLimbJoints(). 
#     # This structure includes as fields the elements to generate a 
#     # CustomJoint using the createCustomJointFromStruct function. See these 
#     # functions for details.
#     # 
#     # joint_defs - optional input specifying the joint definitions used for
#     # arranging and finalizing the partial reference systems obtained
#     # from morphological analysis of the bones. Valid values:
#     # - 'Modenese2018': define the same reference systems described in 
#     # Modenese et al. J Biomech (2018).
#     # - 'auto2020': use the reference systems from morphological analysis as 
#     # much as possible. See Modenese and Renault, JBiomech 2020 for a comparison.
#     # - any definition you want to add. You just need to write a function and 
#     # include it in the "switch" structure where joints are defined. Your 
#     # implementation will be check for completeness by the 
#     # verifyJointStructCompleteness.m function.
#     # 
#     # jointParamFile - optional input specifying the name of a function 
#     # including the parameters of the joints to build. Default value is:
#     # - 'getJointParams.py': same joint parameters as the gait2392
#     # standard model.
#     # - 'getJointParams3DoFKnee.py': file available from the advanced examples, 
#     # shows how to create a 3 dof knee joint.
#     # - any other joint parameters you want to implement. Be careful because 
#     # joint definitions and input bone will have to match:
#     # for example you cannot create a 2 dof ankle joint and
#     # exclude a subtalar joint if you are providing a talus and
#     # calcn segments, as otherwise they would not have any joint.
#     # 
#     # Outputs:
#     # none - the joints are added to the input OpenSim model.
#     # -------------------------------------------------------------------------

# joint_defs = 'auto2020'
# jointParamFile = 'getJointParams'
# # ---------------------------------------
# # ---------------------------------------

# # printout
# print('---------------------')
# print('   CREATING JOINTS   ')
# print('---------------------')

# # add ground body to JCS together with standard ground_pelvis joint.
# # if model is partial, it will be modified.
# JCS['ground'] = {}
# JCS['ground']['ground_pelvis'] = {'parentName': 'ground', \
#                                   'parent_location': np.zeros((3,1)), \
#                                   'parent_orientation': np.zeros((1,3))}

# # based on JCS make a list of bodies and joints
# joint_list = compileListOfJointsInJCSStruct(JCS)

# # TRANSFORM THE JCS FROM MORPHOLOGYCAL ANALYSIS IN JOINT DEFINITION
# # complete the joints parameters
# print('Checking parameters from morphological analysis:')

# # useful list
# fields_v = ['parent_location','parent_orientation','child_location', 'child_orientation']

# if jointParamFile != 'getJointParams':
#     jointParamFuncName = jointParamFile
# else:
#     print('WARNING: Specified function ' + jointParamFile + 'for joint parameters was not found. Using default "getJointParams.py"')
#     jointParamFuncName = 'getJointParams'

# jointStruct = {}
# for cur_joint_name in joint_list:
#     # getting joint parameters using the desired joint param function
#     if jointParamFuncName == 'getJointParams':
#         jointStructTemp = getJointParams(cur_joint_name)
#     elif jointParamFuncName == 'getJointParams3DoFKnee':
#         jointStructTemp = getJointParams3DoFKnee(cur_joint_name)
    
#     # STEP1: check if parent and child body are available
#     parent_name = jointStructTemp['parentName']
#     child_name  = jointStructTemp['childName']

#     # the assumption is that if, given a joint from the analysis, parent is
#     # missing, that's because the model is partial proximally and will be
#     # connected to ground. If child is missing, instead, the model if
#     # partial distally and the chain will be interrupted there.
#     if parent_name not in JCS:
#         if child_name in JCS:
#             print('Partial model detected proximally:')
            
#             # get appropriate parameters for the new joint
#             jointStructTemp = getJointParams('free_to_ground', child_name)
            
#             # adjusting joint parameters
#             old_joint_name = cur_joint_name
#             new_cur_joint_name = jointStructTemp['jointName']
#             parent_name = jointStructTemp['parentName']
#             print('   * Connecting ' + child_name + ' to ground with ' + new_cur_joint_name + ' free joint.')
            
#             # defines the new joints for parent/child location and orientation
#             JCS['ground'][new_cur_joint_name] = JCS['ground']['ground_pelvis']
#             JCS['child_name'][new_cur_joint_name] = JCS['child_name'][old_joint_name]
#         else:
#             new_cur_joint_name = cur_joint_name
#             print('ERROR: Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
#             # loggin.error('Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
#     else:
#         new_cur_joint_name = cur_joint_name
    
#     # if there is a parent but not a child body then the model is partial
#     # distally, i.e. it is missing some distal body/bodies.
#     if child_name not in JCS:
#         if parent_name in JCS:
#             print('Partial model detected distally...')
#             print('* Deleting incomplete joint "' + new_cur_joint_name + '"')
#             continue
#         else:
#             print('ERROR: Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
#             # loggin.error('Incorrect definition of joint ' + jointStructTemp['jointName'] + ': missing both parent and child bones from analysis.')
    
#     # display joint details
#     print('* ' + new_cur_joint_name)
#     print('   - parent: ' + parent_name)
#     print('   - child: ' + child_name)
    
#     # create an appropriate jointStructTemp from the info available in JCS
#     # body_list = list(JCS.keys())
#     for cur_body_name in JCS:
#         if new_cur_joint_name in JCS[cur_body_name]:
#             joint_info = JCS[cur_body_name][new_cur_joint_name]
#             for key in fields_v:
#                 if key in joint_info:
#                     jointStructTemp[key] = joint_info[key]
#         else:
#             continue
    
#     # store the resulting parameters for each joint to the final dictionary
#     jointStruct[new_cur_joint_name] = jointStructTemp
                
# # JOINT DEFINITIONS
# print('Applying joint definitions: ' + joint_defs)

# if joint_defs == 'auto2020':
#     jointStruct = jointDefinitions_auto2020(JCS, jointStruct)
# elif joint_defs == 'Modenese2018':
#     # joint definitions of Modenese et al.
#     jointStruct = jointDefinitions_Modenese2018(JCS, jointStruct)
# else:
#     print('createOpenSimModelJoints.py You need to define joint definitions')
#     # loggin.error('createOpenSimModelJoints.py You need to define joint definitions')

# # completeJoints(jointStruct)
# jointStruct = assembleJointStruct(jointStruct)

# # check that all joints are completed
# verifyJointStructCompleteness(jointStruct)

# # after the verification joints can be added
# print('Adding joints to model:')

# for cur_joint_name in jointStruct:
#     # create the joint
#     # _= createCustomJointFromStruct(osimModel, jointStruct[cur_joint_name])
#     # display what has been created
#     print('   * ' + cur_joint_name)

# print('Done.')

#%% ----------------------------------------------------------------------

# def assignMassPropsToSegments(osimModel, JCS = {}, subj_mass = 0, side_raw = ''):
    # -------------------------------------------------------------------------
    # Assign mass properties to the segments of an OpenSim model created 
    # automatically. Mass and inertia are scaled from the values used in the 
    # gait2392 model.
    # NOTE: this function will be rewritten (prototype).
    # 
    # Inputs:
    # osimModel - an OpenSim model generated automatically for which the mass
    # properties needs to be personalised.
    # 
    # JCS - a dictionary including the joint coordinate system computed from
    # the bone geometries. Required for computing the segment lengths and
    # identifying the COM positions.
    # 
    # subj_mass - the total mass of the individual of which we are building a
    # model, in Kg. Required for scaling the mass properties of the
    # generic gati2392 model.
    # 
    # side_raw - generic string identifying a body side. 'right', 'r', 'left'
    # and 'l' are accepted inputs, both lower and upper cases.
    # 
    # Outputs:
    # osimModel - the OpenSim model with the personalised mass properties.
    # -------------------------------------------------------------------------

# subj_mass = 0
# side_raw = ''

# # ------------------------
# # ------------------------

# if side_raw == '':
#     side = inferBodySideFromAnatomicStruct(JCS)
# else:
#     # get sign correspondent to body side
#     _, side = bodySide2Sign(side_raw)

# femur_name = 'femur_' + side
# tibia_name = 'tibia_' + side
# talus_name = 'talus_' + side
# calcn_name = 'calcn_' + side
# hip_name = 'hip_' + side
# knee_name = 'knee_' + side
# ankle_name = 'ankle_' + side
# toes_name = 'mtp_' + side

# print('------------------------')
# print('  UPDATING MASS PROPS   ')
# print('------------------------')

# # compute lengths of segments from the bones and COM positions using 
# # coefficients from Winter 2015 (book)

# # Keep in mind that all Origin fields have [3x1] dimensions
# print('Updating centre of mass position (Winter 2015)...')
# if femur_name in JCS:
#     # compute thigh length
#     thigh_axis = JCS[femur_name][hip_name]['Origin'] - JCS[femur_name][knee_name]['Origin']
#     thigh_L = np.linalg.norm(thigh_axis)
#     thigh_COM = thigh_L*0.567 * (thigh_axis/thigh_L) + JCS[femur_name][knee_name]['Origin']
#     # assign  thigh COM
#     # osimModel.getBodySet().get(femur_name).setMassCenter(opensim.ArrayDouble.createVec3(thigh_COM/1000))
    
#     # shank
#     if talus_name in JCS:
#         # compute thigh length
#         shank_axis = JCS[talus_name][knee_name]['Origin'] - JCS[talus_name][ankle_name]['Origin']
#         shank_L = np.linalg.norm(shank_axis)
#         shank_COM = shank_L*0.567 * (shank_axis/shank_L) + JCS[talus_name][ankle_name]['Origin']
#         # assign  thigh COM
#         # osimModel.getBodySet().get(tibia_name).setMassCenter(opensim.ArrayDouble.createVec3(shank_COM/1000))
        
#         # foot
#         if calcn_name in JCS:
#             # compute thigh length
#             foot_axis = JCS[talus_name][knee_name]['Origin'] - JCS[calcn_name][toes_name]['Origin']
#             foot_L = np.linalg.norm(foot_axis)
#             calcn_COM = shank_L*0.5 * (foot_axis/foot_L) + JCS[calcn_name][toes_name]['Origin']
#             # assign  thigh COM
#             # osimModel.getBodySet().get(calcn_name).setMassCenter(opensim.ArrayDouble.createVec3(calcn_COM/1000))
            
# # -----------------------------------------------------------------------------
# # map gait2392 properties to the model segments as an initial value
# print('Mapping segment masses and inertias from gait2392 model.')
# # osimModel = mapGait2392MassPropToModel(osimModel)

# # opensim model total mass (consistent in gait2392 and Rajagopal)
# MP = gait2392MassProps('full_body')
# gait2392_tot_mass = MP['mass']

# # calculate mass ratio of subject mass and gait2392 mass
# coeff = subj_mass/gait2392_tot_mass

# # scale gait2392 mass properties to the individual subject
# print('Scaling inertial properties to assigned body weight...')
# # scaleMassProps(osimModel, coeff)

# print('Done.')

# %% --------------------------------------------------------------------------

# # -----------------------------------------------------------------------------
# # def addBoneLandmarksAsMarkers(osimModel, BLStruct, in_mm = 1):
#     # -------------------------------------------------------------------------
#     # Add the bone landmarks listed in the input structure as Markers in the 
#     # OpenSim model.
#     # 
#     # Inputs:
#     # osimModel - an OpenSim model (object) to which to add the bony
#     # landmarks as markers.
#     # 
#     # BLStruct - a Dictionary with two layers. The external layer has
#     # fields named as the bones, the internal layer as fields named as
#     # the bone landmarks to add. The value of the latter fields is a
#     # [1x3] vector of the coordinate of the bone landmark. For example:
#     # BLStruct['femur_r']['RFE'] = [xp, yp, zp].
#     # 
#     # in_mm - if all computations are performed in mm or m. Valid values: 1
#     # or 0.
#     # 
#     # Outputs:
#     # none - the OpenSim model in the scope of the calling function will
#     # include the specified markers.
#     # 
#     # -------------------------------------------------------------------------

# in_mm = 1
# BLStruct = BL.copy()
# # ------------------
# # ------------------

# if in_mm == 1:
#     dim_fact = 0.001
# else:
#     dim_fact = 1

# print('------------------------')
# print('     ADDING MARKERS     ')
# print('------------------------')
# print('Attaching bony landmarks to model bodies:')

# # loop through the bodies specified in BLStruct
# for cur_body_name in BLStruct:
#     # body name
#     print('  ' + cur_body_name + ':')
    
#     # check that cur_body_name actually corresponds to a body
#     if osimModel.getBodySet().getIndex(cur_body_name) < 0:
#         # loggin.warning('Markers assigned to body ' + cur_body_name + ' cannot be added to the model. Body is not in BodySet.')
#         print('Markers assigned to body ' + cur_body_name + ' cannot be added to the model. Body is not in BodySet.')
#         continue
    
#     # loop through the markers
#     cur_body_markers = list(BLStruct[cur_body_name].keys())
#     # skip markers if the structure is empty, otherwise process it
#     if cur_body_markers == []:
#         print('    NO LANDMARKS AVAILABLE')
#         continue
#     else:
#         # the actual markers are fields of the cur_body_markers variable
#         for cur_marker_name in cur_body_markers:
#             # get body
#             cur_phys_frame = osimModel.getBodySet.get(cur_body_name)
#             Loc = cur_body_markers[cur_marker_name]*dim_fact
#             marker = opensim.Marker(cur_marker_name, \
#                                     cur_phys_frame,\
#                                     opensim.Vec3(Loc[0], Loc[1], Loc[2]))
            
#             # add current marker to model
#             osimModel.addMarker(marker)
            
#             # clear coordinates as precaution
#             del Loc
#             print('    * ' + cur_marker_name)
            
# print('Done.')

# # return 0












#%% PLOTS ....................

# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.plot_trisurf(tibiaTri['Points'][:,0], tibiaTri['Points'][:,1], tibiaTri['Points'][:,2], triangles = tibiaTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'blue')
# # # # # ax.plot_trisurf(ProxFemTri['Points'][:,0], ProxFemTri['Points'][:,1], ProxFemTri['Points'][:,2], triangles = ProxFemTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'gray')
# # # # ax.plot_trisurf(DistFemTri['Points'][:,0], DistFemTri['Points'][:,1], DistFemTri['Points'][:,2], triangles = DistFemTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'blue')
# ax.set_box_aspect([1,3,1])
# # # ax.plot_trisurf(Condyle['Points'][:,0], Condyle['Points'][:,1], Condyle['Points'][:,2], triangles = Condyle['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.3, shade=False, color = 'blue')
# # # ax.plot_trisurf(Condyle_edges['Points'][:,0], Condyle_edges['Points'][:,1], Condyle_edges['Points'][:,2], triangles = Condyle_edges['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'red')
# # # ax.plot_trisurf(Condyle_end['Points'][:,0], Condyle_end['Points'][:,1], Condyle_end['Points'][:,2], triangles = Condyle_end['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'green')
# # ax.plot_trisurf(EpiFemTri['Points'][:,0], EpiFemTri['Points'][:,1], EpiFemTri['Points'][:,2], triangles = EpiFemTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'red')
# # ax.plot_trisurf(fullCondyle_Lat_Tri['Points'][:,0], fullCondyle_Lat_Tri['Points'][:,1], fullCondyle_Lat_Tri['Points'][:,2], triangles = fullCondyle_Lat_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'green')
# # ax.plot_trisurf(fullCondyle_Med_Tri['Points'][:,0], fullCondyle_Med_Tri['Points'][:,1], fullCondyle_Med_Tri['Points'][:,2], triangles = fullCondyle_Med_Tri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'blue')
# # # # ax.plot_trisurf(KConvHull['Points'][:,0], KConvHull['Points'][:,1], KConvHull['Points'][:,2], triangles = KConvHull['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.2, shade=False, color = 'green')

    
    


# # # ax.plot_trisurf(Patch_MM_FH['Points'][:,0], Patch_MM_FH['Points'][:,1], Patch_MM_FH['Points'][:,2], triangles = Patch_MM_FH['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'blue')
# # # ax.plot_trisurf(Face_MM_FH['Points'][:,0], Face_MM_FH['Points'][:,1], Face_MM_FH['Points'][:,2], triangles = Face_MM_FH['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=1, shade=False, color = 'red')

# # # ax.plot_trisurf(FemHead0['Points'][:,0], FemHead0['Points'][:,1], FemHead0['Points'][:,2], triangles = FemHead0['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.7, shade=False, color = 'cyan')

# # ax.plot_trisurf(FemHead['Points'][:,0], FemHead['Points'][:,1], FemHead['Points'][:,2], triangles = FemHead['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'green')
# # # ax.plot_trisurf(TRout['Points'][:,0], TRout['Points'][:,1], TRout['Points'][:,2], triangles = TRout['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'green')

# # # Plot sphere
# # # Create a sphere
# # phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
# # x = 5*np.sin(phi)*np.cos(theta)
# # y = 5*np.sin(phi)*np.sin(theta)
# # z = 5*np.cos(phi)

# # ax.plot_surface(x + CSs['CenterFH'][0], y + CSs['CenterFH'][1], z + CSs['CenterFH'][2], color = 'red')
# # ax.plot_surface(x + CenterFH[0,0], y + CenterFH[0,1], z + CenterFH[0,2], color = 'black')
# # ax.plot_surface(x + CSs['CenterFH_Renault'][0], y + CSs['CenterFH_Renault'][1], z + CSs['CenterFH_Renault'][2], color = 'green')


# # # Plot sphere
# # # Create a sphere
# # phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
# # x = CSs['RadiusFH_Renault']*np.sin(phi)*np.cos(theta)
# # y = CSs['RadiusFH_Renault']*np.sin(phi)*np.sin(theta)
# # z = CSs['RadiusFH_Renault']*np.cos(phi)

# # ax.plot_surface(x + CSs['CenterFH_Renault'][0], y + CSs['CenterFH_Renault'][1], z + CSs['CenterFH_Renault'][2], color = 'blue', alpha=0.4)

# # for p in P0:
# #     ax.scatter(p[0], p[1], p[2], color = "red")
# # for p in P1:
# #     ax.scatter(p[0], p[1], p[2], color = "blue")

# # # for p1 in Patch['1']:
# # #     p = Tr['Points'][p1]
# # #     ax.scatter(p[0], p[1], p[2], color = "red")
# # # for p1 in Patch['2']:
# # #     p = Tr['Points'][p1]
# # #     ax.scatter(p[0], p[1], p[2], color = "orange")
# # # for p1 in Patch['3']:
# # #     p = Tr['Points'][p1]
# # #     ax.scatter(p[0], p[1], p[2], color = "green")
# # # for p1 in Patch['4']:
# # #     p = Tr['Points'][p1]
# # #     ax.scatter(p[0], p[1], p[2], color = "cyan")
# # # for p1 in Patch['5']:
# # #     p = Tr['Points'][p1]
# # #     ax.scatter(p[0], p[1], p[2], color = "magenta")

# for key in Curves.keys():
    
#     plt.plot(Curves[key]['Pts'][:,0], Curves[key]['Pts'][:,1], Curves[key]['Pts'][:,2], linewidth=4)
    
# # ax.set_box_aspect([1,1,1])
# plt.show()


