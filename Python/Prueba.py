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
from scipy.spatial import ConvexHull
from pykdtree.kdtree import KDTree
from pathlib import Path
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import path as mpl_path


from Public_functions import load_mesh, freeBoundary, PolyArea

from algorithms import pelvis_guess_CS, STAPLE_pelvis, femur_guess_CS, GIBOC_femur_fitSphere2FemHead, \
    Kai2014_femur_fitSphere2FemHead

from GIBOC_core import plotDot, TriInertiaPpties, TriReduceMesh, TriFillPlanarHoles,\
    TriDilateMesh, cutLongBoneMesh, computeTriCoeffMorpho, TriUnite, sphere_fit, \
    TriErodeMesh, TriKeepLargestPatch, TriOpenMesh, TriPlanIntersect, quickPlotRefSystem

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

# pelvisTri = load_mesh(ruta + 'bone_datasets/TLEM2/stl/pelvis.stl')
# pelvisTri = load_mesh(ruta + 'Python/pelvis_new_simplify.stl')
# pelvisTri = load_mesh(ruta + 'Python/pelvis_new.stl')

# RotPseudoISB2Glob, LargestTriangle, BL = pelvis_guess_CS(pelvisTri, 0)

# STAPLE_pelvis(pelvisTri)


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
# femurTri = load_mesh(ruta + 'Python/Femur_predicted.stl')

debug_prints = 1

# # Z0 = femur_guess_CS(Femur, 1)
# # # ---------
# # fig = plt.figure()
# # ax = fig.add_subplot(projection = '3d')
# # ax.plot_trisurf(TrLB['Points'][:,0], TrLB['Points'][:,1], TrLB['Points'][:,2], triangles = TrLB['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'red')
# # ax.set_box_aspect([1,1,1])
# # plt.show()
# # # -------------

# U_0 = np.reshape(np.array([0, 0, 1]),(3, 1))
# L_ratio = 0.33

# # Convert tiangulation dict to mesh object --------
# tmp_Femur = mesh.Mesh(np.zeros(Femur['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
# for i, f in enumerate(Femur['ConnectivityList']):
#     for j in range(3):
#         tmp_Femur.vectors[i][j] = Femur['Points'][f[j],:]
# # update normals
# tmp_Femur.update_normals()
# # ------------------------------------------------

# V_all, _, _, _, _ = TriInertiaPpties(Femur)

# # Initial estimate of the Distal-to-Proximal (DP) axis Z0
# Z0 = V_all[0]
# Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector

# # Reorient Z0 according to U_0
# Z0 *= np.sign(np.dot(U_0.T, Z0))

# # Get the central 60% of the bone -> The femur diaphysis
# LengthBone = np.max(np.dot(Femur['Points'], Z0)) - np.min(np.dot(Femur['Points'], Z0))


# # create the proximal bone part
# Zprox = np.max(np.dot(Femur['Points'], Z0)) - L_ratio*LengthBone
# ElmtsProx = np.where(np.dot(tmp_Femur.centroids, Z0) > Zprox)[0]
# ProxFem = TriReduceMesh(Femur, ElmtsProx)
# ProxFem = TriFillPlanarHoles(ProxFem)

# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.plot_trisurf(ProxFem['Points'][:,0], ProxFem['Points'][:,1], ProxFem['Points'][:,2], triangles = ProxFem['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'gray')
# ax.set_box_aspect([1,1,1])
# plt.show()
# # # # -------------

# femurTri = Femur.copy()

U_DistToProx = femur_guess_CS(femurTri, 0)
ProxFemTri, DistFemTri = cutLongBoneMesh(femurTri, U_DistToProx)
# DistFemTri, ProxFemTri = cutLongBoneMesh(femurTri, U_DistToProx)

# Compute the coefficient for morphology operations
CoeffMorpho = computeTriCoeffMorpho(femurTri)

# Get inertial principal vectors V_all of the femur geometry & volum center
V_all, CenterVol, InertiaMatrix, _, _ = TriInertiaPpties(femurTri)

# -------------------------------------
# Initial Coordinate system (from inertial axes and femoral head):
# * Z0: points upwards (inertial axis) 
# * Y0: points medio-lat 
# -------------------------------------
# coordinate system structure to store coordinate system's info
AuxCSInfo = {}
AuxCSInfo['CenterVol'] = CenterVol
AuxCSInfo['V_all'] = V_all

# Check that the distal femur is 'below' the proximal femur or invert Z0
Z0 = V_all[0]
Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector
Z0 *= np.sign(np.dot((np.mean(ProxFemTri['Points'], axis=0) - np.mean(DistFemTri['Points'], axis=0)),Z0))
AuxCSInfo['Z0'] = Z0

# # Find Femoral Head Center
# try:
#     # sometimes Renault2018 fails for sparse meshes 
#     # FemHeadAS is the articular surface of the hip
#     AuxCSInfo, FemHeadTri = GIBOC_femur_fitSphere2FemHead(ProxFemTri, AuxCSInfo, CoeffMorpho, 0)
# except:
#     # use Kai if GIBOC approach fails
#     # logging.warning('Renault2018 fitting has failed. Using Kai femoral head fitting. \n')
#     AuxCSInfo, _ = Kai2014_femur_fitSphere2FemHead(ProxFemTri, AuxCSInfo, 0)
#     AuxCSInfo['CenterFH_Renault'] = AuxCSInfo['CenterFH_Kai']
#     AuxCSInfo['RadiusFH_Renault'] = AuxCSInfo['RadiusFH_Kai']

# # X0 points backwards
# AuxCSInfo['X0'] = np.cross(AuxCSInfo['Y0'].T, AuxCSInfo['Z0'].T).T

# # Isolates the epiphysis
# # EpiFemTri = GIBOC_isolate_epiphysis(DistFemTri, Z0, 'distal')
debug_plot = 0

cut_offset = 0.5
step = 0.5
min_coord = np.min(np.dot(ProxFemTri['Points'], Z0)) + cut_offset
max_coord = np.max(np.dot(ProxFemTri['Points'], Z0)) - cut_offset
Alt = np.arange(min_coord, max_coord, step)

# Curves = {}
# Areas = {}

# for it, d in enumerate(-Alt):
    
#     Curves[str(it)], Areas[str(it)], _ = TriPlanIntersect(ProxFemTri, Z0, d)
    
# if debug_plot:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     for key in Curves.keys():
#         if Curves[key]:
#             ax.plot(Curves[key]['1']['Pts'][:,0], Curves[key]['1']['Pts'][:,1], Curves[key]['1']['Pts'][:,2], 'k-', linewidth=2)
#     plt.show()

# print('Sliced #' + str(len(Curves)-1) + ' times')
# maxArea = Areas[max(Areas, key=Areas.get)]
# maxAreaInd = int(max(Areas, key=Areas.get))
# maxAlt = Alt[maxAreaInd]

# Check la función TRIPLANINTERSECT
#------------
n= Z0.copy()
d = -Alt[-95].copy()
Tr = ProxFemTri.copy()

if np.linalg.norm(n) == 0 and np.linalg.norm(d) == 0:
    # loggin.error('Not engough input argument for TriPlanIntersect')
    print('Not engough input argument for TriPlanIntersect')

Pts = Tr['Points']
n = preprocessing.normalize(n, axis=0)

# If d is a point on the plane and not the d parameter of the plane equation        
if type(d) is np.ndarray and len(d) > 2:
    Op = d
    row, col = d.shape
    if col == 1:
        d = np.dot(-d,n)
    elif row == 1:
        d = np.dot(-d.T,n)
    else:
        # loggin.error('Third input must be an altitude or a point on plane')
        print('Third input must be an altitude or a point on plane')
else:
    # Get a point on the plane
    n_principal_dir = np.argmax(abs(n))
    Pts1 = Pts[0].copy()
    Pts1 = np.reshape(Pts1,(Pts1.size, 1)) # convert 1d (3,) to 2d (3,1) vector
    Op = Pts1.copy()
    Pts1[n_principal_dir] = 0
    Op[n_principal_dir] = (np.dot(-Pts1.T, n) - d)/n[n_principal_dir]

## Find the intersected elements (triagles)
# Get Points (vertices) list as being over or under the plan

Pts_Over = [1 if np.dot(p.T, n) + d > 0 else 0 for p in Pts]
Pts_Under = [1 if np.dot(p.T, n) + d < 0 else 0 for p in Pts]
Pts_OverUnder = np.array(Pts_Over) - np.array(Pts_Under)

if np.sum(Pts_OverUnder == 0) > 0:
    # loggin.warning('Points were found lying exactly on the intersecting plan, this case might not be correctly handled')
    print('Points were found lying exactly on the intersecting plan, this case might not be correctly handled')

# Get the facets,elements/triangles/ intersecting the plan
Elmts_Intersecting = []
Elmts = Tr['ConnectivityList'] 
Elmts_IntersectScore = np.sum(Pts_OverUnder[Elmts],axis=1)
Elmts_Intersecting =  Elmts[np.abs(Elmts_IntersectScore)<3]

# Check the existence of an interaction
if len(Elmts_Intersecting) == 0:
    TotArea = 0
    InterfaceTri = []
    Curves = {}
    Curves['1'] = {}
    Curves['1']['NodesID'] = []
    Curves['1']['Pts'] = []
    Curves['1']['Area'] = 0
    Curves['1']['Hole'] = 0
    Curves['1']['Text'] = 'No Intersection'
    # loggin.warning('No intersection found between the plane and the triangulation')
    print('No intersection found between the plane and the triangulation')
    # return 0

# Find the Intersecting Edges among the intersected elements
# Get an edge list from intersecting elmts
Nb_InterSectElmts = len(Elmts_Intersecting)
Edges = np.zeros((3*Nb_InterSectElmts, 2))

i = np.array(range(1,Nb_InterSectElmts+1))
Edges[3*i-3] = Elmts_Intersecting[i-1,:2]
Edges[3*i-2] = Elmts_Intersecting[i-1,1:3]
Edges[3*i-1,0] = Elmts_Intersecting[i-1,-1]
Edges[3*i-1,1] = Elmts_Intersecting[i-1,0]
Edges = Edges.astype(np.int64)

# Identify the edges crossing the plane
# They will have an edge status of 0
Edges_Status = np.sum(Pts_OverUnder[Edges],axis=1)

I_Edges_Intersecting = np.where(Edges_Status == 0)[0]
# Find the edge plane intersecting points
# start and end points of each edges

P0 = Pts[Edges[I_Edges_Intersecting,0]]
P1 = Pts[Edges[I_Edges_Intersecting,1]]

# Vector of the edge
u = P1 - P0

# Get vectors from point on plane (Op) to edge ends
v = P0 - Op.T

EdgesLength = np.dot(u,n)
EdgesUnderPlaneLength = np.dot(-v,n)

ratio = EdgesUnderPlaneLength/EdgesLength

# Get Intersectiong Points Coordinates
PtsInter = P0 + u*ratio

# Make sure the shared edges have the same intersection Points
# Build an edge correspondance table

EdgeCorrespondence = np.empty((3*Nb_InterSectElmts, 1))
EdgeCorrespondence[:] = np.nan

for pos, edge in enumerate(Edges):
    
    tmp = np.where((Edges[:,1] == edge[0]) & (Edges[:,0] == edge[1]))
    
    if tmp[0].size == 0:
        continue
    elif tmp[0].size == 1:
        i = tmp[0][0]
        if np.isnan(EdgeCorrespondence[pos]):
            EdgeCorrespondence[pos] = pos
            EdgeCorrespondence[i] = pos
    else:
        print('Intersecting edge appear in 3 triangles, not good')

# Get edge intersection point
# Edge_IntersectionPtsIndex = np.zeros((3*Nb_InterSectElmts, 1))
Edge_IntersectionPtsIndex = np.empty((3*Nb_InterSectElmts, 1))
Edge_IntersectionPtsIndex[:] = np.nan
tmp_edgeInt = np.array(range(len(I_Edges_Intersecting)))
tmp_edgeInt = np.reshape(tmp_edgeInt,(tmp_edgeInt.size, 1)) # convert 1d (#,) to 2d (#,1) vector
Edge_IntersectionPtsIndex[I_Edges_Intersecting] = tmp_edgeInt

# Don't use intersection point duplicate: only one intersection point per edge
ind_interP = EdgeCorrespondence[I_Edges_Intersecting]
ind_interP = list(ind_interP)
I_Edges_Intersecting = list(I_Edges_Intersecting)
t = np.sort(np.where(np.isnan(ind_interP))[0])
t = t[::-1]
for ind in t:
    del ind_interP[ind]
    del I_Edges_Intersecting[ind]

ind_interP = np.array(ind_interP).astype(np.int64)
I_Edges_Intersecting = np.array(I_Edges_Intersecting).astype(np.int64)

# Edge_IntersectionPtsIndex[I_Edges_Intersecting] = Edge_IntersectionPtsIndex[EdgeCorrespondence[I_Edges_Intersecting].astype(np.int64)][:,0]
Edge_IntersectionPtsIndex[I_Edges_Intersecting] = Edge_IntersectionPtsIndex[ind_interP][:,0]

# Get the segments intersecting each triangle
# The segments are: [Intersecting Point 1 ID , Intersecting Point 2 ID]
# Segments = Edge_IntersectionPtsIndex[Edge_IntersectionPtsIndex > 0].astype(np.int64)
Segments = Edge_IntersectionPtsIndex[Edge_IntersectionPtsIndex != np.nan]
Segments = Segments[~np.isnan(Segments)].astype(np.int64)
Segments = list(Segments.reshape((-1,2)))

# Separate the edges to curves structure containing close curves
j = 1
Curves = {}
i = 1
while Segments:
    # Initialise the Curves Structure, if there are multiple curves this
    # will lead to trailing zeros that will be removed afterwards
    Curves[str(i)] = {}
    Curves[str(i)]['NodesID'] = []
    Curves[str(i)]['NodesID'].append(Segments[0][0])
    Curves[str(i)]['NodesID'].append(Segments[0][1])
    
    # Remove the semgents added to Curves[i] from the segments list
    del Segments[0]
    # j += 1
    
    # Find the edge in segments that has a node already in the Curves[i][NodesID]
    # This edge will be the next edge of the current curve because it's
    # connected to the current segment
    # Is, the index of the next edge
    # Js, the index of the node within this edge already present in NodesID
    
    tmp1 = np.where(Segments == Curves[str(i)]['NodesID'][-1])
    if tmp1[0].size == 0:
        break
    else:
        Is = tmp1[0][0]
        Js = tmp1[1][0]
        
    # Nk is the node of the previuously found edge that is not in the
    # current Curves[i][NodesID] list
    # round(Js+2*(0.5-Js)) give 0 if Js = 1 and 1 if Js = 0
    # It gives the other node not yet in NodesID of the identified next edge
    Nk = Segments[Is][int(np.round(Js+2*(0.5-Js)))]
    del Segments[Is]
        
    # Loop until there is no next node
    while Nk:
        Curves[str(i)]['NodesID'].append(Nk)
        if Segments:
            tmp2 = np.where(Segments == Curves[str(i)]['NodesID'][-1])
            if tmp2[0].size == 0:
                break
            else:
                Is = tmp2[0][0]
                Js = tmp2[1][0]
                       
            Nk = Segments[Is][int(np.round(Js+2*(0.5-Js)))]
            del Segments[Is]
        else:
            break
        
    # If there is on next node then we move to the next curve
    i += 1

# Compute the area of the cross section defined by the curve

# Deal with cases where a cross section presents holes
# 
# Get a matrix of curves inclusion -> CurvesInOut :
# If the curve(i) is within the curve(j) then CurvesInOut(i,j) = 1
# else  CurvesInOut(i,j) = 0

for key in Curves.keys():
    Curves[key]['Pts'] = []
    Curves[key]['Pts'] = PtsInter[Curves[key]['NodesID']]
    
    # Replace the close curve in coordinate system where X, Y or Z is 0
    _, V = np.linalg.eig(np.cov(Curves[key]['Pts'].T))
    
    CloseCurveinRplanar1 = np.dot(V.T, Curves[key]['Pts'].T)
    
    # Get the area of the section defined by the curve 'key'.
    # /!\ the curve.Area value Do not account for the area of potential 
    # holes in the section described by curve 'key'.
    Curves[key]['Area'] = PolyArea(CloseCurveinRplanar1[0,:],CloseCurveinRplanar1[2,:])
    
    CurvesInOut = np.zeros((len(Curves),len(Curves)))
    
    for key1 in Curves.keys():
        if key1 != key:
            Curves[key1]['Pts'] = []
            Curves[key1]['Pts'] = PtsInter[Curves[key1]['NodesID']]
            
            # Replace the close curve in coordinate system where X, Y or Z is 0
            _, V = np.linalg.eig(np.cov(Curves[key1]['Pts'].T))
            
            CloseCurveinRplanar2 = np.dot(V.T, Curves[key1]['Pts'].T)
            
            # Check if the Curves[key] is within the Curves[key1]
            path1 = CloseCurveinRplanar1[0:3:2,:].T
            path2 = CloseCurveinRplanar2[0:3:2,:].T
            
            p = mpl_path.Path(path1)
            if any(p.contains_points(path2)):
                
                CurvesInOut[int(key)-1, int(key1)-1] = 1












#%% PLOTS ....................

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
# # # ax.plot_trisurf(femurTri['Points'][:,0], femurTri['Points'][:,1], femurTri['Points'][:,2], triangles = femurTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'blue')
ax.plot_trisurf(ProxFemTri['Points'][:,0], ProxFemTri['Points'][:,1], ProxFemTri['Points'][:,2], triangles = ProxFemTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'gray')
# # ax.plot_trisurf(DistFemTri['Points'][:,0], DistFemTri['Points'][:,1], DistFemTri['Points'][:,2], triangles = DistFemTri['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'blue')

# # ax.plot_trisurf(Patch_Top_FH['Points'][:,0], Patch_Top_FH['Points'][:,1], Patch_Top_FH['Points'][:,2], triangles = Patch_Top_FH['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'green')
# # ax.plot_trisurf(Face_Top_FH['Points'][:,0], Face_Top_FH['Points'][:,1], Face_Top_FH['Points'][:,2], triangles = Face_Top_FH['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=1, shade=False, color = 'red')

# # ax.plot_trisurf(Patch_MM_FH['Points'][:,0], Patch_MM_FH['Points'][:,1], Patch_MM_FH['Points'][:,2], triangles = Patch_MM_FH['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'blue')
# # ax.plot_trisurf(Face_MM_FH['Points'][:,0], Face_MM_FH['Points'][:,1], Face_MM_FH['Points'][:,2], triangles = Face_MM_FH['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=1, shade=False, color = 'red')

# # ax.plot_trisurf(FemHead0['Points'][:,0], FemHead0['Points'][:,1], FemHead0['Points'][:,2], triangles = FemHead0['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.7, shade=False, color = 'cyan')

# ax.plot_trisurf(FemHead['Points'][:,0], FemHead['Points'][:,1], FemHead['Points'][:,2], triangles = FemHead['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'green')
# # ax.plot_trisurf(TRout['Points'][:,0], TRout['Points'][:,1], TRout['Points'][:,2], triangles = TRout['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.9, shade=False, color = 'green')

# # Plot sphere
# # Create a sphere
# phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
# x = 5*np.sin(phi)*np.cos(theta)
# y = 5*np.sin(phi)*np.sin(theta)
# z = 5*np.cos(phi)

# ax.plot_surface(x + CSs['CenterFH'][0], y + CSs['CenterFH'][1], z + CSs['CenterFH'][2], color = 'red')
# ax.plot_surface(x + CenterFH[0,0], y + CenterFH[0,1], z + CenterFH[0,2], color = 'black')
# ax.plot_surface(x + CSs['CenterFH_Renault'][0], y + CSs['CenterFH_Renault'][1], z + CSs['CenterFH_Renault'][2], color = 'green')


# # Plot sphere
# # Create a sphere
# phi, theta = np.mgrid[0.0:np.pi:50j, 0.0:2.0*np.pi:50j]
# x = CSs['RadiusFH_Renault']*np.sin(phi)*np.cos(theta)
# y = CSs['RadiusFH_Renault']*np.sin(phi)*np.sin(theta)
# z = CSs['RadiusFH_Renault']*np.cos(phi)

# ax.plot_surface(x + CSs['CenterFH_Renault'][0], y + CSs['CenterFH_Renault'][1], z + CSs['CenterFH_Renault'][2], color = 'blue', alpha=0.4)

# for p in P0:
#     ax.scatter(p[0], p[1], p[2], color = "red")
# for p in P1:
#     ax.scatter(p[0], p[1], p[2], color = "blue")

# # for p1 in Patch['1']:
# #     p = Tr['Points'][p1]
# #     ax.scatter(p[0], p[1], p[2], color = "red")
# # for p1 in Patch['2']:
# #     p = Tr['Points'][p1]
# #     ax.scatter(p[0], p[1], p[2], color = "orange")
# # for p1 in Patch['3']:
# #     p = Tr['Points'][p1]
# #     ax.scatter(p[0], p[1], p[2], color = "green")
# # for p1 in Patch['4']:
# #     p = Tr['Points'][p1]
# #     ax.scatter(p[0], p[1], p[2], color = "cyan")
# # for p1 in Patch['5']:
# #     p = Tr['Points'][p1]
# #     ax.scatter(p[0], p[1], p[2], color = "magenta")

for key in Curves.keys():
    
    plt.plot(Curves[key]['Pts'][:,0], Curves[key]['Pts'][:,1], Curves[key]['Pts'][:,2], linewidth=4)
    
# ax.set_box_aspect([1,1,1])
plt.show()


