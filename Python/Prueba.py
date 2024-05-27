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


from Public_functions import load_mesh, freeBoundary

from algorithms import pelvis_guess_CS, STAPLE_pelvis

from GIBOC_core import plotDot, TriInertiaPpties, TriReduceMesh

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

tri_geom = mesh.Mesh.from_file(ruta + 'bone_datasets/TLEM2/stl/pelvis.stl')

Faces = tri_geom.vectors
P = Faces.reshape(-1, 3)
Vertex = np.zeros(np.shape(tri_geom.v0), dtype=np.int64)
# Vertex1 = np.zeros(np.shape(tri_geom.v0), dtype=np.int64)

_, idx = np.unique(P, axis=0, return_index=True)
Points = P[np.sort(idx)]

# Vertex1[:,0] = [np.where(Points == elem)[0][0] for elem in tri_geom.v0]
# Vertex1[:,1] = [np.where(Points == elem)[0][0] for elem in tri_geom.v1]
# Vertex1[:,2] = [np.where(Points == elem)[0][0] for elem in tri_geom.v2]

for pos, elem in enumerate(tri_geom.v0):
    tmp = np.where(Points == elem)[0]
    if len(tmp) > 3:
        l0 = []
        l0 = list(tmp)
        tmp1 = [x for x in l0 if l0.count(x) > 1]
        Vertex[pos,0] = tmp1[0]
    else:
        Vertex[pos,0] = tmp[0]
        
for pos, elem in enumerate(tri_geom.v1):
    tmp = np.where(Points == elem)[0]
    if len(tmp) > 3:
        l0 = []
        l0 = list(tmp)
        tmp1 = [x for x in l0 if l0.count(x) > 1]
        Vertex[pos,1] = tmp1[0]
    else:
        Vertex[pos,1] = tmp[0]

for pos, elem in enumerate(tri_geom.v2):
    tmp = np.where(Points == elem)[0]
    if len(tmp) > 3:
        l0 = []
        l0 = list(tmp)
        tmp1 = [x for x in l0 if l0.count(x) > 1]
        Vertex[pos,2] = tmp1[0]
    else:
        Vertex[pos,2] = tmp[0]

# tmp_t0 = [np.where(Points == elem) for elem in tri_geom.v0]
# tmp_t1 = [np.where(Points == elem) for elem in tri_geom.v1]
# tmp_t2 = [np.where(Points == elem) for elem in tri_geom.v2]

triangle = {'Points': Points, 'ConnectivityList': Vertex}


# Creating Mesh objects from a list of vertices and faces
# Create the mesh
new_mesh = mesh.Mesh(np.zeros(Vertex.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(Vertex):
    # print(f)
    for j in range(3):
        new_mesh.vectors[i][j] = Points[f[j],:]

aux = new_mesh.vectors

# Write the mesh to file "pelvis_new.stl"
new_mesh.save(ruta + 'Python/pelvis_new.stl')

# # check differences
# tri_geom1 = mesh.Mesh.from_file('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/pelvis_new.stl')
# Faces1 = tri_geom1.vectors
# l0 = []
# l1 = []
# l2 = []
# for i in range(len(Faces)):
#     tmp_norm = np.linalg.norm(Faces[i] - Faces1[i])
#     if tmp_norm > 0:
#         print(i, tmp_norm)
#         l0 = list(tmp_t0[i][0])
#         print('t0', tmp_t0[i], [x for x in l0 if l0.count(x) > 1])
#         l1 = list(tmp_t1[i][0])
#         print('t1', tmp_t1[i], [x for x in l1 if l1.count(x) > 1])
#         l2 = list(tmp_t2[i][0])
#         print('t2', tmp_t2[i], [x for x in l2 if l2.count(x) > 1])


# reduce number of triangles

points_out, faces_out = fast_simplification.simplify(Points, Vertex, 0.7) # 30%

new_mesh1 = mesh.Mesh(np.zeros(faces_out.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces_out):
    for j in range(3):
        new_mesh1.vectors[i][j] = points_out[f[j],:]

aux = new_mesh1.vectors

# Write the mesh to file "pelvis_new.stl"
new_mesh1.save(ruta + 'Python/pelvis_new_simplify.stl')


# triangle = []
# for vert in range(0,len(Faces),3):
#     vert1 = []
#     vert2 = []
#     vert3 = []
    
#     for pos, elem in enumerate(P):
#         if np.array_equal(elem, Faces[vert,0,:]):
#             vert1.append(pos)
#         if np.array_equal(elem, Faces[vert+1,0,:]):
#             vert2.append(pos)
#         if np.array_equal(elem, Faces[vert+2,0,:]):
#             vert3.append(pos)
#     triangle.append([vert1[0], vert2[0], vert3[0]])
    

# print('Triangle 2: \n')
# print(np.where(P == Faces[1,0,:]))
# print(np.where(P == Faces[1,1,:]))
# print(np.where(P == Faces[1,2,:]))

# print('Triangle 3: \n')
# print(np.where(P == Faces[2,0,:]))
# print(np.where(P == Faces[2,1,:]))
# print(np.where(P == Faces[2,2,:]))

# print('Triangle 4: \n')
# print(np.where(P == Faces[3,0,:]))
# print(np.where(P == Faces[3,1,:]))
# print(np.where(P == Faces[3,2,:]))

# print('Triangle 5: \n')
# print(np.where(P == Faces[4,0,:]))
# print(np.where(P == Faces[4,1,:]))
# print(np.where(P == Faces[4,2,:]))

# print('Triangle 6: \n')
# print(np.where(P == Faces[5,0,:]))
# print(np.where(P == Faces[5,1,:]))
# print(np.where(P == Faces[5,2,:]))

# for pos, elem in enumerate(P):
#     pr

# vertex = Faces[0]

# triangle_index = 0
# vertex_indices = Faces[triangle_index]

# vertices_xyz = Points.transpose()


# # import numpy as np

# points = np.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])

# from scipy.spatial import Delaunay

# tri = Delaunay(points)
# # tri = Delaunay(Points)

# import matplotlib.pyplot as plt

# plt.triplot(points[:,0], points[:,1], tri.simplices)

# plt.plot(points[:,0], points[:,1], 'o')

# plt.show()

# print('polyhedron(faces = [')
# #for vert in tri.triangles:
# for vert in tri.simplices:
#     print('[%d,%d,%d],' % (vert[0],vert[1],vert[2]), '], points = [')
# for i in range(x.shape[0]):
#     print('[%f,%f,%f],' % (x[i], y[i], z[i]), ']);')


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')

# # The triangles in parameter space determine which x, y, z points are
# # connected by an edge
# #ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap=plt.cm.Spectral)
# ax.plot_trisurf(aux[:,0], aux[:,1], aux[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)


# plt.show()





# import open3d as o3d

# mesh = o3d.io.read_triangle_mesh('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/bone_datasets/TLEM2/stl/pelvis.stl')
# mesh = mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh], window_name="STL", left=1000, top=200, width=800, height=650, mesh_show_wireframe=True)

# mesh1 = o3d.io.read_triangle_mesh('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_new.stl')
# mesh1 = mesh1.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh1], window_name="STL NEW", left=1000, top=200, width=800, height=650, mesh_show_wireframe=True)

# mesh1 = o3d.io.read_triangle_mesh('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/Python/pelvis_new_simplify.stl')
# mesh1 = mesh1.compute_vertex_normals()
# o3d.visualization.draw_geometries([mesh1], window_name="STL NEW", left=1000, top=200, width=800, height=650, mesh_show_wireframe=True)

# from mpl_toolkits import mplot3d
# from matplotlib import pyplot
# # Create a new plot
# figure = pyplot.figure()
# axes = mplot3d.Axes3D(figure)

# # Load the STL files and add the vectors to the plot
# your_mesh = mesh.Mesh.from_file('/home/emi/Documents/Codigos MATLAB_PYTHON/STAPLE/pelvis.stl')
# axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))

# # Auto scale to the mesh size
# scale = your_mesh.points.flatten()
# # axes.auto_scale_xyz(scale, scale, scale)

# # Show the plot to the screen
# pyplot.show()
#%%
# A = np.array([[1, 2, 10], [3, 4, 20], [9, 6, 15]])

# triangle['Points']

hull = ConvexHull(triangle['Points'])

HullPoints = hull.points[hull.vertices]
HullConect = hull.simplices

# hull object doesn't remove unreferenced vertices
# create a mask to re-index faces for only referenced vertices
vid = np.sort(hull.vertices)
mask = np.zeros(len(hull.points), dtype=np.int64)
mask[vid] = np.arange(len(vid))
# remove unreferenced vertices here
faces = mask[hull.simplices].copy()
# rescale vertices back to original size
vertices = hull.points[vid].copy()



# Hulltriangle = {'Points': HullPoints, 'ConnectivityList': HullConect}
Hulltriangle = {'Points': vertices, 'ConnectivityList': faces}






# Convert tiangulation dict to mesh object
Tr_mesh = mesh.Mesh(np.zeros(Hulltriangle['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(Hulltriangle['ConnectivityList']):
    for j in range(3):
        Tr_mesh.vectors[i][j] = Hulltriangle['Points'][f[j],:]


# aux = sum(TR.incenter.*repmat(Properties.Area,1,3),1)/Properties.TotalArea;

tmp1 = new_mesh1.areas
tmp2 = new_mesh1.centroids

tmp_meanNormal = np.sum(new_mesh1.get_unit_normals()*tmp1/np.sum(tmp1),0)

center = np.sum(tmp2*tmp1/np.sum(tmp2), 0)

# print(np.were(tmp2, np.min(tmp2-center)))

tree = KDTree(triangle['Points'])
# pts = np.array([[0, 0.2, 0.2]])
pts = np.array([center])
dist, idx = tree.query(pts)
print(triangle['Points'][idx])


TMPtriangle = {}
TMPtriangle['Points'] = points_out
TMPtriangle['ConnectivityList'] = faces_out

I = np.argmax(tmp1)
aux1 = TMPtriangle['Points'][TMPtriangle['ConnectivityList'][I]]


eigenvalues, eigenvectors = np.linalg.eig(np.array([[1, 1, 2], [1, 2, 2], [1, 3, 0]]))

aux2 = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])

normalized_arr = preprocessing.normalize(aux2,axis=0)
print(normalized_arr)

aux3 = np.dot(TMPtriangle['Points'],aux2)


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
TrLB = load_mesh(ruta + 'Python/femur_new_simplify.stl')


# # ---------
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.plot_trisurf(TrLB['Points'][:,0], TrLB['Points'][:,1], TrLB['Points'][:,2], triangles = TrLB['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.5, shade=False, color = 'red')
# ax.set_box_aspect([1,1,1])
# plt.show()
# # -------------

U_0 = np.reshape(np.array([0, 0, 1]),(3, 1))
L_ratio = 0.33

# Convert tiangulation dict to mesh object --------
tmp_TrLB = mesh.Mesh(np.zeros(TrLB['ConnectivityList'].shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(TrLB['ConnectivityList']):
    for j in range(3):
        tmp_TrLB.vectors[i][j] = TrLB['Points'][f[j],:]
# update normals
tmp_TrLB.update_normals()
# ------------------------------------------------

V_all, _, _, _, _ = TriInertiaPpties(TrLB)

# Initial estimate of the Distal-to-Proximal (DP) axis Z0
Z0 = V_all[0]
Z0 = np.reshape(Z0,(Z0.size, 1)) # convert 1d (3,) to 2d (3,1) vector

# Reorient Z0 according to U_0
Z0 *= np.sign(np.dot(U_0.T, Z0))

# Fast and dirty way to split the bone
LengthBone = np.max(np.dot(TrLB['Points'], Z0)) - np.min(np.dot(TrLB['Points'], Z0))

# create the proximal bone part
Zprox = np.max(np.dot(TrLB['Points'], Z0)) - L_ratio*LengthBone
ElmtsProx = np.where(np.dot(tmp_TrLB.centroids, Z0) > Zprox)[0]
TrProx = TriReduceMesh(TrLB, ElmtsProx)
# # TrProx = TriFillPlanarHoles( TrProx )

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_trisurf(TrProx['Points'][:,0], TrProx['Points'][:,1], TrProx['Points'][:,2], triangles = TrProx['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.1, shade=False, color = 'blue')
ax.set_box_aspect([1,1,1])
# plt.show()
# -------------

# FreeB = {}
# FreeB['ID'] = []
# FreeB['Coord'] = []

# for IDpoint, Point in enumerate(TrProx['Points']):
#     vertex_triangle = []
#     CloudPoint = []
#     tmp_norm = []
    
#     # identify the triangles whit this vertex
#     vertex_triangle = list(np.where(TrProx['ConnectivityList'] == IDpoint)[0])
    
#     # identify the neighborhood of points (all point from triangles that inlcude Point)
#     for neighbor in vertex_triangle:
        
#         v0 = TrProx['Points'][TrProx['ConnectivityList'][neighbor, 0]]
#         v1 = TrProx['Points'][TrProx['ConnectivityList'][neighbor, 1]]
#         v2 = TrProx['Points'][TrProx['ConnectivityList'][neighbor, 2]]
        
#         if np.linalg.norm(v0 - Point) != 0:
#             CloudPoint.append(v0)
#         if np.linalg.norm(v1 - Point) != 0:
#             CloudPoint.append(v1)
#         if np.linalg.norm(v2 - Point) != 0:
#             CloudPoint.append(v2)
    
#     # for each neighborhood compute the norm with the another neighborhood. 
#     # If this norm is zero in minus of two times, this point is in the bounder
#     for neig in CloudPoint:
#         tmp_norm = [np.linalg.norm(neig - val) for val in CloudPoint]
#         if tmp_norm.count(0) < 2:
#             # duplicate points
#             FreeB['ID'].append(IDpoint)
#             FreeB['Coord'].append(Point)

# # remove duplicate points
# FreeB['ID'] = FreeB['ID'][::2]
# FreeB['Coord'] = FreeB['Coord'][::2]
FreeB = freeBoundary(TrProx)

for point in FreeB['Coord']:
    ax.scatter(point[0], point[1], point[2], color='red')

plt.show()
# # -------------
# Fill planar convex holes in the triangulation
# For now the holes have to be planar
# FOR NOW WORKS WITH ONLY ONE HOLE
# 
# Author: Emiliano P. Ravera (emiliano.ravera@uner.edu.ar)
# -------------------------------
Trout = {}
FreeB = freeBoundary(TrProx)
Tr = TrProx

if not FreeB:
    print('No holes on triangulation.')
    Trout = TrProx
    # return Trout

# Fill the holes
# center of the triangulation with hole
TriCenter = np.mean(Tr['Points'], axis=0)
TriCenter = np.reshape(TriCenter,(TriCenter.size, 1)) # convert d (3,) to 2d (3,1) vector

# center onf the hole
HoleCenter = np.mean(FreeB['Coord'], axis=0)
HoleCenter = np.reshape(HoleCenter,(HoleCenter.size, 1)) # convert d (3,) to 2d (3,1) vector

NewNode = np.max(Tr['ConnectivityList']) + 1

U = preprocessing.normalize(HoleCenter-TriCenter, axis=0)

ConnecL = []
Free_Points = FreeB['Coord']
tmp_Free_Points = FreeB['Coord'][-1:] + FreeB['Coord'][:-1]
for p1, p2 in zip(Free_Points, tmp_Free_Points):
        
    Vctr1 = p1 - HoleCenter.T
    Vctr2 = p2 - HoleCenter.T
        
    normal = preprocessing.normalize(np.cross(Vctr1, Vctr2), axis=1)
    
    ind_p1 = np.where(FreeB['Coord'] == p1)[0][0]
    ind_p2 = np.where(FreeB['Coord'] == p2)[0][0]
    
    # Invert node ordering if the normals are inverted
    if np.dot(normal, U) < 0:
        ConnecL.append(np.array([FreeB['ID'][ind_p1], NewNode, FreeB['ID'][ind_p2]]))
    else:
        ConnecL.append(np.array([FreeB['ID'][ind_p1], FreeB['ID'][ind_p2], NewNode]))
            
tmp_Points = list(Tr['Points'])
tmp_Points.append(HoleCenter[:,0].T)
NewPoints = np.array(tmp_Points)

tmp_ConnectivityList = list(Tr['ConnectivityList'])
# ConnecL = ConnectivityList + ConnecL  
tmp_ConnectivityList += ConnecL 

Trout['Points'] = NewPoints
Trout['ConnectivityList'] = np.array(tmp_ConnectivityList)


ax.plot_trisurf(Trout['Points'][:,0], Trout['Points'][:,1], Trout['Points'][:,2], triangles = Trout['ConnectivityList'], edgecolor=[[0,0,0]], linewidth=1.0, alpha=0.6, shade=False, color = 'green')
ax.set_box_aspect([1,1,1])

