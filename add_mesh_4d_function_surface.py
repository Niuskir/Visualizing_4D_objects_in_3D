import bpy
import sys
import numpy as np
from mathutils import *
from mathutils import Vector
from math import *
from bpy.types import Operator
from bpy.props import (StringProperty,IntProperty,FloatProperty,BoolProperty)
from os import system
import builtins
from collections import OrderedDict
import math
from math import (sqrt,dist,radians,acos,pi,atan2)
from bpy.app.translations import pgettext_data as data_
from bpy_extras import object_utils
cls = lambda: system('cls')
cls() #this function call will clear the System Console

# some shortcuts
PRECISION=1e-3
center = np.array([[0,0,0,0]]) # row vector. ROW VECTORS AAAAWWWW YEAH
zero = np.array([[0,0,0,0]])
realm_point = np.array([0,0,0,0])
x = np.array([[1,0,0,0]])
y = np.array([[0,1,0,0]])
z = np.array([[0,0,1,0]])
w = np.array([[0,0,0,1]])
basis=np.r_[x,y,z,w]

# List of safe functions for eval()
safe_list = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh',
    'degrees', 'e', 'exp', 'fabs', 'floor', 'fmod', 'frexp', 'hypot',
    'ldexp', 'log', 'log10', 'modf', 'pi', 'pow', 'radians',
    'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'gcd']

# Use the list to filter the local namespace
safe_dict = dict((k, globals().get(k, None)) for k in safe_list)
safe_dict['math'] = math
safe_dict['numpy'] = safe_dict['np'] = np
safe_dict['lcm'] = np.lcm
safe_dict['max'] = max
safe_dict['min'] = min

objxyz_object_name="object_4D_3D"

""" Create Geometry Node Group for wireframe view of object """
def add_geometrynodes(group_name,tree_name,material_name,obj):
    try:
        group = bpy.data.node_groups[tree_name]
    except:
        group = bpy.data.node_groups.new(tree_name, 'GeometryNodeTree')   
        gi_Node=group.nodes.new('NodeGroupInput')
        group.outputs.new('NodeSocketGeometry', data_("Geometry"))
        go_Node=group.nodes.new('NodeGroupOutput')
        group.inputs.new('NodeSocketGeometry', data_("Geometry"))
        mtc_Node=group.nodes.new(type="GeometryNodeMeshToCurve")
        ctm_Node=group.nodes.new(type="GeometryNodeCurveToMesh")
        nsm_Node=group.nodes.new(type="GeometryNodeSetMaterial")
        nsm_Node.inputs[2].default_value = bpy.data.materials[material_name]
        cpc_Node=group.nodes.new(type="GeometryNodeCurvePrimitiveCircle")    
        cpc_Node.inputs[4].default_value = 0.004
        
        group.links.new(group.nodes["Group Input"].outputs[0], group.nodes["Mesh to Curve"].inputs[0])
        group.links.new(group.nodes["Mesh to Curve"].outputs[0], group.nodes["Curve to Mesh"].inputs[0])
        group.links.new(group.nodes["Curve Circle"].outputs[0], group.nodes["Curve to Mesh"].inputs[1])
        group.links.new(group.nodes["Curve to Mesh"].outputs[0], group.nodes["Set Material"].inputs[0])
        group.links.new(group.nodes["Set Material"].outputs[0], group.nodes["Group Output"].inputs[0])

        gi_Node.select = False
        mtc_Node.select = False
        ctm_Node.select = False
        nsm_Node.select = False
        cpc_Node.select = False
        go_Node.select = False

        gi_Node.location.x=-400
        mtc_Node.location.x=-200
        ctm_Node.location.x=000
        cpc_Node.location.x=-200
        cpc_Node.location.y=-200
        nsm_Node.location.x=200
        go_Node.location.x=400
    
    modifier = obj.modifiers.new(group_name, "NODES")
    #bpy.ops.node.new_geometry_node_group_assign()
    #node_tree = bpy.data.node_groups[tree_name]
    modifier.node_group = group

def create_material(name,color):
    # create new material
    material = bpy.data.materials.new(name=name)
    # enable creating a material via nodes
    material.use_nodes = True

    # get a reference to the Principled BSDF shader node
    principled_bsdf_node = material.node_tree.nodes["Principled BSDF"]

    # set the base color of the material
    principled_bsdf_node.inputs["Base Color"].default_value = color

    # set the metallic value of the material
    principled_bsdf_node.inputs["Metallic"].default_value = 0.5

    # set the roughness value of the material
    principled_bsdf_node.inputs["Roughness"].default_value = 0.5

    return material

def delete_objects(object_name):
    bpy.ops.object.select_all(action='DESELECT')
    for ob in bpy.context.scene.objects:              
        if ob.name.startswith(object_name):
            #Select the object
            ob.select_set(True)     
    #Delete all objects selected above 
    bpy.ops.object.delete()

""" Main method: takes 4D data previously generated and creates views in 3D realm """    
def generate_4D_to_3D(TRx_from,TRx_to,TRx_fixed,TRy_from,TRy_to,TRy_fixed,TRz_from,TRz_to,
                    TRz_fixed,TRw_from,TRw_to,TRw_fixed,Rxy_from,Rxy_to,Rxy_fixed,Rxz_from,Rxz_to,
                    Rxz_fixed,Rxw_from,Rxw_to,Rxw_fixed,Ryz_from,Ryz_to,Ryz_fixed,Ryw_from,Ryw_to,
                    Ryw_fixed,Rzw_from,Rzw_to,Rzw_fixed,frame_start,frame_end,vertices4D,edges4D,
                    polygons4D,object_origin4D,show_wire,context):

    bpy.context.scene.cursor.location = (0,0,0) #put curser in world center
    bpy.context.scene.cursor.rotation_euler = (0,0,0) #reset curser rotation

    if bpy.context.mode!='OBJECT': # if not in OBJECT mode, set OBJECT mode
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT') 

    scene = bpy.context.scene
    
      # set frame range
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.frame_set(frame_start)
    frame_range=frame_end-frame_start



    if frame_range!=0: # frame range is > 1
        # Translation per frame per dimension
        if TRx_fixed == 0:
            Tx = (TRx_to - TRx_from)/frame_range
        else:
            Mx = TRx_fixed      
        if TRy_fixed == 0:
            Ty = (TRy_to - TRy_from)/frame_range
        else:
            My = TRy_fixed
        if TRz_fixed == 0:
            Tz = (TRz_to - TRz_from)/frame_range
        else:
            Mz = TRz_fixed
        if TRw_fixed == 0:
            Tw = (TRw_to - TRw_from)/frame_range
        else:
            Mw = TRw_fixed 

        #Rotation step per frame for all 6 rotation planes 
        if Rxy_fixed == 0:
            Rxy = (Rxy_to - Rxy_from)/frame_range
        else:
            Axy = Rxy_fixed
        if Rxz_fixed == 0:
            Rxz = (Rxz_to - Rxz_from)/frame_range
        else:
            Axz = Rxz_fixed
        if Rxw_fixed == 0:
            Rxw = (Rxw_to - Rxw_from)/frame_range
        else:
            Axw = Rxw_fixed
        if Ryz_fixed == 0:
            Ryz = (Ryz_to - Ryz_from)/frame_range
        else:
            Ayz = Ryz_fixed
        if Ryw_fixed == 0:
            Ryw = (Ryw_to - Ryw_from)/frame_range
        else:
            Ayw = Ryw_fixed
        if Rzw_fixed == 0:
            Rzw = (Rzw_to - Rzw_from)/frame_range
        else:
            Azw = Rzw_fixed

    for frame in range(frame_start,frame_end+1):
        if frame_range==0: # if frame_range==0 there is only one frame to be processed (??)
            # set fixed to middle of transformation ranges
            Mx=(TRx_to + TRx_from)/2
            My=(TRy_to + TRy_from)/2
            Mz=(TRz_to + TRz_from)/2
            Mw=(TRw_to + TRw_from)/2

            Axy=(Rxy_to + Rxy_from)/2
            Axz=(Rxz_to + Rxz_from)/2
            Axw=(Rxw_to + Rxw_from)/2
            Ayz=(Ryz_to + Ryz_from)/2
            Ayw=(Ryw_to + Ryw_from)/2
            Azw=(Rzw_to + Rzw_from)/2
        else:
            frame_reset=frame-frame_start+1
            # compute translation for each dimension for this frame
            if TRx_fixed==0:
                Mx = TRx_from+(frame_reset-1)*Tx
            if TRy_fixed==0:
                My = TRy_from+(frame_reset-1)*Ty
            if TRz_fixed==0:
                Mz = TRz_from+(frame_reset-1)*Tz
            if TRw_fixed==0:
                Mw = TRw_from+(frame_reset-1)*Tw

            # compute rotations for each 4D plane at this frame
            if Rxy_fixed==0: 
                Axy = Rxy_from+(frame_reset-1)*Rxy
            if Rxz_fixed==0:
                Axz = Rxz_from+(frame_reset-1)*Rxz
            if Rxw_fixed==0:
                Axw = Rxw_from+(frame_reset-1)*Rxw
            if Ryz_fixed==0:
                Ayz = Ryz_from+(frame_reset-1)*Ryz
            if Ryw_fixed==0:
                Ayw = Ryw_from+(frame_reset-1)*Ryw
            if Rzw_fixed==0:
                Azw = Rzw_from+(frame_reset-1)*Rzw
        hyperobject_name="hyperObject." + str(frame)
        # transform the 4D object, find the intersections with the 3D realm, define faces, and generate the 3D object 
        make_transform(hyperobject_name,frame,[Mx,My,Mz,Mw],[Axy,Axz,Axw,Ayz,Ayw,Azw],vertices4D,edges4D,polygons4D,object_origin4D,show_wire,context)
        
        #Create text with translation and rotation info
        text_name="Text_TR."+str(frame)
        font_curve_name="font_curve."+str(frame)
        C=make_text_object(text_name,font_curve_name,Mx,My,Mz,Mw,Axy,Axz,Axw,Ayz,Ayw,Azw,context)
        insert_keyframe(frame,C)
        C.hide_viewport=False

    #set current frame in the middle of the frame range
    set_frame=round((((frame_end-frame_start)/2)+frame_start))
    scene.frame_set(set_frame)
    print("3D visualizations of transformed 4D objects have been created")

""" Make 3D object from 4D object and add hide/unhide keyframes """    
def make_transform(hyperobject_name,frame,move4D,angle4D,vertices4D,edges4D,polygons4D,object_origin4D,show_wire,context):
    hyperobj=Transform_4D_hyperobject(hyperobject_name,move4D,angle4D,vertices4D,edges4D,polygons4D,object_origin4D,show_wire,context) #add 3d object from 4d object
    if not hyperobj:
        return # no 3D object could be generated as the 4D object does not intersect with the realm
    insert_keyframe(frame,hyperobj)
    hyperobj.hide_viewport=False
    if show_wire: # seems like an overkill but otherwise the GN modifier will not show in viewport
        hyperobj.modifiers["GN"].show_viewport = False
        hyperobj.modifiers["GN"].show_viewport = True


""" Add hide/unhide viewport and render keyframes to 3D object """ 
def insert_keyframe(frame,C):
    C.hide_viewport = True
    C.keyframe_insert('hide_viewport', frame=frame-1)
    C.hide_viewport = False
    C.keyframe_insert('hide_viewport', frame=frame)
    C.hide_viewport = True
    C.keyframe_insert('hide_viewport', frame=frame+1)
    C.hide_render = True
    C.keyframe_insert('hide_render', frame=frame-1)
    C.hide_render = False
    C.keyframe_insert('hide_render', frame=frame)
    C.hide_render = True
    C.keyframe_insert('hide_render', frame=frame+1)
    return

""" Transform 4D object """ 
def Transform_4D_hyperobject(hyperobject_name,translation4D,rotation4D,vertices4D,edges4D,polygons4D,object_origin4D,show_wire,context):
    tmatrix = get_transformation_matrix(translation4D,rotation4D) # get transformation matrix
    # Transform all 4D vertices
    vertices4D_transformed = np.c_[vertices4D, np.ones(len(vertices4D))] # add column of ones
    vertices4D_transformed = np.dot(tmatrix,vertices4D_transformed.T).T #transform all vertices of the 4D object
    vertices4D_transformed = np.delete(vertices4D_transformed, -1, axis=1) # remove last column
    # transform 4D origin
    object_origin4D_transformed = np.append(object_origin4D, 1) # add column of ones
    object_origin4D_transformed = np.dot(tmatrix,object_origin4D_transformed.T).T #transform
    object_origin4D_transformed = np.delete(object_origin4D_transformed, -1, axis=0) #remove last col
    realm = np.r_[x,y,z] #4x3 matrix
    vertices4D_transformed=np.round(vertices4D_transformed,decimals=5)
    hyperobj=find_intersections_polygons(hyperobject_name,realm,vertices4D_transformed,polygons4D,object_origin4D_transformed,show_wire,context)
    return hyperobj

""" Generate transformation matrix """
def get_transformation_matrix(translation4D,rotation4D):
    dx = translation4D[0]
    dy = translation4D[1]
    dz = translation4D[2]
    dw = translation4D[3]
    xyC, xyS = trig(rotation4D[0])
    xzC, xzS = trig(rotation4D[1])
    xwC, xwS = trig(rotation4D[2])
    yzC, yzS = trig(rotation4D[3])
    ywC, ywS = trig(rotation4D[4])
    zwC, zwS = trig(rotation4D[5])
    
    Translate_matrix = np.array([[1, 0, 0, 0, dx],
                                [0, 1, 0, 0, dy],
                                [0, 0, 1, 0, dz],
                                [0, 0, 0, 1, dw],
                                [0, 0, 0, 0, 1]])

    Rotate_xy_matrix = np.array([[xyC, -xyS, 0, 0, 0], #zw-axis rotation
                                [xyS, xyC, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
    Rotate_xz_matrix = np.array([[xzC, 0, -xzS, 0, 0], #yw-axis rotation
                                [0, 1, 0, 0, 0],
                                [xzS, 0, xzC, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
    Rotate_xw_matrix = np.array([[xwC, 0, 0,-xwS, 0],  #yz-axis rotation
                                [0, 1, 0, 0, 0],
                                [0, 0, 1, 0, 0],
                                [xwS, 0, 0, xwC, 0],
                                [0, 0, 0, 0, 1]])
    Rotate_yz_matrix = np.array([[1, 0, 0, 0, 0],      #xw-axis rotation
                                [0, yzC, -yzS, 0, 0],
                                [0, yzS, yzC, 0, 0],
                                [0, 0, 0, 1, 0],
                                [0, 0, 0, 0, 1]])
    Rotate_yw_matrix = np.array([[1, 0, 0, 0, 0],      #xz-axis rotation
                                [0, ywC, 0, -ywS, 0],
                                [0, 0, 1, 0, 0],
                                [0, ywS, 0, ywC, 0],
                                [0, 0, 0, 0, 1]])
    Rotate_zw_matrix = np.array([[1, 0, 0, 0, 0],      #xy-axis rotation
                                [0, 1, 0, 0, 0],
                                [0, 0, zwC, -zwS, 0],
                                [0, 0, zwS, zwC, 0],
                                [0, 0, 0, 0, 1]])
    return np.matmul(Translate_matrix,np.matmul(Rotate_zw_matrix,np.matmul(Rotate_yw_matrix,
                    np.matmul(Rotate_xw_matrix,np.matmul(Rotate_xz_matrix,np.matmul(Rotate_xy_matrix,
                    Rotate_yz_matrix))))))

""" Convert angle to radians and determine cosine and sine values """ 
def trig(angle_degrees):
    angle_radians = np.radians(angle_degrees)
    return np.cos(angle_radians), np.sin(angle_radians)

""" Find the intersections of transformed 4D polygons with the realm"""
def find_intersections_polygons(hyperobject_name,realm,vertices4D_transformed,polygons4D,object_origin4D_transformed,show_wire,context):
    normal=get_normal(realm) #get the normal to the realm
    
    # for every every edge of every 4D polygon check if it intersects with the realm
    polygons_intersections=[]
    vertices_intersections=[]
    edges4Dto3D_indexes=[]
    indexcount=0
    new_edge=[]
    new_face=[]
    intersections=[]
    faces4D=[]
    rowblock=0
    edge_block=[]
    vert_block=[]
    phi=[]
    theta=[]
    r=[]
    hyperobj=[]
    # in xyzw_function_surface_faces() polygons4D is created in batches of 5 so we look for
    # intersections with the 3D realm at w = 0 for each batch of 5 polygons4D and when possible
    # create a 3D face from these intersections.
    for row in range(len(polygons4D)):
        rowblock+=1    
        if rowblock==5:   
            check=True
            rowblock=0
        else:
            check=False
        rowlength=len(polygons4D[row])
        for index in range(rowlength):
            if index == rowlength-1:
                edge=(vertices4D_transformed[polygons4D[row][index]],vertices4D_transformed[polygons4D[row][0]])
            else: 
                edge=(vertices4D_transformed[polygons4D[row][index]],vertices4D_transformed[polygons4D[row][index+1]])
            intersection=edge4D_4Dplane_intersection(realm,edge,object_origin4D_transformed,normal)        
            for vertex in intersection:
                vertices_intersections.append(vertex.tolist())
                
        #remove doubles
        for i in vertices_intersections: 
            if i not in intersections:
                intersections.append(i)        

        # "intersections" contains all intersections with the 3D ream of polygon defined in "row"
        intersections_count=len(intersections)
        if intersections_count>1: # no intersections found for this polygons
            for i in range(intersections_count):
                polygons_intersections.append(intersections[i])
                indexcount+=1

                # generate edge            
                if intersections_count!=1:
                    if i==(intersections_count-1): # if last loop:
                        if intersections_count>2:
                            new_edge=[indexcount-1,indexcount-intersections_count]
                    else:
                        new_edge=[indexcount-1,indexcount]
                    if new_edge:
                        edges4Dto3D_indexes.append(new_edge)
                        edge_block.append(new_edge)
                        vert_block.append(new_edge[0])
                        vert_block.append(new_edge[1])
                    new_edge=[]
        
        # Generate face for each edge_block      
        if check and len(edge_block)>1:
            vert_block=list(dict.fromkeys(vert_block))
            piece=([polygons_intersections[b] for b in vert_block])
            piece=np.array(piece).dot(realm.T)
            piece=np.round(piece,decimals=5)
            piece=piece+0
            piece,returnindex,returninverse,returncounts=np.unique(piece,
                                        return_index=True,
                                        return_inverse=True,
                                        return_counts=True,
                                        axis=0
                                        )
            if len(returnindex)>2:
                new_face=([vert_block[b] for b in returnindex])
           
                # rebuild edges based on removing doubles
                edge_block_new=[]
                for e in edge_block:
                    e[0]=new_face[returninverse[vert_block.index(e[0])]]
                    e[1]=new_face[returninverse[vert_block.index(e[1])]]
                    if e[0]!=e[1] and e not in edge_block_new and sorted(e) not in edge_block_new:
                        edge_block_new.append(e)
                edge_block=edge_block_new

                # find center of vertices
                a=piece 
                cent=(sum([p[0] for p in a])/len(a),sum([p[1] for p in a])/len(a),sum([p[2] for p in a])/len(a))
                for vert in a:
                    r_a,theta_a,phi_a=asSpherical(vert,cent)
                    theta.append(theta_a)
                    phi.append(phi_a)
                    r.append(r_a)
                if cent[2]<0:
                    sorted_index_phi=sorted(range(len(phi)), key=phi.__getitem__,reverse=True) # sort but give indexes only
                else:
                    sorted_index_phi=sorted(range(len(phi)), key=phi.__getitem__) # sort but give indexes only
                new_face=[new_face[h] for h in sorted_index_phi]
                if len(edge_block)>=len(new_face)-1:
                    faces4D.append(new_face)
                phi=[]
                theta=[]
                r=[]                
            new_face=[]
            edge_block=[]
            vert_block=[]
        vertices_intersections=[]
        intersections=[]
     
    if len(polygons_intersections)==0: # no polygons were created so 4D object does not intersect with realm at this transformation stage 
        return
   
    # rotate the intersections + the transformed 4D origin by the realm 
    # so that the intersection is now in xyz realm 
    polygons_intersections=np.array(polygons_intersections).dot(realm.T)
    object_origin3D_transformed=object_origin4D_transformed.dot(realm.T)
    hyperobj=make_3D_object(hyperobject_name,polygons_intersections,[],faces4D,object_origin3D_transformed,show_wire,context)
    
    return hyperobj

""" Make the 3D object from the intersection vertices """
def make_3D_object(hyperobject_name,verts,edges,faces,object_origin3D_transformed,show_wire,context):
    print("creating:",hyperobject_name)
    hypermesh = bpy.data.meshes.new('new_mesh')
    hypermesh.from_pydata(verts, edges, faces)
    hypermesh.update()
    ## make object from mesh
    #hyperobject = bpy.data.objects.new(hyperobject_name, hypermesh)
    ## add object to scene collection
    #bpy.context.collection.objects.link(hyperobject)
    hyperobject = object_utils.object_data_add(context, hypermesh, operator=None)
    hyperobject.name=hyperobject_name
    # add material
    mat_name="hyperObject_mat"
    # Get object material
    mat = bpy.data.materials.get(mat_name)
    if not mat: # if it does not exist, create it
        create_material(mat_name,(0, 0, 1, 1))
    # Assign material to object
    hyperobject.data.materials.append(mat)
    merge_vertices(hyperobject,0.0001)
    if show_wire:
        mat_name="wireframe_mat"
        mat = bpy.data.materials.get(mat_name)
        if not mat: # if it does not exist, create it
            color=(0,0,1,1) # blue
            create_material(mat_name,color)
        add_geometrynodes("GN","wireframe_tree",mat_name,hyperobject)
    return hyperobject

''' merge vertices '''
def merge_vertices(obj,merge_threshold):
    bpy.context.view_layer.objects.active = obj
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold = merge_threshold)
    bpy.ops.object.mode_set(mode='OBJECT')

""" Create text with translation and rotation data"""
def make_text_object(object_name,font_curve_name,Mx,My,Mz,Mw,Axy,Axz,Axw,Ayz,Ayw,Azw,context):
    font_curve = bpy.data.curves.new(type="FONT", name=font_curve_name)
    font_curve.body = ('Rotate\nxy {0:>+3.0f}\nxz {1:>+3.0f}\nxw {2:>+3.0f}\nyz {3:>+3.0f}\nyw {4:>+3.0f}\nzw {5:>+3.0f}\n\n\n\n\n\n\n\n\n\n\nTranslate\nx {6:>+3.1f}\ny {7:>+3.1f}\nz {8:>+3.1f}\nw {9:>+3.1f}'.format(round(Axy,0),round(Axz,0),round(Axw,0),round(Ayz,0),round(Ayw,0),round(Azw,0),round(Mx,1),round(My,1),round(Mz,1),round(Mw,1)))
    #font_obj = bpy.data.objects.new(name=object_name, object_data=font_curve)
    font_obj = object_utils.object_data_add(context, font_curve, operator=None)
    font_obj.name=object_name
    # Link text object to Camera
    ob = bpy.context.scene.objects[object_name]  # Get the object
    bpy.ops.object.select_all(action='DESELECT') # Deselect all objects
    bpy.context.view_layer.objects.active = ob   # Make the object the active object 
    ob.select_set(True)                          # Select the object
    bpy.context.object.data.extrude = 0.02 
    bpy.data.objects[object_name].scale = (0.076,0.076,0.076)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='MEDIAN') #set origin to the middle
    cam = bpy.context.scene.camera
    bpy.context.view_layer.objects.active = cam   # Make the camera the active object 
    bpy.ops.object.parent_no_inverse_set() # Parent text to camera without inverse
    # Set offset from Camera
    bpy.data.objects[object_name].location.x += 1.4901
    bpy.data.objects[object_name].location.y += 0
    bpy.data.objects[object_name].location.z += -4.62
    mat_name="text_material"
    # Get object material
    mat = bpy.data.materials.get(mat_name)
    color=(1,1,1,1) # white
    if not mat: # if it does not exist, create it
        create_material(mat_name,color)
    ob.data.materials.append(mat) # Assign material to object
    return font_obj

def asSpherical(xyz,cent):
    #takes list xyz (single coord)
    x       = xyz[0]-cent[0]
    y       = xyz[1]-cent[1]
    z       = xyz[2]-cent[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  acos(z/r)
    phi     =  atan2(y,x)
    return [r,theta,phi]

""" Find the intersection vertices """
def edge4D_4Dplane_intersection(realm,segment,object_origin4D_transformed,normal):
    # find a point at the intersection of a realm and a segment
    # in case when the entire segments lies on the realm, returns two endpoints
    # segment is given as matrix 2x4
    # adapted from wikipedia https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    direction=segment[1]-segment[0]
    if all(x == 0 for x in list(direction)):
        return []
    length=np.linalg.norm(direction)
    direction=direction/length
    if abs(direction.dot(normal)) < PRECISION:
        # segment is parallel to the realm
        if abs(segment[0].dot(normal)) < PRECISION:
            # segment is parallel and contained in the realm
            return segment
        else:
            # line is not intersecting with the realm
            return []
    else:
        # line is intersecting with the realm
        d = ([0,0,0,0]-segment[0]).dot(normal)/direction.dot(normal)
        if d >= 0 and d <= length:
            # segment is intersecting with the realm
            vertex=[segment[0] + d*direction]
            return vertex
        else:
            return []

""" Get normal to the realm """
def get_normal(realm):
    A = (realm[1][0] * realm[2][1]) - (realm[1][1] * realm[2][0])
    B = (realm[1][0] * realm[2][2]) - (realm[1][2] * realm[2][0])
    C = (realm[1][0] * realm[2][3]) - (realm[1][3] * realm[2][0])
    D = (realm[1][1] * realm[2][2]) - (realm[1][2] * realm[2][1])
    E = (realm[1][1] * realm[2][3]) - (realm[1][3] * realm[2][1])
    F = (realm[1][2] * realm[2][3]) - (realm[1][3] * realm[2][2])
    result=((realm[0][1] * F) - (realm[0][2] * E) + (realm[0][3] * D)),(- (realm[0][0] * F) + (realm[0][2] * C) - (realm[0][3] * B)),((realm[0][0] * E) - (realm[0][1] * C) + (realm[0][3] * A)),(- (realm[0][0] * D) + (realm[0][1] * B) - (realm[0][2] * A))
    return result

""" Get the vertices and faces of the object defined by the formulas for x,y,z and w """
def xyzw_function_surface_faces(self, x_eq, y_eq, z_eq, w_eq,
            range_u_min, range_u_max, range_u_step, wrap_u,
            range_v_min, range_v_max, range_v_step,wrap_v,
            range_t_min, range_t_max, range_t_step, wrap_t,
            a_eq, b_eq, c_eq, f_eq, g_eq, h_eq,close_v, close_t):

    verts = []
    edges=[]
    faces = []

    # Distance of each step in Blender Units
    uStep = (range_u_max - range_u_min) / range_u_step
    vStep = (range_v_max - range_v_min) / range_v_step
    tStep = (range_t_max - range_t_min) / range_t_step

    # Number of steps in the vertex creation loops.
    # Number of steps is the number of faces
    #   => Number of points is +1 unless wrapped.
    uRange = range_u_step + 1
    vRange = range_v_step + 1
    tRange = range_t_step + 1

    if wrap_u:
        uRange = uRange - 1

    if wrap_v:
        vRange = vRange - 1
        
    if wrap_t:
        tRange = tRange - 1

    try:
        expr_args_x = (
            compile(x_eq, __file__.replace(".py", "_x.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_y = (
            compile(y_eq, __file__.replace(".py", "_y.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_z = (
            compile(z_eq, __file__.replace(".py", "_z.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_w = (
            compile(w_eq, __file__.replace(".py", "_w.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_a = (
            compile(a_eq, __file__.replace(".py", "_a.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_b = (
            compile(b_eq, __file__.replace(".py", "_b.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_c = (
            compile(c_eq, __file__.replace(".py", "_c.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_f = (
            compile(f_eq, __file__.replace(".py", "_f.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_g = (
            compile(g_eq, __file__.replace(".py", "_g.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
        expr_args_h = (
            compile(h_eq, __file__.replace(".py", "_h.py"), 'eval'),
            {"__builtins__": None},
            safe_dict)
    except:
        import traceback
        self.report({'WARNING'}, "Error parsing expression(s) - "
                    "Check the console for more info")
        print("\n[Add X, Y, Z Function Surface]:\n\n", traceback.format_exc(limit=1))
        return [], [], []

    for tN in range(tRange):
        t = range_t_min + (tN * tStep)

        for vN in range(vRange):
            v = range_v_min + (vN * vStep)
            
            for uN in range(uRange):
                u = range_u_min + (uN * uStep)

                safe_dict['u'] = u
                safe_dict['v'] = v
                safe_dict['t'] = t

                # Try to evaluate the equations.
                try:
                    safe_dict['a'] = float(eval(*expr_args_a))
                    safe_dict['b'] = float(eval(*expr_args_b))
                    safe_dict['c'] = float(eval(*expr_args_c))
                    safe_dict['f'] = float(eval(*expr_args_f))
                    safe_dict['g'] = float(eval(*expr_args_g))
                    safe_dict['h'] = float(eval(*expr_args_h))

                    vertsrow=(
                        float(eval(*expr_args_x)),
                        float(eval(*expr_args_y)),
                        float(eval(*expr_args_z)),
                        float(eval(*expr_args_w)))
                    verts.append(np.round(vertsrow,decimals=5)+0)
                except:
                    import traceback
                    self.report({'WARNING'}, "Error evaluating expression(s) - "
                                 "Check the console for more info")
                    print("\n[Add X, Y, Z Function Surface]:\n\n", traceback.format_exc(limit=1))
                    return [], [], []

            #""" Here we build the faces """
            #fr=len(verts)-uRange
            #to=len(verts)-1
            #print("from:",fr,"to:",to)
            #if vN>0 and tN>0:
            #    for vert_index in range(fr,to):
            #        print("vert_index:",vert_index)
            #        face=[vert_index-uRange,
            #            vert_index+1-uRange,
            #            vert_index+1,
            #            vert_index]                
            #        faces.append(face)
            #        print("vN>0 ",faces[-1],len(faces)-1)
            #        # "extrude" face to previous v    
            #        faces.append([face[0]-(vRange*uRange),
            #                    face[1]-(vRange*uRange),
            #                    face[1],
            #                    face[0]])
            #        faces.append([face[1]-(vRange*uRange),
            #                    face[2]-(vRange*uRange),
            #                    face[2],
            #                    face[1]])
            #        faces.append([face[2]-(vRange*uRange),
            #                    face[3]-(vRange*uRange),
            #                    face[3],
            #                    face[2]])
            #        faces.append([face[3]-(vRange*uRange),
            #                    face[0]-(vRange*uRange),
            #                    face[0],
            #                    face[3]])
            #        faces.append([face[0]-(vRange*uRange),
            #                    face[1]-(vRange*uRange),
            #                    face[2]-(vRange*uRange),
            #                    face[3]-(vRange*uRange)])
            #        print("tN>0 ",faces[-5:],len(faces)-1)
            #        if wrap_u and vert_index==to-1:
            #            face=[vert_index+1-uRange,
            #                vert_index-2-uRange,
            #                vert_index+2-uRange,
            #                vert_index+1]                
            #            faces.append(face)
            #            print("vN>0 ",faces[-1],len(faces)-1)
            #            # "extrude" face to previous v    
            #            faces.append([face[0]-(vRange*uRange),
            #                        face[1]-(vRange*uRange),
            #                        face[1],
            #                        face[0]])
            #            faces.append([face[1]-(vRange*uRange),
            #                        face[2]-(vRange*uRange),
            #                        face[2],
            #                        face[1]])
            #            faces.append([face[2]-(vRange*uRange),
            #                        face[3]-(vRange*uRange),
            #                        face[3],
            #                        face[2]])
            #            faces.append([face[3]-(vRange*uRange),
            #                        face[0]-(vRange*uRange),
            #                        face[0],
            #                        face[3]])
            #            faces.append([face[0]-(vRange*uRange),
            #                        face[1]-(vRange*uRange),
            #                        face[2]-(vRange*uRange),
            #                        face[3]-(vRange*uRange)])
            #            print("tN>0 ",faces[-5:],len(faces)-1)

            #fr=len(verts)-uRange
            #to=len(verts)
            #if close_v:
            #    if vN==0: # make face at first v step
            #        face=np.arange(fr,to)[::-1]
            #        faces.append(face)
            #        print(faces[-1])
            #    if vN==vRange-1: # make face at the last v step
            #        face=np.arange(fr,to)
            #        faces.append(face)
            #        print(faces[-1])
            
            #if close_t:
            #    if tN==0: # make face at first t step
            #        face=np.arange(fr,to)[::-1]
            #        faces.append(face)
            #        print(faces[-1])
            #    if tN==tRange-1: # make face at the last t step
            #        face=np.arange(fr,to)
            #        faces.append(face)
            #        print(faces[-1])

    for tN in range(range_t_step):
        tNext = tN + 1

        if wrap_t and (tNext >= tRange):
            tNext = 0
            
        for vN in range(range_v_step):
            vNext = vN + 1

            if wrap_v and (vNext >= vRange):
                vNext = 0

            for uN in range(range_u_step):
                uNext = uN + 1

                if wrap_u and (uNext >= uRange):
                    uNext = 0

                face=[(vNext * uRange) + uNext + (uRange*vRange*tN),
                    (vNext * uRange) + uN + (uRange*vRange*tN),
                    (vN * uRange) + uN + (uRange*vRange*tN),
                    (vN * uRange) + uNext + (uRange*vRange*tN)]                
                faces.append(face)

                faces.append([face[0],
                            face[1],
                            face[1]+(uRange*vRange),
                            face[0]+(uRange*vRange)])
                faces.append([face[1],
                            face[2],
                            face[2]+(uRange*vRange),
                            face[1]+(uRange*vRange)])
                faces.append([face[2],
                            face[3],
                            face[3]+(uRange*vRange),
                            face[2]+(uRange*vRange)]) 
                faces.append([face[3],
                            face[0],
                            face[0]+(uRange*vRange),
                            face[3]+(uRange*vRange)])


    return verts,edges,faces


class AddXYZWFunctionSurface(Operator):
    bl_idname = "mesh.primitive_xyzw_function_surface"
    bl_label = "Add X, Y, Z, W Function Surface"
    bl_description = ("Add a surface defined defined by 4 functions:\n"
                      "x=F1(u,v,t), y=F2(u,v,t) z=F3(u,v,t) and w=F4(u,v,t)")
    bl_options = {'REGISTER', 'UNDO', 'PRESET'}

    x_eq: StringProperty(
                name="X equation",
                description="Equation for x=F(u,v,t). "
                            "Also available: a, b, c, f, g, h",
                default="a*sin(pi*t)*sin(pi*v)*sin(2*pi*u)"
                )
    y_eq: StringProperty(
                name="Y equation",
                description="Equation for y=F(u,v,t). "
                            "Also available: a, b, c, f, g, h",
                default="a*sin(pi*t)*sin(pi*v)*cos(2*pi*u)"
                )
    z_eq: StringProperty(
                name="Z equation",
                description="Equation for z=F(u,v,t). "
                            "Also available: a, b, c, f, g, h",
                default="a*sin(pi*t)*cos(pi*v)"
                )
    w_eq: StringProperty(
                name="W equation",
                description="Equation for z=F(u,v,t). "
                            "Also available: a, b, c, f, g, h",
                default="a*cos(pi*t)"
                )
    range_u_min: FloatProperty(
                name="u min",
                description="Minimum u value. Lower boundary of u range",
                min=-100.00,
                max=100.00,
                default=0.00
                )
    range_u_max: FloatProperty(
                name="u max",
                description="Maximum u value. Upper boundary of u range",
                min=-100.00,
                max=100.00,
                default=1
                )
    range_u_step: IntProperty(
                name="u step",
                description="u Subdivisions",
                min=1,
                max=1024,
                default=8
                )
    wrap_u: BoolProperty(
                name="u wrap",
                description="u Wrap around",
                default=False
                )
    range_v_min: FloatProperty(
                name="v min",
                description="Minimum v value. Lower boundary of v range",
                min=-100.00,
                max=100.00,
                default=0.00
                )
    range_v_max: FloatProperty(
                name="v max",
                description="Maximum v value. Upper boundary of v range",
                min=-100.00,
                max=100.00,
                default=1
                )
    range_v_step: IntProperty(
                name="v step",
                description="v Subdivisions",
                min=1,
                max=1024,
                default=8
                )
    wrap_v: BoolProperty(
                name="v wrap",
                description="v Wrap around",
                default=False
                )
    close_v: BoolProperty(
                name="close v (not active)",
                description="Create faces for first and last "
                            "v values (only if u is wrapped)",
                default=False
                )
    range_t_min: FloatProperty(
                name="t min",
                description="Minimum t value. Lower boundary of t range",
                min=-100.00,
                max=100.00,
                default=0.00
                )
    range_t_max: FloatProperty(
                name="t max",
                description="Maximum t value. Upper boundary of t range",
                min=-100.00,
                max=100.00,
                default=1
                )
    range_t_step: IntProperty(
                name="t step",
                description="t Subdivisions",
                min=1,
                max=1024,
                default=4
                )
    wrap_t: BoolProperty(
                name="t wrap",
                description="t Wrap around",
                default=False
                )
    close_t: BoolProperty(
                name="close t (not active)",
                description="Create faces for first and last "
                            "t values (only if v is wrapped)",
                default=False
                )
    a_eq: StringProperty(
                name="a function",
                description="Equation for a=F(u,v,t). Also available: n",
                default="1"
                )
    b_eq: StringProperty(
                name="b function",
                description="Equation for b=F(u,v,t). Also available: n",
                default="0"
                )
    c_eq: StringProperty(
                name="c function",
                description="Equation for c=F(u,v,t). Also available: n",
                default="0"
                )
    f_eq: StringProperty(
                name="f function",
                description="Equation for f=F(u,v,t). Also available: n, a, b, c",
                default="0"
                )
    g_eq: StringProperty(
                name="g function",
                description="Equation for g=F(u,v,t). Also available: n, a, b, c",
                default="0"
                )
    h_eq: StringProperty(
                name="h function",
                description="Equation for h=F(u,v,t). Also available: n, a, b, c",
                default="0"
                )
    show_wire : BoolProperty(
            name="Show wireframe",
            default=False,
            description="Add the object’s wireframe over solid drawing"
            )
    edit_mode : BoolProperty(
            name="Show in edit mode",
            default=False,
            description="Show in edit mode"
            )
    TRx_from : FloatProperty(
            name="From",
            description="x value in Blender units to translate from",
            default=0.00
            )
    TRx_to : FloatProperty(
            name="To",
            description="x value in Blender units to translate to",
            default=0.00
            )
    TRx_fixed : FloatProperty(
            name="Fixed",
            description="define fixed value for x (overrides from/to)",
            default=0.00
            )
    TRy_from : FloatProperty(
            name="From",
            description="y value in Blender units to translate from",
            default=0.00
            )
    TRy_to : FloatProperty(
            name="To",
            description="y value in Blender units to translate to",
            default=0.00
            )
    TRy_fixed : FloatProperty(
            name="Fixed",
            description="define fixed value for y (overrides from/to)",
            default=0.00
            )
    TRz_from : FloatProperty(
            name="From",
            description="z value in Blender units to translate from",
            default=0.00
            )
    TRz_to : FloatProperty(
            name="To",
            description="z value in Blender units to translate to",
            default=0.00
            )
    TRz_fixed : FloatProperty(
            name="Fixed",
            description="define fixed value for z (overrides from/to)",
            default=0.00
            )
    TRw_from : FloatProperty(
            name="From",
            description="w value in Blender units to translate from",
            default=-1.25
            )
    TRw_to : FloatProperty(
            name="To",
            description="w value in Blender units to translate to",
            default=1.25
            )
    TRw_fixed : FloatProperty(
            name="Fixed",
            description="define fixed value for w (overrides from/to)",
            default=0.00
            )
    Rxy_from : IntProperty(
            name="From",
            description="Rotate xy-plane in degrees from this value",
            default=0
            )
    Rxy_to : IntProperty(
            name="To",
            description="Rotate xy-plane in degrees to this value",
            default=0
            )
    Rxy_fixed : IntProperty(
            name="Fixed",
            description="Fixed xy-plane rotation value in degrees (overrides from/to)",
            default=0
            )
    Rxz_from : IntProperty(
            name="From",
            description="Rotate xz-plane in degrees from this value",
            default=0
            )
    Rxz_to : IntProperty(
            name="To",
            description="Rotate xz-plane in degrees to this value",
            default=0
            )
    Rxz_fixed : IntProperty(
            name="Fixed",
            description="Fixed xz-plane rotation value in degrees (overrides from/to)",
            default=0
            )
    Rxw_from : IntProperty(
            name="From",
            description="Rotate xw-plane in degrees from this value",
            default=0
            )
    Rxw_to : IntProperty(
            name="To",
            description="Rotate xw-plane in degrees to this value",
            default=0
            )
    Rxw_fixed : IntProperty(
            name="Fixed",
            description="Fixed xw-plane rotation value in degrees (overrides from/to)",
            default=0
            )
    Ryz_from : IntProperty(
            name="From",
            description="Rotate yz-plane in degrees from this value",
            default=0
            )
    Ryz_to : IntProperty(
            name="To",
            description="Rotate yz-plane in degrees to this value",
            default=0
            )
    Ryz_fixed : IntProperty(
            name="Fixed",
            description="Fixed yz-plane rotation value in degrees (overrides from/to)",
            default=0
            )
    Ryw_from : IntProperty(
            name="From",
            description="Rotate yw-plane in degrees from this value",
            default=0
            )
    Ryw_to : IntProperty(
            name="To",
            description="Rotate yw-plane in degrees to this value",
            default=0
            )
    Ryw_fixed : IntProperty(
            name="Fixed",
            description="Fixed yw-plane rotation value in degrees (overrides from/to)",
            default=0
            )
    Rzw_from : IntProperty(
            name="From",
            description="Rotate zw-plane in degrees from this value",
            default=0
            )
    Rzw_to : IntProperty(
            name="To",
            description="Rotate zw-plane in degrees to this value",
            default=0
            )
    Rzw_fixed : IntProperty(
            name="Fixed",
            description="Fixed zw-plane rotation value in degrees (overrides from/to)",
            default=0
            )
    frame_start: IntProperty(
            name="Startframe",
            description="Startframe of rotations and/or transformations",
            min=1,
            default=1
            )
    frame_end: IntProperty(
            name="Endframe",
            description="Endframe of rotations and/or transformations",
            min=1,
            default=20
            )
    generate : BoolProperty(
            name="Visualize 4D transformation in 3D",
            default=False,
            description="Generate 3D visualization of transformed 4D objects"
            )

    # Display the options
    def draw(self, context):
        layout = self.layout
        layout.operator("wm.operator_defaults")

        box = layout.box()
        #box.label(text="Equations")
        col = box.column(align=True)
        col.prop(self, "x_eq")
        col.prop(self, "y_eq")
        col.prop(self, "z_eq")
        col.prop(self, "w_eq")

        box = layout.box()
        #box.label(text="Define u")
        col = box.column(align=True)
        col.prop(self, "range_u_min")
        col.prop(self, "range_u_max")
        col.prop(self, "range_u_step")
        col.prop(self, "wrap_u")

        box = layout.box()
        #box.label(text="Define v")
        col = box.column(align=True)
        col.prop(self, "range_v_min")
        col.prop(self, "range_v_max")
        col.prop(self, "range_v_step")
        col.prop(self, "wrap_v")
        col.prop(self, "close_v")

        box = layout.box()
        #box.label(text="Define t")
        col = box.column(align=True)
        col.prop(self, "range_t_min")
        col.prop(self, "range_t_max")
        col.prop(self, "range_t_step")
        col.prop(self, "wrap_t")
        col.prop(self, "close_t")

        box = layout.box()
        box.label(text="Helper Functions")
        col = box.column(align=True)
        col.prop(self, "a_eq")
        col.prop(self, "b_eq")
        col.prop(self, "c_eq")
        col.prop(self, "f_eq")
        col.prop(self, "g_eq")
        col.prop(self, "h_eq")

        box = layout.box()
        #box.label(text="")
        col = box.column(align=True)
        col.prop(self, "show_wire")
        col.prop(self, "edit_mode")

        box = layout.box()
        box.label(text="X plane translation")
        col = box.column(align=True)
        col.prop(self, "TRx_from")
        col.prop(self, "TRx_to")
        col.prop(self, "TRx_fixed")

        box = layout.box()
        box.label(text="Y plane translation")
        col = box.column(align=True)
        col.prop(self, "TRy_from")
        col.prop(self, "TRy_to")
        col.prop(self, "TRy_fixed")

        box = layout.box()
        box.label(text="Z plane translation")
        col = box.column(align=True)
        col.prop(self, "TRz_from")
        col.prop(self, "TRz_to")
        col.prop(self, "TRz_fixed")

        box = layout.box()
        box.label(text="W plane translation")
        col = box.column(align=True)
        col.prop(self, "TRw_from")
        col.prop(self, "TRw_to")
        col.prop(self, "TRw_fixed")

        box = layout.box()
        box.label(text="XY plane rotation")
        col = box.column(align=True)
        col.prop(self, "Rxy_from")
        col.prop(self, "Rxy_to")
        col.prop(self, "Rxy_fixed")

        box = layout.box()
        box.label(text="XY plane rotation")
        col = box.column(align=True)
        col.prop(self, "Rxz_from")
        col.prop(self, "Rxz_to")
        col.prop(self, "Rxz_fixed")

        box = layout.box()
        box.label(text="XW plane rotation")
        col = box.column(align=True)
        col.prop(self, "Rxw_from")
        col.prop(self, "Rxw_to")
        col.prop(self, "Rxw_fixed")

        box = layout.box()
        box.label(text="YZ plane rotation")
        col = box.column(align=True)
        col.prop(self, "Ryz_from")
        col.prop(self, "Ryz_to")
        col.prop(self, "Ryz_fixed")

        box = layout.box()
        box.label(text="YW plane rotation")
        col = box.column(align=True)
        col.prop(self, "Ryw_from")
        col.prop(self, "Ryw_to")
        col.prop(self, "Ryw_fixed")

        box = layout.box()
        box.label(text="ZW plane rotation")
        col = box.column(align=True)
        col.prop(self, "Rzw_from")
        col.prop(self, "Rzw_to")
        col.prop(self, "Rzw_fixed")
            
        box = layout.box()
        box.label(text="Set frame sequence for 4D->3D generate")
        col = box.column(align=True)
        col.prop(self, "frame_start")
        col.prop(self, "frame_end")

        box = layout.box()
        box.prop(self, "generate", toggle=True)

    def invoke(self,context,event):
        print("Start")
        return self.execute(context)
    
    def execute(self, context):
 
        if bpy.context.mode!='OBJECT': # if not in OBJECT mode, set OBJECT mode
            bpy.ops.object.mode_set(mode='OBJECT')
            bpy.ops.object.select_all(action='DESELECT')
        
        #unhide all hidden objects
        for obj in bpy.data.objects:
            obj.hide_set(False)
            obj.hide_viewport=False
        #delete all objects from previous run if applicable 
        delete_objects("hyperObject")
        delete_objects("Text_TR") 
        delete_objects("object_4D_3D")
        
        # remove unsused blocks
        for block in bpy.data.meshes:
            if block.users == 0:
                bpy.data.meshes.remove(block)
        for block in bpy.data.texts:
            if block.users == 0:
                bpy.data.texts.remove(block)
        for block in bpy.data.node_groups:
            if block.users == 0:
                bpy.data.node_groups.remove(block)
        for block in bpy.data.actions:
            if block.users == 0:
                bpy.data.actions.remove(block)
        for block in bpy.data.curves:
            if block.users == 0:
                bpy.data.curves.remove(block)
        for block in bpy.data.cameras:
            if block.users == 0:
                bpy.data.cameras.remove(block)

        
        verts4D,edges4D,faces4D = xyzw_function_surface_faces(
                            self,
                            self.x_eq,
                            self.y_eq,
                            self.z_eq,
                            self.w_eq,
                            self.range_u_min,
                            self.range_u_max,
                            self.range_u_step,
                            self.wrap_u,
                            self.range_v_min,
                            self.range_v_max,
                            self.range_v_step,
                            self.wrap_v,
                            self.range_t_min,
                            self.range_t_max,
                            self.range_t_step,
                            self.wrap_t,
                            self.a_eq,
                            self.b_eq,
                            self.c_eq,
                            self.f_eq,
                            self.g_eq,
                            self.h_eq,
                            self.close_v,
                            self.close_t
                            )
    
        if not verts4D:
            return {'CANCELLED'}

        verts4D=np.round(verts4D,decimals=5)
        verts4D=verts4D+0 # get rid of negative zeros

        # Create 4D object in 3D using only x,y,z
        verts3D=np.array(verts4D)[:,0:3] # remove last column w
        objxyz=make_3D_object(objxyz_object_name,verts3D,[],faces4D,[],self.show_wire,context)

        for obj in bpy.data.objects:
            obj.hide_set(False)
            obj.hide_viewport=False
        #delete all objects from previous run if applicable 
        delete_objects("hyperObject")
        delete_objects("Text_TR") 

        if self.edit_mode:
            bpy.ops.object.mode_set(mode = 'EDIT')
        else:
            bpy.ops.object.mode_set(mode = 'OBJECT')
        
        if self.generate: # generate 3D views of the 4D object

            #hide object_4D_3D 
            objxyz.hide_viewport=True

            #generate 3D objects from (4D object + defined transformation)
            generate_4D_to_3D(self.TRx_from,
                            self.TRx_to,
                            self.TRx_fixed,
                            self.TRy_from,
                            self.TRy_to,
                            self.TRy_fixed,
                            self.TRz_from,
                            self.TRz_to,
                            self.TRz_fixed,
                            self.TRw_from,
                            self.TRw_to,
                            self.TRw_fixed,
                            self.Rxy_from,
                            self.Rxy_to,
                            self.Rxy_fixed,
                            self.Rxz_from,
                            self.Rxz_to,
                            self.Rxz_fixed,
                            self.Rxw_from,
                            self.Rxw_to,
                            self.Rxw_fixed,
                            self.Ryz_from,
                            self.Ryz_to,
                            self.Ryz_fixed,
                            self.Ryw_from,
                            self.Ryw_to,
                            self.Ryw_fixed,
                            self.Rzw_from,
                            self.Rzw_to,
                            self.Rzw_fixed,
                            self.frame_start,
                            self.frame_end,
                            verts4D,
                            edges4D,
                            faces4D,
                            [0,0,0,0],
                            self.show_wire,
                            context
                            )
            self.generate=False

        return {'FINISHED'}