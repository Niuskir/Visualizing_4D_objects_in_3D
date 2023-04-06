# Visualizing_4D_objects_in_3D
Visualizing 4D objects in the 3D realm using Blender 3D (https://www.blender.org/).

This "4D Objects" Blender add-on is based on the "Add x,y,z Function Surface" part of the "Extra Objects" add-on which comes standard with blender.

It allows you to create 4D objects based on math formulas for x,y,z,w you define and will immediately show you how the 4D object looks like in 3D when the w dimension is removed.

When you have defined the 4D object correctly you can set any translation and/or rotation combination you want the 4D object to go trough in equal steps in the frame sequence you define. Clicking on "Generate" will create a 3D object for each frame separately based on the transformation of the 4D object at that frame. Generating these 3D views could take some (or considerable) time depending on the resolution you define for your 4D object.  

Download the zip file add_mesh_4D_objects_visualize.zip to install this add-on in Blender the usual way. 

Copy the presets to C:\Users\"Your directory"\AppData\Roaming\Blender Foundation\Blender\3.5\scripts\presets\operator\mesh.primitive_xyzw_function_surface 

The add-on has been tested to work correctly for a 4D hypersphere (default in the add-on) and a 4D hypertorus (as defined in file Hypertorus_4D.py). 

Here's a Youtube video made completely in Blender using this add-on https://youtu.be/-uN1YjbK5no

This was a hobby project and i hope someone can bring this add-on to the next level.
