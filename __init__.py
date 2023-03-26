# SPDX-License-Identifier: GPL-3.0-or-later

# Contributed to by:
# Pontiac, Fourmadmen, varkenvarken, tuga3d, meta-androcto, metalliandy     #
# dreampainter, cotejrp1, liero, Kayo Phoenix, sugiany, dommetysk, Jambay   #
# Phymec, Anthony D'Agostino, Pablo Vazquez, Richard Wilks, lijenstina,     #
# Sjaak-de-Draak, Phil Cote, cotejrp1, xyz presets by elfnor, revolt_randy, #
# Vladimir Spivak (cwolf3d), Niek Kort#


bl_info = {
    "name": "4D Objects",
    "author": "Multiple Authors",
    "version": (0, 0, 1),
    "blender": (3, 4, 0),
    "location": "View3D > Add > Mesh",
    "description": "Add 4D object types",
    "warning": "",
    "doc_url": "",
    "category": "Add Mesh",
}

if "bpy" in locals():
    import importlib
    importlib.reload(add_mesh_4d_function_surface)
else:
    from . import add_mesh_4d_function_surface

import bpy
from bpy.types import Menu

class VIEW4D_MT_mesh_math_add(Menu):
    # Define the "Math Function" menu
    bl_idname = "VIEW4D_MT_mesh_math_add"
    bl_label = "Math Functions"

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_REGION_WIN'
        layout.operator("mesh.primitive_xyzw_function_surface",text="XYZW Math Surface->3D view")

# Register all operators and panels

# Define "Extras" menu
def menu_func(self, context):
    layout = self.layout
    layout.operator_context = 'INVOKE_REGION_WIN'
    layout.separator()
    layout.operator("mesh.primitive_xyzw_function_surface",
                    text="XYZW Math Surface->3D view", icon="PACKAGE")

# Register
classes = [VIEW4D_MT_mesh_math_add,
           add_mesh_4d_function_surface.AddXYZWFunctionSurface]

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    # Add "Extras" menu to the "Add Mesh" menu 
    bpy.types.VIEW3D_MT_mesh_add.append(menu_func)


def unregister():
    # Remove "Extras" menu from the "Add Mesh" menu 
    bpy.types.VIEW3D_MT_mesh_add.remove(menu_func)

    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

if __name__ == "__main__":
    register()
    