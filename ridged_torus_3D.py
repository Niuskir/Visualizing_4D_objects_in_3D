import bpy
op = bpy.context.active_operator

op.x_eq = 'a*cos(u)+(b*sin(f*u)+c)*cos(u)*cos(v)'
op.y_eq = 'a*sin(u)+(b*sin(f*u)+c)*sin(u)*cos(v)'
op.z_eq = '(b*sin(f*u)+c)*sin(v)'
op.w_eq = '0'
op.range_u_min = 0.0
op.range_u_max = 6.2831854820251465
op.range_u_step = 128
op.wrap_u = False
op.range_v_min = 0.0
op.range_v_max = 6.2831854820251465
op.range_v_step = 32
op.wrap_v = False
op.close_v = False
op.range_t_min = 0.0
op.range_t_max = 1.0
op.range_t_step = 4
op.wrap_t = False
op.close_t = False
op.a_eq = '5'
op.b_eq = '0.6'
op.c_eq = '2'
op.f_eq = '10'
op.g_eq = '0'
op.h_eq = '0'
op.show_wire = False
op.edit_mode = False
op.TRx_from = 0.0
op.TRx_to = 0.0
op.TRx_fixed = 0.0
op.TRy_from = 0.0
op.TRy_to = 0.0
op.TRy_fixed = 0.0
op.TRz_from = 0.0
op.TRz_to = 0.0
op.TRz_fixed = 0.0
op.TRw_from = -1.25
op.TRw_to = 1.25
op.TRw_fixed = 0.0
op.Rxy_from = 0
op.Rxy_to = 0
op.Rxy_fixed = 0
op.Rxz_from = 0
op.Rxz_to = 0
op.Rxz_fixed = 0
op.Rxw_from = 0
op.Rxw_to = 0
op.Rxw_fixed = 0
op.Ryz_from = 0
op.Ryz_to = 0
op.Ryz_fixed = 0
op.Ryw_from = 0
op.Ryw_to = 0
op.Ryw_fixed = 0
op.Rzw_from = 0
op.Rzw_to = 0
op.Rzw_fixed = 0
op.frame_start = 1
op.frame_end = 10
op.generate = False
