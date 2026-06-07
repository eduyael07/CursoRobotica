import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/edu/CursoRobotica/proyecto/src/install/robot_hardware'
