# pylint: disable=no-member
""" scatter using MarkersVisual """

import numpy as np
import sys
from FileManager import FileManager

from vispy import app, visuals, scene, gloo
from vispy.color import ColorArray

fileManager = FileManager()

# Define a simple vertex shader. We use $template variables as placeholders for
# code that will be inserted later on.
vertex_shader = """
void main()
{
    vec4 visual_pos = vec4($position, 1);
    vec4 doc_pos = $visual_to_doc(visual_pos);
    gl_Position = $doc_to_render(doc_pos);
}
"""

fragment_shader = """
void main() {
  gl_FragColor = $color;
}
"""


# now build our visuals
class Plot3DVisual(visuals.Visual):
    """template"""

    def __init__(self, x, y, z, color=(0.0, 1.0, 0.0, 1.0)):
        """plot 3D"""
        visuals.Visual.__init__(self, vertex_shader, fragment_shader)

        # build Vertices buffer
        data = np.c_[x, y, z]
        v = gloo.VertexBuffer(data.astype(np.float32))

        # bind data
        self.shared_program.vert["position"] = v
        self.shared_program.frag["color"] = color

        # config
        self.set_gl_state("opaque")
        self._draw_mode = "points"

    def _prepare_transforms(self, view):
        """This method is called when the user or the scenegraph has assigned
        new transforms to this visual"""
        # Note we use the "additive" GL blending settings so that we do not
        # have to sort the mesh triangles back-to-front before each draw.
        tr = view.transforms
        view_vert = view.view_program.vert
        view_vert["visual_to_doc"] = tr.get_transform("visual", "document")
        view_vert["doc_to_render"] = tr.get_transform("document", "render")


# The real-things : plot using scene
# build canvas
canvas = scene.SceneCanvas(keys='interactive', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera="turntable"
view.camera.fov = 45
#view.camera.scale_factor = 50

# Plotting the 3D localization lines
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
loc_lines = np.load(fileManager._get_path_3D_lines_image('168_100_1e-05_50_4_6'))
X, Y, Z = np.where(loc_lines[:,:,:,3]>30)
pos = np.array(list(zip(X,Y,Z)))
N = loc_lines[X,Y,Z].shape[0]
colors = ColorArray(loc_lines[X,Y,Z].astype(np.float64)/255.0)
p_lines = Scatter3D(parent=view.scene)
p_lines.set_gl_state('opaque', blend=False, depth_test=True)
p_lines.set_data(pos, face_color=colors, symbol='o', size=2,
            edge_width=0, edge_color=[0,0,0,0])

micro = np.load(fileManager._get_path_npy_image('168_100_1e-05_50_4_6'))
Nx,Ny,Nz = micro.shape
view.camera.center = (Nx/2, Ny/2, Nz/2)
view.camera.distance = max(Nz, max(Nx, Ny))

# Adding microstructure inclusions into the model
Plot3D = scene.visuals.create_visual_node(Plot3DVisual)
x_micro_1, y_micro_1, z_micro_1 = np.where(micro==2)
p_micro_1 = Plot3D(x_micro_1, y_micro_1, z_micro_1, parent=view.scene, color=(0.0, 1.0, 0.0, 1.0))
x_micro_2, y_micro_2, z_micro_2 = np.where(micro==3)
p_micro_2 = Plot3D(x_micro_2, y_micro_2, z_micro_2, parent=view.scene, color=(0.0, 0.0, 1.0, 1.0))

# run
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
