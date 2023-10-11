import torch
import os
import numpy as np
import cv2
import PIL.Image as pil_img
import subprocess

subprocess.run(
    'pip install networkx==2.5'
    .split()
)

import gradio as gr

import trimesh
import plotly.graph_objects as go

from models.deco import DECO
from common import constants

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

description = '''
### DECO: Dense Estimation of 3D Human-Scene Contact in the Wild (ICCV 2023, Oral)
<table>
<th width="20%">
<ul>
<li><strong><a href="https://deco.is.tue.mpg.de/">Homepage</a></strong>
<li><strong><a href="https://github.com/sha2nkt/deco">Code</a></strong>
<li><strong><a href="https://openaccess.thecvf.com/content/ICCV2023/html/Tripathi_DECO_Dense_Estimation_of_3D_Human-Scene_Contact_In_The_Wild_ICCV_2023_paper.html">Paper</a></strong>
</ul>
<br>
<ul>
<li><strong>Colab Notebook</strong> <a href='https://colab.research.google.com/drive/1fTQdI2AHEKlwYG9yIb2wqicIMhAa067_?usp=sharing'><img style="display: inline-block;" src='https://colab.research.google.com/assets/colab-badge.svg' alt='Google Colab'></a></li>
</ul>
<br>
<iframe src="https://ghbtns.com/github-btn.html?user=sha2nkt&repo=deco&type=star&count=true&v=2&size=small" frameborder="0" scrolling="0" width="100" height="20"></iframe>
</th>
</table>

#### Citation
```
@InProceedings{tripathi2023deco,
    author    = {Tripathi, Shashank and Chatterjee, Agniv and Passy, Jean-Claude and Yi, Hongwei and Tzionas, Dimitrios and Black, Michael J.},
    title     = {{DECO}: Dense Estimation of {3D} Human-Scene Contact In The Wild},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {8001-8013}
}
```

<details>
<summary>More</summary>

#### Acknowledgments:
- [ECON](https://huggingface.co/spaces/Yuliang/ECON)

</details>
'''   

DEFAULT_LIGHTING = dict(
    ambient=0.6,
    diffuse=0.5,
    fresnel=0.01,
    specular=0.1,
    roughness=0.001)

DEFAULT_LIGHT_POSITION = dict(x=6, y=0, z=10)

def initiate_model(model_path):
    deco_model = DECO('hrnet', True, device)

    print(f'Loading weights from {model_path}')
    checkpoint = torch.load(model_path)
    deco_model.load_state_dict(checkpoint['deco'], strict=True)

    deco_model.eval()

    return deco_model   

def create_layout(dummy, camera=None):
    if camera is None:
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=dummy.x.mean(), y=0, z=3),
            projection=dict(type='perspective'))

    layout = dict(
        scene={
            "xaxis": {
                'showgrid': False,
                'zeroline': False,
                'visible': False,
                "range": [dummy.x.min(), dummy.x.max()]
            },
            "yaxis": {
                'showgrid': False,
                'zeroline': False,
                'visible': False,
                "range": [dummy.y.min(), dummy.y.max()]
            },
            "zaxis": {
                'showgrid': False,
                'zeroline': False,
                'visible': False,
                "range": [dummy.z.min(), dummy.z.max()]
            },
        },
        autosize=False,
        width=750, height=1000,
        scene_camera=camera,
        scene_aspectmode="data",
        clickmode="event+select",
        margin={'l': 0, 't': 0}
    )

    return layout

def create_fig(dummy, camera=None):
    fig = go.Figure(
        data=dummy.mesh_3d(),
        layout=create_layout(dummy, camera))
    return fig

class Dummy:

    def __init__(self, mesh_path):
        """A simple polygonal dummy with colored patches."""
        self._load_trimesh(mesh_path)

    def _load_trimesh(self, path):
        """Load a mesh given a path to a .PLY file."""
        self._trimesh = trimesh.load(path, process=False)
        self._vertices = np.array(self._trimesh.vertices)
        self._faces = np.array(self._trimesh.faces)
        self.colors = self._trimesh.visual.vertex_colors

    @property
    def vertices(self):
        """All the mesh vertices."""
        return self._vertices

    @property
    def faces(self):
        """All the mesh faces."""
        return self._faces

    @property
    def n_vertices(self):
        """Number of vertices in a mesh."""
        return self._vertices.shape[0]

    @property
    def n_faces(self):
        """Number of faces in a mesh."""
        return self._faces.shape[0]

    @property
    def x(self):
        """An array of vertex x coordinates"""
        return self._vertices[:, 0]

    @property
    def y(self):
        """An array of vertex y coordinates"""
        return self._vertices[:, 1]

    @property
    def z(self):
        """An array of vertex z coordinates"""
        return self._vertices[:, 2]

    @property
    def i(self):
        """An array of the first face vertices"""
        return self._faces[:, 0]

    @property
    def j(self):
        """An array of the second face vertices"""
        return self._faces[:, 1]

    @property
    def k(self):
        """An array of the third face vertices"""
        return self._faces[:, 2]

    @property
    def default_selection(self):
        """Default patch selection mask."""
        return dict(vertices=[])

    def mesh_3d(
            self,
            lighting=DEFAULT_LIGHTING,
            light_position=DEFAULT_LIGHT_POSITION
    ):
        """Construct a Mesh3D object give a clickmask for patch coloring."""

        return go.Mesh3d(
            x=self.x, y=self.y, z=self.z,
            i=self.i, j=self.j, k=self.k,
            vertexcolor=self.colors,
            lighting=lighting,
            lightposition=light_position,
            hoverinfo='none')

def main(pil_img, out_dir='demo_out', model_path='checkpoint/deco_best.pth', mesh_colour=[130, 130, 130, 255], annot_colour=[0, 255, 0, 255]):
    deco_model = initiate_model(model_path)
    
    smpl_path = os.path.join(constants.SMPL_MODEL_DIR, 'smpl_neutral_tpose.ply')
    
    img = np.array(pil_img)
    img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2,0,1)/255.0
    img = img[np.newaxis,:,:,:]
    img = torch.tensor(img, dtype = torch.float32).to(device)

    with torch.no_grad():
        cont, _, _ = deco_model(img)
    cont = cont.detach().cpu().numpy().squeeze()
    cont_smpl = []
    for indx, i in enumerate(cont):
        if i >= 0.5:
            cont_smpl.append(indx)
        
    img = img.detach().cpu().numpy()		
    img = np.transpose(img[0], (1, 2, 0))		
    img = img * 255		
    img = img.astype(np.uint8)
        
    contact_smpl = np.zeros((1, 1, 6890))
    contact_smpl[0][0][cont_smpl] = 1

    body_model_smpl = trimesh.load(smpl_path, process=False)
    for vert in range(body_model_smpl.visual.vertex_colors.shape[0]):
        body_model_smpl.visual.vertex_colors[vert] = mesh_colour
    body_model_smpl.visual.vertex_colors[cont_smpl] = annot_colour
                  
    mesh_out_dir = os.path.join(out_dir, 'Preds')
    os.makedirs(mesh_out_dir, exist_ok=True)          

    print(f'Saving mesh to {mesh_out_dir}')
    body_model_smpl.export(os.path.join(mesh_out_dir, 'pred.obj'))

    dummy = Dummy(os.path.join(mesh_out_dir, 'pred.obj'))
    fig = create_fig(dummy)

    return fig, os.path.join(mesh_out_dir, 'pred.obj') 

with gr.Blocks(title="DECO", css=".gradio-container") as demo:
    gr.Markdown(description)

    gr.HTML("""<h1 style="text-align:center; color:#10768c">DECO Demo</h1>""")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image", type="pil")
        with gr.Column():
            output_image = gr.Plot(label="Renders")
            output_meshes = gr.File(label="3D meshes")

    gr.HTML("""<br/>""")

    with gr.Row():
        send_btn = gr.Button("Infer")
        send_btn.click(fn=main, inputs=[input_image], outputs=[output_image, output_meshes])

    example_images = gr.Examples([
        ['/home/user/app/example_images/213.jpg'], 
        ['/home/user/app/example_images/pexels-photo-207569.webp'], 
        ['/home/user/app/example_images/pexels-photo-3622517.webp'], 
        ['/home/user/app/example_images/pexels-photo-15732209.jpeg'], 
        ], 
        inputs=[input_image])


demo.launch(debug=True) 