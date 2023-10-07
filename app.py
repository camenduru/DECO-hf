import torch
import os
import glob
import numpy as np
import cv2
import PIL.Image as pil_img
import sys
import gradio as gr

import trimesh
import pyrender

print(os.path.abspath(__file__))
os.system('pip install /home/user/app/vendor/pyrender')
sys.path.append('/home/user/app/vendor/pyrender')
os.system('pip install gradio==3.47.1')
os.system('sh fetch_data.sh')

from models.deco import DECO
from common import constants

os.environ['PYOPENGL_PLATFORM'] = 'egl'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def initiate_model(model_path):
    deco_model = DECO('hrnet', True, device)

    print(f'Loading weights from {model_path}')
    checkpoint = torch.load(model_path)
    deco_model.load_state_dict(checkpoint['deco'], strict=True)

    deco_model.eval()

    return deco_model

def render_image(scene, img_res, img=None, viewer=False):
    '''
    Render the given pyrender scene and return the image. Can also overlay the mesh on an image.
    '''
    if viewer:
        pyrender.Viewer(scene, use_raymond_lighting=True)
        return 0
    else:
        r = pyrender.OffscreenRenderer(viewport_width=img_res,
                                       viewport_height=img_res,
                                       point_size=1.0)
        color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if img is not None:
            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img.detach().cpu().numpy()
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)
        else:
            output_img = color
        return output_img

def create_scene(mesh, img, focal_length=500, camera_center=250, img_res=500):
    # Setup the scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=(0.3, 0.3, 0.3))
    # add mesh for camera
    camera_pose = np.eye(4)
    camera_rotation = np.eye(3, 3)
    camera_translation = np.array([0., 0, 2.5])
    camera_pose[:3, :3] = camera_rotation
    camera_pose[:3, 3] = camera_rotation @ camera_translation
    pyrencamera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=camera_center, cy=camera_center)
    scene.add(pyrencamera, pose=camera_pose)
    # create and add light
    light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)
    light_pose = np.eye(4)
    for lp in [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]]:
        light_pose[:3, 3] = mesh.vertices.mean(0) + np.array(lp)
        # out_mesh.vertices.mean(0) + np.array(lp)
        scene.add(light, pose=light_pose)
    # add body mesh
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 1.0, 0.9, 1.0))
    mesh_images = []

    # resize input image to fit the mesh image height
    img_height = img_res
    img_width = int(img_height * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (img_width, img_height))
    mesh_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for sideview_angle in [0, 90, 180, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(
            np.radians(sideview_angle), [0, 1, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        output_img = render_image(scene, img_res)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

    # show upside down view
    for topview_angle in [90, 270]:
        out_mesh = mesh.copy()
        rot = trimesh.transformations.rotation_matrix(
            np.radians(topview_angle), [1, 0, 0])
        out_mesh.apply_transform(rot)
        out_mesh = pyrender.Mesh.from_trimesh(
            out_mesh,
            material=material)
        mesh_pose = np.eye(4)
        scene.add(out_mesh, pose=mesh_pose, name='mesh')
        output_img = render_image(scene, img_res)
        output_img = pil_img.fromarray((output_img * 255).astype(np.uint8))
        output_img = np.asarray(output_img)[:, :, :3]
        mesh_images.append(output_img)
        # delete the previous mesh
        prev_mesh = scene.get_nodes(name='mesh').pop()
        scene.remove_node(prev_mesh)

    # stack images
    IMG = np.hstack(mesh_images)
    IMG = pil_img.fromarray(IMG)
    IMG.thumbnail((3000, 3000))
    return IMG    

def main(args, img_src, out_dir, mesh_colour=[130, 130, 130, 255], annot_colour=[0, 255, 0, 255]):
    if os.path.isdir(img_src):
        images = glob.iglob(img_src + '/*', recursive=True)
    else:
        images = [img_src]

    deco_model = initiate_model(args)
    
    smpl_path = os.path.join(constants.SMPL_MODEL_DIR, 'smpl_neutral_tpose.ply')
    
    for img_name in images:
        img = cv2.imread(img_name)
        img = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
        img = img.transpose(2,0,1)/255.0
        img = img[np.newaxis,:,:,:]
        img = torch.tensor(img, dtype = torch.float32).to(device)

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

        rend = create_scene(body_model_smpl, img)
        os.makedirs(os.path.join(out_dir, 'Renders'), exist_ok=True) 
        rend.save(os.path.join(out_dir, 'Renders', os.path.basename(img_name).split('.')[0] + '.png'))
                  
        mesh_out_dir = os.path.join(out_dir, 'Preds', os.path.basename(img_name).split('.')[0])
        os.makedirs(mesh_out_dir, exist_ok=True)          

        print(f'Saving mesh to {mesh_out_dir}')
        body_model_smpl.export(os.path.join(mesh_out_dir, 'pred.obj'))

    return out_dir   

with gr.Blocks(title="DECO", css=".gradio-container") as demo:

    gr.HTML("""<div style="font-weight:bold; text-align:center; color:royalblue;">DECO</div>""")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input image", type="pil")
        with gr.Column():
            output_image = gr.Image(label="Renders", type="pil")
            output_meshes = gr.File(label="3D meshes")

    gr.HTML("""<br/>""")

    # with gr.Row():
    #     threshold = gr.Slider(0, 1.0, value=0.6, label='Detection Threshold')
    #     send_btn = gr.Button("Infer")
    #     send_btn.click(fn=main, inputs=[input_image, threshold], outputs=[output_image, output_meshes])

    # example_images = gr.Examples([
    #     ['/home/user/app/assets/test1.png'], 
    #     ['/home/user/app/assets/test2.jpg'], 
    #     ['/home/user/app/assets/test3.jpg'], 
    #     ['/home/user/app/assets/test4.jpg'], 
    #     ['/home/user/app/assets/test5.jpg'], 
    #     ], 
    #     inputs=[input_image, 0.6])


#demo.queue()
demo.launch(debug=True) 