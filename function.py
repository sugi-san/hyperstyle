def generate_mp4(out_name, images, kwargs):
    writer = imageio.get_writer(out_name + '.mp4', **kwargs)
    for image in images:
        writer.append_data(image)
    writer.close()


def get_latent_and_weight_deltas(inputs, net, opts):
    opts.resize_outputs = False
    opts.n_iters_per_batch = 5
    with torch.no_grad():
        _, latent, weights_deltas, _ = run_inversion(inputs.to("cuda").float(), net, opts)
    weights_deltas = [w[0] if w is not None else None for w in weights_deltas]
    return latent, weights_deltas
    

def get_result_from_vecs(vectors_a, vectors_b, weights_deltas_a, weights_deltas_b, alpha):
    results = []
    for i in range(len(vectors_a)):
        with torch.no_grad():
            cur_vec = vectors_b[i] * alpha + vectors_a[i] * (1 - alpha)
            cur_weight_deltas = interpolate_weight_deltas(weights_deltas_a, weights_deltas_b, alpha)
            res = net.decoder([cur_vec],
                              weights_deltas=cur_weight_deltas,
                              randomize_noise=False,
                              input_is_latent=True)[0]
            results.append(res[0])
    return results

def interpolate_weight_deltas(weights_deltas_a, weights_deltas_b, alpha):
    cur_weight_deltas = []
    for weight_idx, w in enumerate(weights_deltas_a):
        if w is not None:
            delta = weights_deltas_b[weight_idx] * alpha + weights_deltas_a[weight_idx] * (1 - alpha)
        else:
            delta = None
        cur_weight_deltas.append(delta)
    return cur_weight_deltas
    
def show_mp4(filename, width):
    mp4 = open(filename + '.mp4', 'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    display(HTML("""
    <video width="%d" controls autoplay loop>
        <source src="%s" type="video/mp4">
    </video>
    """ % (width, data_url)))



# --- display_pic ---
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def display_pic(folder):
    fig = plt.figure(figsize=(30, 60))
    files = os.listdir(folder)
    files.sort()
    for i, file in enumerate(files):
        if file=='.ipynb_checkpoints':
           continue
        if file=='.DS_Store':
           continue
        img = Image.open(folder+'/'+file)    
        images = np.asarray(img)
        ax = fig.add_subplot(10, 5, i+1, xticks=[], yticks=[])
        image_plt = np.array(images)
        ax.imshow(image_plt)
        #name = os.path.splitext(file)
        ax.set_xlabel(file, fontsize=30)               
    plt.show()
    plt.close()


# --- reset_folder ---
import shutil

def reset_folder(path):
    if os.path.isdir(path):
      shutil.rmtree(path)
    os.makedirs(path,exist_ok=True)
