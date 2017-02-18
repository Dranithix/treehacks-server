from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit

import base64

import sys
import os

os.environ["THEANO_FLAGS"] = "device=gpu0, floatX=float32, nvcc.fastmath=True, cuda.root=/usr/local/cuda"

import model_def.dcgan_theano as TheanoDCGAN
import constrained_opt_theano as TheanoProp
import constrained_opt
import cv2
import numpy as np

app = Flask(__name__)
sio = SocketIO(app, logger=False, engineio_logger=False, binary=True)

opt_engine = None
npx = None
im_color = None
im_color_mask = None
im_edge = None


@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index.html')


@sio.on('connect', namespace='/data')
def connect():
    pass


@sio.on('draw', namespace='/data')
def message(data, final):
    global opt_engine, npx, im_color, im_color_mask, im_edge

    with open("imageToSave.png", "wb") as fh:
        print(base64.decodestring(data))
        fh.write(base64.decodestring(data))

    constraints = [im_color, im_color_mask[..., [0]], im_edge, im_edge[..., [0]]]

    if final:
        for x in range(5): opt_engine.update_invert(constraints=constraints)
    else:
        opt_engine.update_invert(constraints=constraints)

    result = np.concatenate(opt_engine.get_current_results(), 1)
    send(cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 50])[1].tostring())
    # emit('preview', base64.encodestring(cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]))
    # results = opt_engine.get_current_results()
    #
    # final_result = np.concatenate(results, 1)
    # # final_vis = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
    #
    # emit('preview', base64.encodestring(cv2.imencode('.png', final_result)[1]))


@sio.on('disconnect', namespace='/data')
def disconnect():
    pass


def preprocess_image(img_path, npx):
    im = cv2.imread(img_path, 1)
    if im.shape[0] != npx or im.shape[1] != npx:
        out = cv2.resize(im, (npx, npx))
    else:
        out = np.copy(im)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def load_model():
    global opt_engine, npx, im_color, im_color_mask, im_edge
    model = TheanoDCGAN.Model(model_name="shoes_64", model_file="./models/shoes_64.dcgan_theano")

    opt_solver = TheanoProp.OPT_Solver(model, batch_size=64, d_weight=0.0)
    opt_engine = constrained_opt.Constrained_OPT(opt_solver, batch_size=64, n_iters=100, topK=16)

    npx = model.npx

    im_color = preprocess_image('./pics/input_color.png', npx)
    im_color_mask = preprocess_image('./pics/input_color_mask.png', npx)
    im_edge = preprocess_image('./pics/input_edge.png', npx)

    print(npx)

    print(im_color, im_color_mask, im_edge)

    # run the optimization
    opt_engine.init_z()


if __name__ == '__main__':
    load_model()
    sio.run(app)
