import base64
import os
import time
from threading import Thread

from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_socketio import send, emit

current_milli_time = lambda: int(round(time.time() * 1000))

os.environ["THEANO_FLAGS"] = "device=gpu0, floatX=float32, nvcc.fastmath=True, cuda.root=/usr/local/cuda"

import model_def.dcgan_theano as TheanoDCGAN
import constrained_opt_theano as TheanoProp
import constrained_opt
import cv2
import numpy as np

app = Flask(__name__)
sio = SocketIO(app, logger=False, engineio_logger=False)

constraints = None
opt_engine = None
npx = None
im_color = None
im_color_mask = None
im_edge = None

iterationsPassed = 0
iterationLimit = 40


@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index.html')


@sio.on('connect', namespace='/data')
def connect():
    pass


@sio.on('resetCanvas', namespace='/data')
def resetCanvas():
    global opt_engine, iterationsPassed, im_color_mask, im_color, im_edge, constraints
    opt_engine.init_z()
    iterationsPassed = 0
    im_color = np.zeros((npx, npx, 3), np.uint8)
    im_color_mask = np.zeros((npx, npx, 3), np.uint8)
    im_edge = np.zeros((npx, npx, 3), np.uint8)

    constraints = [im_color, im_color_mask[..., [0]], im_edge, im_edge[..., [0]]]


@sio.on('draw', namespace='/data')
def message(edges, color):
    global opt_engine, iterationsPassed, npx, im_color, im_color_mask, im_edge, constraints

    if edges is not None:
        edgeBuffer = cv2.imdecode(np.asarray(bytearray(base64.decodestring(edges)), dtype='uint8'), 1)[:, :, 0:3]
        im_edge = edgeBuffer
        iterationsPassed = 0

    if color is not None:
        colorBuffer = cv2.imdecode(np.asarray(bytearray(base64.decodestring(color)), dtype='uint8'), 1)[:, :, 0:3]
        im_color = colorBuffer
        im_color_mask = cv2.cvtColor(cv2.cvtColor(colorBuffer, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

    constraints = [im_color, im_color_mask[..., [0]], im_edge, im_edge[..., [0]]]
    result = opt_engine.get_current_results()[0]
    emit('frame', base64.encodestring(cv2.imencode('.jpg', result)[1]))


@sio.on('disconnect', namespace='/data')
def disconnect():
    pass


def preprocess_image(im):
    global npx

    if im.shape[0] != npx or im.shape[1] != npx:
        out = cv2.resize(im, (npx, npx))
    else:
        out = np.copy(im)

    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out


def optimize_thread():
    global opt_engine, iterationsPassed, constraints
    while True:
        if iterationsPassed < iterationLimit:
            opt_engine.update_invert(constraints=constraints)
            iterationsPassed += 1


def load_model():
    global opt_engine, npx, im_color, im_color_mask, im_edge, constraints
    model = TheanoDCGAN.Model(model_name="handbag_64", model_file="./models/handbag_64.dcgan_theano")

    opt_solver = TheanoProp.OPT_Solver(model, batch_size=64, d_weight=0.0)
    opt_engine = constrained_opt.Constrained_OPT(opt_solver, batch_size=64, n_iters=iterationLimit, topK=16)

    npx = model.npx

    im_color = np.zeros((npx, npx, 3), np.uint8)
    im_color_mask = np.zeros((npx, npx, 3), np.uint8)
    im_edge = np.zeros((npx, npx, 3), np.uint8)

    constraints = [im_color, im_color_mask[..., [0]], im_edge, im_edge[..., [0]]]

    # run the optimization
    opt_engine.init_z()

    thread = Thread(target=optimize_thread, args=())
    thread.start()


if __name__ == '__main__':
    load_model()
    sio.run(app)
