from attr import s
import numpy as np
import pygame as pg
import sys

import shader as sh
import controller as ct

pg.init()

clock = pg.time.Clock()

# Screen
width, height = 800, 600

screen = pg.display.set_mode((width, height))

pg.display.set_caption("Travelling Salesman Problem")

# Shader
print("Creating shader")
shader = sh.Shader(width, height)
shader.scene.add_objects([sh.Point(np.array([0, 0, 0])), sh.Point(np.array([1, 0, 0])), sh.Point([0, 1, 0]), sh.Point([0, 0, 1])])
print("Shader created")

camControl = ct.controller()

running = True
dt = 0
while running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_e:
                camControl.translation[2] = 1
            if event.key == pg.K_q:
                camControl.translation[2] = -1
            if event.key == pg.K_a:
                camControl.translation[0] = -1
            if event.key == pg.K_d:
                camControl.translation[0] = 1
            if event.key == pg.K_w:
                camControl.translation[1] = 1
            if event.key == pg.K_s: 
                camControl.translation[1] = -1
            if event.key == pg.K_UP:
                camControl.rotation[1] = 1
            if event.key == pg.K_DOWN:
                camControl.rotation[1] = -1
            if event.key == pg.K_LEFT:
                camControl.rotation[0] = -1
            if event.key == pg.K_RIGHT:
                camControl.rotation[0] = 1
        if event.type == pg.KEYUP:
            if event.key == pg.K_e:
                camControl.translation[2] = 0
            if event.key == pg.K_q:
                camControl.translation[2] = 0
            if event.key == pg.K_a:
                camControl.translation[0] = 0
            if event.key == pg.K_d:
                camControl.translation[0] = 0
            if event.key == pg.K_w:
                camControl.translation[1] = 0
            if event.key == pg.K_s: 
                camControl.translation[1] = 0
            if event.key == pg.K_UP:
                camControl.rotation[1] = 0
            if event.key == pg.K_DOWN:
                camControl.rotation[1] = 0
            if event.key == pg.K_LEFT:
                camControl.rotation[0] = 0
            if event.key == pg.K_RIGHT:
                camControl.rotation[0] = 0
    
    camControl.transform(shader.camera, 1, 0.1, dt)

    print("Rendering")
    screen.blit(shader.rasterize(), (0, 0))
    pg.display.flip()
    dt = clock.tick(30) / 1000

sys.exit()