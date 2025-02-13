import numpy as np
import pygame as pg
import sys
import shader as sh

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

running = True
while running:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    print("Rendering")
    screen.blit(shader.rasterize(), (0, 0))
    pg.display.flip()
    clock.tick(30)

sys.exit()