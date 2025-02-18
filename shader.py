import numpy as np
import pygame as pg
import shapes as sp

pg.init()

class Shader():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pg.Surface((width, height))
        self.camera = Camera(np.array([35, -60, 16], dtype=np.float64), np.array([1.5, 0, 0.05], dtype=np.float64), 0.5, 2, self.width, self.height)
        self.scene = Scene()

    def rasterize(self):
        self.surface.fill((0,0,0))
        for obj in self.scene.objects:
            if obj.objectType == "point":
                self.draw_point((255, 255,255), obj, 3)
            else:
                self.draw_shape(obj)
        return self.surface
        
    def draw_point(self, colour, point, point_size):
        th_x = self.camera.orientation[0]
        th_y = self.camera.orientation[1]
        th_z = self.camera.orientation[2]

        Mx = np.array([[1,            0,            0],
                       [0, np.cos(th_x), np.sin(th_x)],
                       [0,-np.sin(th_x), np.cos(th_x)]])
        My = np.array([[np.cos(th_y), 0, -np.sin(th_y)],
                       [0,            1,            0],
                       [np.sin(th_y), 0, np.cos(th_y)]])
        Mz = np.array([[np.cos(th_z), np.sin(th_z), 0],
                       [-np.sin(th_z),np.cos(th_z), 0],
                       [0,            0,            1]])
        d = np.matmul(np.matmul(np.matmul(Mx, My), Mz), (point.position - self.camera.position))
        if d[2] >= 0:
            return None
        bx = d[0]*self.width/(d[2]*self.camera.r_x * self.camera.r_z) + self.width/2
        by = d[1]*self.height/(d[2]*self.camera.r_y * self.camera.r_z) + self.height/2
        pg.draw.circle(self.surface, colour, (bx, by), point_size)
        return np.array([bx, by])

    def draw_shape(self, shape):
        pixels = []
        for i, point in enumerate(shape.points):
            pix = self.draw_point(shape.pointColours[i], point, 2)
            pixels.append(pix)
        for i, line in enumerate(shape.lines):
            if pixels[line[0]] is not None and pixels[line[1]] is not None:
                pg.draw.aaline(self.surface, shape.lineColours[i], pixels[line[0]], pixels[line[1]])        

class Camera():
    def __init__(self, position, orientation, r_z, r_x, width, height):
        self.position = position
        self.orientation = orientation
        self.r_z = r_z
        self.ratio = width/height
        self.r_x = r_x
        self.r_y = r_x/self.ratio

    def transform(self, translation, rotation):
        self.position += translation
        self.orientation += rotation

class Scene():
    def __init__(self):
        self.objects = np.array([sp.Axes(), sp.FloorPlane()])

    def add_objects(self, obj):
        self.objects = np.append(self.objects, obj)

    def edit_objects(self, obj, index):
        self.objects[index] = obj












