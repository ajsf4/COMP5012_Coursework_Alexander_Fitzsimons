import numpy as np
import pygame as pg

pg.init()

class Shader():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pg.Surface((width, height))
        self.camera = Camera(np.array([-10, 0, 0], dtype=np.float64), np.array([0, 0], dtype=np.float64), 90, self.width, self.height)

        self.scene = Scene()

    def rasterize(self):
        self.surface.fill((0,0,0))
        for obj in self.scene.objects:
            if obj.objectType == "point":
                self.draw_point((255, 255,255), obj)
            if obj.objectType == "axes":
                self.draw_axes(obj)
        return self.surface
        
    def draw_point(self, colour, point):
        v = point.position - self.camera.position
        pix_X = self.width/2 - self.camera.d * np.tan(np.atan(v[1]/v[0]) - self.camera.orientation[0])
        pix_Y = self.height/2 - self.camera.d * np.tan(np.atan(v[2]/np.sqrt(v[0]**2 + v[1]**2)) - self.camera.orientation[1])
        pg.draw.circle(self.surface, colour, (pix_X, pix_Y), 3)
        return np.array([pix_X, pix_Y])

    def draw_axes(self, axes):
        pixels = []
        for i, point in enumerate(axes.points):
            pixels.append(self.draw_point(axes.pointColours[i], point))
        for i, line in enumerate(axes.lines):
            pg.draw.aaline(self.surface, axes.lineColours[i], pixels[line[0]], pixels[line[1]])
    
        

class Camera():
    def __init__(self, position, orientation, fov, width, height):
        self.position = position
        self.orientation = orientation
        self.vfov = fov
        self.hfov = fov * width/height
        self.d = width / (2 * np.tan(self.hfov/2))
        

    def transform(self, translation, rotation):
        self.position += translation
        self.orientation += rotation

class Scene():
    def __init__(self):
        self.objects = np.array([Axes()])

    def add_objects(self, obj):
        self.objects = np.append(self.objects, obj)

class Point():
    def __init__(self, position=np.array([0,0,0])):
        self.objectType = "point"
        self.position = position

class Composite():
    def __init__(self):
        self.objectType = "composite"
        self.objects = np.array([])
    def add_objects(self, obj):
        self.objects = np.append(self.objects, obj)

class Axes():
    def __init__(self):
        self.objectType = "axes"
        self.points = np.array([Point(np.array([0, 0, 0])), Point(np.array([1, 0, 0])), Point(np.array([0, 1, 0])), Point(np.array([0, 0, 1]))])
        self.pointColours = np.array([(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
        self.lines = np.array([[0, 1], [0, 2], [0, 3]])
        self.lineColours = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)])








