from tkinter import Grid
import numpy as np
import pygame as pg

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
            if obj.objectType == "axes":
                self.draw_axes(obj)
            if obj.objectType == "floor":
                self.draw_floor(obj)
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

    def draw_axes(self, axes):
        pixels = []
        for i, point in enumerate(axes.points):
            pix = self.draw_point(axes.pointColours[i], point, 2)
            pixels.append(pix)
        for i, line in enumerate(axes.lines):
            if pixels[line[0]] is not None and pixels[line[1]] is not None:
                pg.draw.aaline(self.surface, axes.lineColours[i], pixels[line[0]], pixels[line[1]])

    def draw_floor(self, floor):
        pixels = []
        for point in floor.points:
            pix = self.draw_point(floor.colour, point, 1)
            pixels.append(pix)
        for i, line in enumerate(floor.lines):
            if pixels[line[0]] is not None and pixels[line[1]] is not None:
                pg.draw.aaline(self.surface, floor.colour, pixels[line[0]], pixels[line[1]])

    
        

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
        self.objects = np.array([Axes(), FloorPlane()])

    def add_objects(self, obj):
        self.objects = np.append(self.objects, obj)

class Point():
    def __init__(self, position=np.array([0,0,0])):
        self.objectType = "point"
        self.position = position

class Axes():
    def __init__(self):
        self.objectType = "axes"
        self.points = np.array([Point(np.array([0, 0, 0])), Point(np.array([1, 0, 0])), Point(np.array([0, 1, 0])), Point(np.array([0, 0, 1]))])
        self.pointColours = np.array([(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)])
        self.lines = np.array([[0, 1], [0, 2], [0, 3]])
        self.lineColours = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)])

class FloorPlane():
    def __init__(self):
        self.objectType = "floor"
        self.points = np.array([])
        self.lines = []
        self.colour = (80, 80, 80)
        self.add_points()
    
    def add_points(self):
        grid_size = 20
        grid_spacing = 10
        for x in range(-grid_size, grid_size):
            for y in range(-grid_size, grid_size):
                self.points = np.append(self.points, Point(np.array([grid_spacing*x, grid_spacing*y, 0])))
                
                if y < 2*grid_size - 1:
                    self.lines.append([x*2*grid_size + y, x*2*grid_size + (y+1)])
                if x < 2*grid_size - 1:
                    self.lines.append([x*2*grid_size + y, (x+1)*2*grid_size + y])










