import numpy as np
import pygame as pg

pg.init()

class Shader():
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pg.Surface((width, height))
        self.camera = Camera(np.array([-10, 0, 0], dtype=np.float64), np.array([0, 0]), 90, self.width, self.height)

        self.scene = Scene()

    def rasterize(self):
        for col in range(self.width):
            for row in range(self.height):
                colour = (255,255,255) if self.cast_ray(np.array([col, row])) else (0,0,0)
                self.surface.set_at((col, row), colour)
        return self.surface
                        
    def cast_ray(self, pixel):
        local_phi = np.arctan(np.tan(self.camera.hfov/2)*(pixel[0]/self.width - 1/2))
        local_theta = np.arctan(np.tan(self.camera.vfov/2)*(pixel[1]/self.height - 1/2))

        global_phi = self.camera.orientation[0] + local_phi
        global_theta = self.camera.orientation[1] + local_theta

        ray = np.linalg.norm(np.array([np.cos(global_phi)*np.cos(global_theta),
                       np.sin(global_phi)*np.cos(global_theta), np.sin(global_theta)]))

        hit_bool = False

        limit = 15
        test_point = self.camera.position
        for i in range(limit):
            dist, obj_index = self.dist_closest_object(test_point)
            if dist < 0.01:
                hit_bool = True
                break
            else:
                test_point += ray * dist

        return hit_bool, obj_index

    def dist_closest_object(self, test_point):
        distances = np.ones(len(self.scene.objects)) * np.inf
        for i, obj in enumerate(self.scene.objects):
            if obj.objectType == "point":
                distances[i] = np.linalg.norm(obj.position - test_point)
        return np.min(distances), np.argmin(distances)    
        

class Camera():
    def __init__(self, position, orientation, fov, width, height):
        self.position = position
        self.orientation = orientation
        self.vfov = fov
        self.hfov = fov * width/height
        

    def transform(self, translation, rotation):
        self.position += translation
        self.orientation += rotation

class Scene():
    def __init__(self):
        self.objects = np.array([])

    def add_objects(self, obj):
        self.objects = np.append(self.objects, obj)

class Point():
    def __init__(self, position=np.array([0,0,0]), objectType="point"):
        self.objectType = objectType
        self.position = position





