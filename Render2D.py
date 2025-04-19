import pygame as pg
import numpy as np

class Camera2D:
    def __init__(self, resolution):
        self.position = np.array([0, 0], dtype=np.float64)
        self.original_resolution = np.array(resolution)
        self.surface = pg.Surface(resolution)
        self.scene = []

    def add_to_scene(self, obj):
        self.scene.append(obj)

    def render(self):
        self.surface.fill((0, 0, 0))
        
        for obj in self.scene:
            self.surface.blit(obj.surface, obj.position-self.position)


class MapObj:
    def __init__(self, position, size, colour, points, connections):
        self.rescale = 4
        self.position = position
        self.colour = colour
        self.surface = pg.Surface(self.rescale*size)
        self.points = self.rescale * points
        
    def update(self, new_connections):
        self.surface.fill((0, 0, 0))
       
        for i, c in enumerate(new_connections):
            pg.draw.aaline(self.surface, self.colour, self.points[new_connections[i-1]], self.points[c], 2)
            pg.draw.circle(self.surface, self.colour, self.points[c], 5)


class GraphObj:
    def __init__(self, position, initial_data, size, title, x_label, y_label):
        self.position = position
        self.data = initial_data
        self.size = size
        self.surface = pg.Surface(size)
        self.s = 5 # spacing

        self.minX = min(initial_data[0])
        self.maxX = max(initial_data[0])
        self.minY = min(initial_data[1])
        self.maxY = max(initial_data[1])
        self.rangeX = self.maxX - self.minX
        self.rangeY = self.maxY - self.minY

        self.axes_colour = (100, 200, 0)
        self.datapoint_colour = (255, 0, 0)
        self.text_colour = (0, 255, 0)

        self.font = pg.font.SysFont("consolas", 10)
        self.t = title
        self.title = self.font.render(title, True, self.text_colour)
        self.x_label = self.font.render(x_label, True, self.text_colour)
        self.y_label = pg.transform.rotate(self.font.render(y_label, True, self.text_colour), 90)

        self.pareto_front = []

    def update(self, new_data, pareto_front):
        self.pareto_front = pareto_front.copy()
        self.data = new_data.copy()
        self.minX = min(self.data[0])
        self.maxX = max(self.data[0])
        self.minY = min(self.data[1])
        self.maxY = max(self.data[1])
        self.rangeX = self.maxX - self.minX
        self.rangeY = self.maxY - self.minY

    def draw_surface(self):
        self.surface.fill((0,0,0))
        
        p1 = np.array([self.s*3, self.s*3])
        p2 = np.array([self.s*3, self.size[1]-self.s*3])
        p3 = np.array([self.size[0]-self.s*3, self.size[1]-self.s*3])

        graph_width = p3[0] - p1[0]
        graph_height = p3[1] - p1[1]

        pg.draw.line(self.surface, self.axes_colour, p1, p2)
        pg.draw.line(self.surface, self.axes_colour, p2, p3)

        self.surface.blit(self.title, (self.size[0]//2 - self.title.get_width()//2, 0))
        self.surface.blit(self.x_label, (self.size[0]//2 - self.x_label.get_width()//2, self.size[1]-self.x_label.get_height()))
        self.surface.blit(self.y_label, (0, self.size[1]//2 - self.y_label.get_height()//2))

        # draw the scale on the axes
        minx_surf = self.font.render(f"{self.minX:.2f}", True, self.text_colour)
        miny_surf = pg.transform.rotate(self.font.render(f"{self.minY:.2f}", True, self.text_colour), 90)
        maxx_surf = self.font.render(str(f"{self.maxX:.2f}"), True, self.text_colour)
        maxy_surf = pg.transform.rotate(self.font.render(f"{self.maxY:.2f}", True, self.text_colour), 90)

        self.surface.blit(minx_surf, (self.s*3, self.size[1]-self.s*3))
        self.surface.blit(miny_surf, (self.s*3 - miny_surf.get_width(), self.size[1]-self.s*3-miny_surf.get_height()))
        self.surface.blit(maxx_surf, (self.size[0]-self.s*3 - maxx_surf.get_width(), self.size[1]-self.s*3))
        self.surface.blit(maxy_surf, (self.s*3 - maxy_surf.get_width(), self.s*3))

        # draw the data points
        if len(self.data[0]) >= 2 and self.rangeX != 0 and self.rangeY != 0:
            for i, x, y in zip(list(range(len(self.data[0]))), self.data[0], self.data[1]):
                plot_x = p2[0] + (x-self.minX) * graph_width / self.rangeX
                plot_y = p2[1] - (y-self.minY) * graph_height / self.rangeY
                if i in self.pareto_front:
                    pg.draw.circle(self.surface, (0, 0, 255), (int(plot_x), int(plot_y)), 2)
                else:
                    pg.draw.circle(self.surface, self.datapoint_colour, (int(plot_x), int(plot_y)), 2)