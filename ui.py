from gettext import find
import pygame as pg
import numpy as np
pg.init()

class Graph():
    def __init__(self, size, graph_type = "line"):
        self.size = size
        self.graph_type = graph_type
        self.surface = pg.Surface(size).convert_alpha()
        self.surface.fill((100,100,100))
        self.surface.set_alpha(168)
        self.draw_colour = (0, 255, 0)
        self.font = pg.font.SysFont("consolas", 12)
        self.update_ui(np.array([[0, 1], [0, 1]]), "x", "y")

    
    def update_ui(self, data, x_label, y_label, pareto_front=[]):
        self.surface.fill((100,100,100))
        self.surface.set_alpha(168)
        self.x_text = self.font.render(x_label, True, self.draw_colour)
        self.y_text = pg.transform.rotate(self.font.render(y_label, True, self.draw_colour), 90)
        self.surface.blit(self.x_text, (self.size[0]/2 - self.x_text.get_width()/2, self.size[1] - self.x_text.get_height()))
        self.surface.blit(self.y_text, (0, self.size[1]/2 - self.y_text.get_height()/2))

        p1 = np.array((self.y_text.get_width()*2, 5))
        p2 = np.array((self.y_text.get_width()*2, self.size[1] - self.x_text.get_height()*2.5))
        p3 = np.array((self.size[0] - 5         , self.size[1] - self.x_text.get_height()*2.5))
        pg.draw.aalines(self.surface, (self.draw_colour), False, (p1, p2, p3))

        if (len(data[0]) > 1):
            scale_x = (p3[0] - p2[0]) / (max(data[0]) - min(data[0]))
            scale_y = (p2[1] - p1[1]) / (max(data[1]) - min(data[1]))

            fitted_x = p2[0] + scale_x * (data[0]-min(data[0]))
            fitted_y = p2[1] - scale_y * (data[1]-min(data[1]))
            fitted_data = np.array([fitted_x, fitted_y])

            if self.graph_type == "line":
                pg.draw.aalines(self.surface, self.draw_colour, False, fitted_data.transpose())
            elif self.graph_type == "scatter":
                for i, point in enumerate(fitted_data.transpose()):
                    if i in pareto_front:
                        pg.draw.circle(self.surface, (0,0,255), point, 2)
                    else:
                        pg.draw.circle(self.surface, (0,255,0), point, 2)

            min_x_tx = self.font.render(f"{min(data[0]):.2e}", True, self.draw_colour)
            max_x_tx = self.font.render(f"{max(data[0]):.2e}", True, self.draw_colour)
            min_y_tx = pg.transform.rotate(self.font.render(f"{float(min(data[1])):.1f}", True, self.draw_colour), 90)
            max_y_tx = pg.transform.rotate(self.font.render(f"{float(max(data[1])):.1f}", True, self.draw_colour), 90)

            self.surface.blit(min_x_tx, p2 - np.array([0, -5]))
            self.surface.blit(max_x_tx, p3 - np.array([max_x_tx.get_width(), -5]))
            self.surface.blit(min_y_tx, p2 - np.array([min_y_tx.get_width(), min_y_tx.get_height()]))
            self.surface.blit(max_y_tx, p1 - np.array([max_y_tx.get_width(), 0]))










