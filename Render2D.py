# installed module imports
import pygame as pg
import numpy as np

# camera control: collectes objects in scene and draws the to a surface to be blitted to the screen
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

# map object displays the objects in R^2 euclidean space
class MapObj:
    def __init__(self, position, size, colour, points, connections, title):
        self.rescale = 2.5
        self.size = size
        self.position = np.array(position)
        self.colour = np.array(colour)
        self.surface = pg.Surface(self.rescale*size)
        self.points = self.rescale * points
        self.font = pg.font.SysFont("Consolas", 12)
        self.text = self.font.render(title, True, self.colour)
        
    def update(self, new_connections):
        self.surface.fill((0, 0, 0))

        for point in self.points:
            pg.draw.circle(self.surface, (self.colour*0.5).astype(int), point, 4)

       
        for i, c in enumerate(new_connections):
            pg.draw.aaline(self.surface, self.colour, self.points[new_connections[i-1]], self.points[c], 2)
            pg.draw.circle(self.surface, self.colour, self.points[c], 4)

        pg.draw.circle(self.surface, (255,255,255), self.points[0], 2)

        self.surface.blit(self.text, (0, self.size[1]-self.text.get_height()))

# Graph displays data provided to it, in this case it is objective space data
class GraphObj:
    def __init__(self, position, initial_data, size, title, x_label, y_label):
        self.position = position
        self.data = initial_data
        self.new_data_points = [i for i in range(len(self.data[0]))]
        self.highlight_population = [i for i in range(len(self.data[0]))]
        self.size = np.array(size)
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

        self.font = pg.font.SysFont("consolas", 14)
        self.t = title
        self.title = self.font.render(title, True, self.text_colour)
        self.x_label = self.font.render(x_label, True, self.text_colour)
        self.y_label = pg.transform.rotate(self.font.render(y_label, True, self.text_colour), 90)

        self.pareto_front = []
        self.sorted_pareto_front = []

        self.p1 = np.array([self.s*3, self.s*3])
        self.p2 = np.array([self.s*3, self.size[1]-self.s*3])
        self.p3 = np.array([self.size[0]-self.s*3, self.size[1]-self.s*3])
        self.p4 = np.array([self.size[0]-self.s*3, self.s*3])

        self.graph_width = self.p3[0] - self.p1[0]
        self.graph_height = self.p3[1] - self.p1[1]

        self.label_surface = pg.Surface((self.size[0]/2.5, self.size[1]/3))

        self.ref_dist = 50000
        self.ref_dmnd = 800
        self.n_hypervolume_indicator = 0
        self.max_volume = 0

    def sort_pareto_front(self):
        pareto_distances = [self.data[0][i] for i in self.pareto_front]
        sorted_pareto_indices = np.flip(np.argsort(pareto_distances))       
        self.sorted_pareto_front = [self.pareto_front[i] for i in sorted_pareto_indices]

    def update(self, new_data, pareto_front):
        self.pareto_front = pareto_front.copy()
        number_of_new_points = len(new_data[0]) - len(self.data[0])
        self.new_data_points = [i+len(self.data[0]) for i in range(number_of_new_points)]
        self.data = new_data.copy()
        self.sort_pareto_front()

        self.minX = min(self.data[0])
        self.maxX = max(self.data[0])
        self.minY = min(self.data[1])
        self.maxY = max(self.data[1])
        self.rangeX = self.maxX - self.minX
        self.rangeY = self.maxY - self.minY

        self.calculate_hypervolume()

        label_heading = self.font.render("Key:", True, (0,200,0))
        label_H_Indicator1 = self.font.render("Normalised Hypervolume ", True, (0,200,0))
        label_H_Indicator2 = self.font.render(f"Indicator = {self.n_hypervolume_indicator:.4f}", True, (0,200,0))
        label_Volume = self.font.render(f"Max Volume = {self.max_volume:.4f}", True, (0, 200, 0))

        label_red_key = self.font.render(" = Historical Solution", True, (0, 200, 0))
        label_yellow_key = self.font.render(" = Just Added", True, (0, 200, 0))
        label_cyan_key = self.font.render(" = Next Generation", True, (0, 200, 0))
        label_blue_key = self.font.render(" = Pareto Front", True, (0, 200, 0))

        self.label_surface.fill((10, 20, 10))
        pg.draw.rect(self.label_surface, (0, 200, 0), (0, 0, self.label_surface.get_width(), self.label_surface.get_height()), width=2)
        self.label_surface.blit(label_heading, (5,5))
        self.label_surface.blit(label_red_key, (10,25))
        pg.draw.circle(self.label_surface, (255, 0, 0), (10, 32), 3)
        self.label_surface.blit(label_yellow_key, (10,45))
        pg.draw.circle(self.label_surface, (255, 255, 0), (10, 52), 3)
        self.label_surface.blit(label_cyan_key, (10,65))
        pg.draw.circle(self.label_surface, (0, 200, 250), (10, 72), 3)
        self.label_surface.blit(label_blue_key, (10,85))
        pg.draw.circle(self.label_surface, (0, 0, 255), (10, 92), 2)

        self.label_surface.blit(label_H_Indicator1, (5, 110))
        self.label_surface.blit(label_H_Indicator2, (5, 125))

        self.label_surface.blit(label_Volume, (5, 145))

    def draw_surface(self):
        self.surface.fill((0,0,0))
       

        pg.draw.line(self.surface, self.axes_colour, self.p1, self.p2)
        pg.draw.line(self.surface, self.axes_colour, self.p2, self.p3)

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
                plot_x, plot_y = self.plot_xy(x,y)
                if i in self.new_data_points:
                    pg.draw.circle(self.surface, (255, 255, 0), (int(plot_x), int(plot_y)), 3)
                else:
                    if i > len(self.data[0])-1000:
                        pg.draw.circle(self.surface, self.datapoint_colour, (int(plot_x), int(plot_y)), 3)
            
            for i, x, y in zip(list(range(len(self.data[0]))), self.data[0], self.data[1]):                
                if i in self.highlight_population:
                    plot_x, plot_y = self.plot_xy(x,y)
                    pg.draw.circle(self.surface, (0, 200, 250), (int(plot_x), int(plot_y)), 3)

            for i, x, y in zip(list(range(len(self.data[0]))), self.data[0], self.data[1]):
                if i in self.pareto_front:
                    plot_x, plot_y = self.plot_xy(x,y)
                    pg.draw.circle(self.surface, (0, 0, 255), (int(plot_x), int(plot_y)), 2)

        self.surface.blit(self.label_surface, (self.size[0] - self.label_surface.get_width(), 0))

    def select_solution(self, selected_solution):
        x = self.data[0][self.sorted_pareto_front[selected_solution]]
        y = self.data[1][self.sorted_pareto_front[selected_solution]]

        plot_x, plot_y = self.plot_xy(x,y)
        pg.draw.line(self.surface, (0, 200, 0), (plot_x, plot_y), (self.size[0]-self.label_surface.get_width(), self.label_surface.get_height()), 2)

        
        label_heading = self.font.render("Selected Solution:", True, (0, 200, 0))
        label_demand = self.font.render(f"Demand = {y:.2f}", True, (0, 200, 0))
        label_distance = self.font.render(f"Distance = {x:.2f}", True, (0, 200, 0))
        label_id = self.font.render(f"Solution number = {self.sorted_pareto_front[selected_solution]}", True, (0,200,0))

        self.label_surface.fill((10, 20, 10))
        pg.draw.rect(self.label_surface, (0, 200, 0), (0, 0, self.label_surface.get_width(), self.label_surface.get_height()), width=2)
        self.label_surface.blit(label_heading, (5,5))
        self.label_surface.blit(label_demand, (5,30))
        self.label_surface.blit(label_distance, (5,55))
        self.label_surface.blit(label_id, (5, 80))

    def plot_xy(self, x, y):
        plot_x = self.p2[0] + (x-self.minX) * self.graph_width / self.rangeX
        plot_y = self.p2[1] - (y-self.minY) * self.graph_height / self.rangeY

        return plot_x, plot_y

    def calculate_hypervolume(self):
        self.n_hypervolume_indicator = 0.0
        prev_dist = self.ref_dist


        for solution_number in self.sorted_pareto_front:
            dist = self.data[0][solution_number]
            dmnd = self.data[1][solution_number]

            width = prev_dist - dist 

            height = self.ref_dmnd - dmnd

            self.n_hypervolume_indicator += width*height
            prev_dist = dist

        # normalise the hypervolume indicator:

        min_dist = min(self.data[0])
        min_demand = min(self.data[1])

        self.max_volume = (self.ref_dist * self.ref_dmnd)
        if self.max_volume <= 0:
            self.n_hypervolume_indicator = 0.0
        else:
            self.n_hypervolume_indicator /= self.max_volume







