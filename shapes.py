import numpy as np

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
        self.defaultGrey = 100*np.ones(3)
        self.pointColours = []
        self.lineColours = []
        self.add_points()
    
    def add_points(self):
        grid_size = 20
        grid_spacing = 10
        for x in range(-grid_size, grid_size):
            for y in range(-grid_size, grid_size):
                self.points = np.append(self.points, Point(np.array([grid_spacing*x, grid_spacing*y, 0])))
                self.pointColours.append(self.defaultGrey)
                
                if y < 2*grid_size - 1:
                    self.lines.append([x*2*grid_size + y, x*2*grid_size + (y+1)])
                    self.lineColours.append(self.defaultGrey)
                if x < 2*grid_size - 1:
                    self.lines.append([x*2*grid_size + y, (x+1)*2*grid_size + y])
                    self.lineColours.append(self.defaultGrey)


class Path():
    def __init__(self, points, route):
        self.objectType = "Path"

        self.points = []
        self.pointColours = []
        self.lines = []
        self.lineColours = []
        self.defaultColour = (255, 255, 255)
        
        for i, p in enumerate(points):
            print(i, p)
            self.points.append(Point(np.array(p)))
            self.pointColours.append(self.defaultColour)
            if i < len(route)-1:
                self.lines.append((route[i], route[i+1]))
                self.lineColours.append(self.defaultColour)
            else:
                self.lines.append((route[i], route[0]))
                self.lineColours.append(self.defaultColour)

