import numpy as np

class controller():
    def __init__(self):
        self.translation = np.array([0, 0, 0])
        self.rotation = np.array([0, 0])

    def transform(self, camera, speed, angular, dt):
        vx = self.translation[1] * np.cos(camera.orientation[0]) + self.translation[0] * np.sin(camera.orientation[0])
        vy = self.translation[1] * np.sin(camera.orientation[0]) - self.translation[0] * np.cos(camera.orientation[0])
        globalTranslation = np.array([vx, vy, self.translation[2]]) * speed * dt
        camera.transform(globalTranslation, self.rotation * angular * dt)


