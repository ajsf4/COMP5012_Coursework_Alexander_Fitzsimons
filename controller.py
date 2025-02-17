import numpy as np

class controller():
    def __init__(self):
        self.translation = np.array([0, 0, 0])
        self.rotation = np.array([0, 0, 0])

    def transform(self, camera, speed, angular, dt):
        # Calculate the forward and right vectors based on the camera's orientation
        forward = np.array([np.cos(camera.orientation[2]),  np.sin(camera.orientation[2]), 0])
        right   = np.array([np.sin(camera.orientation[2]), -np.cos(camera.orientation[2]), 0])
        
        # Calculate the global translation based on the input and camera orientation
        globalTranslation = (
            self.translation[0] * right +
            self.translation[2] * forward +
            np.array([0, 0, self.translation[1]])
        ) * speed * dt
        
        camera.transform(globalTranslation, self.rotation * angular * dt)


