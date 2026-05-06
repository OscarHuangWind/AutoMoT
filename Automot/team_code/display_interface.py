"""
Display Interface for visualizing autonomous driving agent behavior.
Provides real-time visualization of camera view, BEV trajectory, and decisions.
"""

import pygame
import numpy as np
import cv2


class DisplayInterface(object):
    def __init__(self):
        self._width = 1200
        self._height = 600
        self._surface = None

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )

        pygame.display.set_caption("MoT Agent")

    def run_interface(self, input_data):
        rgb = input_data['rgb']
        trajectory = input_data['predicted_trajectory']
        decision_1s = input_data['decision_1s']
        decision_2s = input_data['decision_2s']
        decision_3s = input_data['decision_3s']
        surface = np.zeros((600, 1200, 3), np.uint8)
        surface[:, :800] = rgb
        surface[:400, 800:1200] = input_data['bev_traj']
        surface[440:600, 1000:1200] = trajectory[0:160, :]
        surface = cv2.putText(surface, input_data['language_1'], (20, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        surface = cv2.putText(surface, input_data['language_2'], (20, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        surface = cv2.putText(surface, input_data['control'], (20, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data['speed'], (20, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        surface = cv2.putText(surface, input_data['time'], (20, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # surface = cv2.putText(surface, 'Left  View', (40,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        # surface = cv2.putText(surface, 'Focus View', (335,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)
        # surface = cv2.putText(surface, 'Right View', (640,135), cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,0), 2)

        surface = cv2.putText(surface, 'Behavior Decision', (820, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        surface = cv2.putText(surface, 'Planned Trajectory', (1010, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        surface = cv2.putText(surface, decision_1s, (820, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        surface = cv2.putText(surface, decision_2s, (820, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        surface = cv2.putText(surface, decision_3s, (820, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # surface[:150,198:202]=0
        # surface[:150,323:327]=0
        # surface[:150,473:477]=0
        # surface[:150,598:602]=0
        # surface[148:152, :200] = 0
        # surface[148:152, 325:475] = 0
        # surface[148:152, 600:800] = 0
        surface[430:600, 998:1000] = 255
        surface[0:600, 798:800] = 255
        surface[0:600, 1198:1200] = 255
        surface[0:2, 800:1200] = 255
        surface[598:600, 800:1200] = 255
        surface[398:400, 800:1200] = 255

        # display image
        self._surface = pygame.surfarray.make_surface(surface.swapaxes(0, 1))
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))

        pygame.display.flip()
        pygame.event.get()
        return surface

    def _quit(self):
        pygame.quit()
