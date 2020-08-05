
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        prcsn_dict={'high':100, 'low':1000, 'medium':500}
        spd_dict={'fast':1, 'slow':10, 'medium':5}

        self.precision=prcsn_dict[precision]
        self.speed=spd_dict[speed]

    def move(self, x, y):
        pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)
