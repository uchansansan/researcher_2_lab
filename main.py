from math import sin, cos, pi


class DuckAI:
    # input: data = ((x, y), angle)
    def __init__(self, data):
        self.startPos = data[0]
        self.angle = data[1]
        self.nowRoad = None

        self.points = []
        self.lines = []
        self.nowPos = {
            'x': data[0][0],
            'y': data[0][1]
        }

    # input: movements=(v_speed, v_movement)
    def tick(self, movements):
        s = movements[0]
        angle_offset = movements[1]

        self.angle += angle_offset
        self.angle += -360 if self.angle > 180 else (360 if self.angle <= -180 else 0)

        y = s * sin(self.angle * pi / 180)
        x = s * cos(self.angle * pi / 180)

        self.nowPos['x'] += x
        self.nowPos['y'] += y

    #input: data=[None/]
    def cross_road(self, data):
        


class CrossRoad:
    # input: duckbot=DuckAI child, data=(count of roads, (x, y))
    def __init__(self, duck_bot, data):
        self.siezds = []
        self.pos = data[1]
        self.id = len(duck_bot.points)


class Road:
    # input: duckbot=DuckAI child, points=(int)//ids of cross roads
    def __init__(self, duckbot, points):
        self.id = len(duckbot.points)
        self.points = points

