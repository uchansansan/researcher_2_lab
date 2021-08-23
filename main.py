from math import sin, cos, pi
from random import choice

movements = {
    1: 'l',
    2: 'f',
    3: 'r'
}

class DuckAI:
    # input: data = ((x, y), angle)
    def __init__(self, data):
        self.crossroad = {'title': 0.4, 'msrl': 0.1}
        self.angle = data[1]
        self.nowRoad = None
        self.nowCrossRoad = None

        self.points = []
        self.lines = []
        self.pos = {
            'x': data[0][0],
            'y': data[0][1]
        }

    def __offset(self, angle, s):
        y = s * sin(angle * pi / 180)
        x = s * cos(angle * pi / 180)
        return x, y

    # input: movements=(v_speed, v_movement)
    def tick(self, movements):
        s = movements[0]
        angle_offset = movements[1]

        self.angle += angle_offset
        self.angle += -360 if self.angle > 180 else (360 if self.angle <= -180 else 0)

        x, y = self.__offset(self.angle, s)

        self.pos['x'] += x
        self.pos['y'] += y

    # input: data=[offsetAngle, [str]]
    def cross_road(self, data=[]):
        data[1].remove('1')
        x, y = self.__offset(self.angle + data[0], self.crossroad['title'] + self.crossroad['msrl'])
        x += self.pos['x']
        y += self.pos['y']
        points = list(filter(lambda cross: x - self.crossroad['title'] < cross.pos[0]
                                           and x + self.crossroad['title'] > cross.pos[0]
                                           and y - self.crossroad['title'] < cross.pos[1]
                                           and y + self.crossroad['title'] > cross.pos[1],
                             self.points))
        if len(points) > 0:
            #ji
            pass
        else:
            type = data[1][0]
            slicer = (self.angle + data[0]) // 90 + 2

            point = CrossRoad(self, (type, (x, y), slicer))
            self.points.append(point)
            for j, el in enumerate(point.type):
                if el == 0:
                    continue
                road = Road(self, [point], False if j != (slicer+2) % 4 else True)
                point.set_road(road, j)
                self.lines.append(road)
            line = choice(list(filter(lambda a: isinstance(a, Road) and not a.used, point.type)))
            line.used = True
            move = (point.type.index(line) + slicer - 3) % 4
            self.nowCrossRoad = point
            if self.nowRoad:
                self.nowRoad.points.add(point)
            self.nowRoad = line
            print(move)

            # (control callback(move))


class CrossRoad:
    # input: duck_bot=DuckAI child, data=(cross type, (x, y), slicer))
    def __init__(self, duck_bot, data):
        # l, f, r, b
        type = {'8': [1, 1, 1, 1],
                '11': [1, 0, 1, 1],
                '9': [0, 1, 1, 1],
                '10': [1, 1, 0, 1]
                }
        self.type = type[data[0]][:data[-1]] + type[data[0]][data[-1]:]
        self.pos = data[1]
        self.id = len(duck_bot.points)

    def set_road(self, road, ind):
        self.type[ind] = road


class Road:
    # input: duck_bot=DuckAI child, points=()//cross roads
    def __init__(self, duck_bot, points, used=False):
        self.id = len(duck_bot.points)
        self.points = points
        self.used = used


duck = DuckAI(((1, 1), 180))

for i in range(5):
    duck.tick((0, 0))

duck.cross_road([1, ['1', '9']])
