from math import sin, cos, pi
from random import choice


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

        self.angle += (angle_offset + 180) % 360 - 180

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
            print(len(points))
            point = points[0]
            if not (point in self.nowRoad.points):
                self.nowRoad.points.append(point)
            self.nowCrossRoad = point

            line = None

            lines = list(filter(lambda a: not a.used,point.lines))
            if len(lines) == 0:
                next_point = None

                neighbours = self.__neighbour_points(point)
                for i in neighbours:
                    if len(list((lambda a: not a.used,i.lines))) > 0:
                        next_point = i
                        break

            else:
                line = choice(list(filter(lambda a: isinstance(a, Road) and not a.used and a != self.nowRoad, point.type)))

        else:
            type = data[1][0]
            slicer = (self.angle + data[0]) // 90 + 2
            now = slicer - (2*int(not slicer % 2))

            point = CrossRoad(self, (type, (x, y), slicer))
            if self.nowRoad:
                self.nowRoad.points.append(point)
            self.points.append(point)

            for j, el in enumerate(point.type):
                if el == 0:
                    continue
                if j == now and self.nowRoad:
                    point.set_road(self.nowRoad, j)
                    continue
                road = Road(self, [point], False if j != (slicer+2) % 4 else True)
                point.set_road(road, j)
                self.lines.append(road)

            line = choice(list(filter(lambda a: isinstance(a, Road) and not a.used, point.type)))
            line.used = True
            move = (point.type.index(line) + slicer - 3) % 4
            self.nowCrossRoad = point
            self.nowRoad = line

            # (control callback(move))
    def __neighbour_points(self, point):
        points = []
        for i, el in enumerate(point.type):
            if el == 0:
                continue
            elif len(el.points) > 1:
                points.append(list(filter(lambda a: a!=point,el.points))[0])
        return points

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

# tests
duck = DuckAI(((1, 1), 180))

for i in range(5):
    duck.tick((0, 0))

duck.cross_road([0, ['1', '8']])

duck.tick((3, 90))

duck.cross_road([0, ['1', '10']])
