from math import sin, cos, pi
from random import choice


class DuckAI:
    # input: data = ((x, y), angle)
    def __init__(self, data):
        self.crossroad = {'title': 0.4, 'msrl': 0.1}
        print(data)
        self.angle = data[1]*180/pi
        self.nowRoad = None
        self.nowCrossRoad = None

        self.points = []
        self.lines = []
        self.pos = {
            'x': data[0][0],
            'y': data[0][1]
        }

    def __offset(self, angle, s):
        y = s * sin(angle)
        x = s * cos(angle)
        return x, y

    # input: movements=(v_speed, v_movement)
    def tick(self, movements):
        s = movements[0]
        angle_offset = movements[1]

        self.angle = ((self.angle + angle_offset/1.75) + 180) % 360 - 180

        x, y = self.__offset(self.angle, s)

        self.pos['x'] += x
        self.pos['y'] += y

        return self.pos,self.angle

    # input: data=[[str]]

    def cross_road(self, data=[0, []]):
        d = round(self.angle/90)*90
        x, y = self.__offset(d, self.crossroad['title'] + self.crossroad['msrl'])
        x += self.pos['x']
        y += self.pos['y']

        slicer = d // 90 + 2
        points = list(filter(lambda cross: x - self.crossroad['title'] < cross.pos[0] and x + self.crossroad['title'] > cross.pos[0] and y - self.crossroad['title'] < cross.pos[1] and y + self.crossroad['title'] > cross.pos[1], self.points))


        line = None

        if len(self.points) > 0 and (self.lines) == len(list(filter(lambda l: len(l.points)==2, self.lines))):
            point = points[0]
            line = choice(list(filter(lambda a: isinstance(a, Road) and a != self.nowRoad, point.type)))

        elif len(points) > 0:
            point = points[0]
            if not (point in self.nowRoad.points):
                self.nowRoad.points.append(point)
            self.nowCrossRoad = point

            lines = list(filter(lambda a: not a.used, point.lines))
            if len(lines) == 0:
                next_point = None

                neighbours = self.__neighbour_points(point)
                for j in neighbours:
                    if len(list((lambda a: not a.used, j.lines))) > 0:
                        next_point = j
                        break
                line = list(filter(lambda a: a != 0 and point in a.points, next_point.type))[0]
            else:
                line = choice(list(filter(lambda a: isinstance(a, Road) and not a.used and a != self.nowRoad, point.type)))
        else:
            type_in = data[0]
            print(type_in)
            now = slicer - (2*int(not slicer % 2))
            point = CrossRoad(self, (type_in, (x, y)), slicer)
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
        move = (point.type.index(line) + slicer + 1) % 4
        print('debug in move', point.type.index(line), move, self.angle)
        self.nowCrossRoad = point
        self.nowRoad = line
        return move
        # send data to gui

    def debug(self,data):
        self.pos['x'] = data[0][0]
        self.pos['y'] = data[0][1]
        self.angle = data[1]

    def __neighbour_points(self, point):
        points = []
        for el in point.type:
            if el == 0:
                continue
            elif len(el.points) > 1:
                points.append(list(filter(lambda a: a != point, el.points))[0])
        return points


class CrossRoad:
    # input: duck_bot=DuckAI child, data=(cross type, (x, y)), slicer)
    def __init__(self, duck_bot, data, slicer):
        print(duck_bot.points)
        # l, f, r, b
        type_crossroad = {'8': [1, 1, 1, 1],
                          '11': [1, 0, 1, 1],
                          '9': [0, 1, 1, 1],
                          '10': [1, 1, 0, 1]
                          }
        self.type = type_crossroad[str(data[0])][:slicer] + type_crossroad[str(data[0])][slicer:]
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
