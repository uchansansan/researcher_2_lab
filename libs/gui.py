import turtle
import networkx.drawing.nx_pydot
import pygraphviz
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import cv2


class GUI:
    #turtle.left(90)
    #screen = turtle.Screen()
    #screen.setup(800, 800)
    #screen.title('Map')
    #turtle.pensize(1)
    frame_skip = 1
    IMGNAME = "graph.png"

    def update_graph(tiles, cur_edge):
        g = nx.MultiGraph()
        nodes = tiles.keys()
        g.add_nodes_from(nodes)
        ids = []

        for node in nodes:
            for direction in tiles[node]:
                for info in tiles[node][direction]:
                    if info == 'edge' and tiles[node][direction][info] not in ids:
                        ids.append(tiles[node][direction][info])
                        if cur_edge == tiles[node][direction][info]:
                            g.add_edge(node, tiles[node][direction]['vertex'], data=str(tiles[node][direction][info]),
                                       color="red")
                        else:
                            g.add_edge(node, tiles[node][direction]['vertex'], data=str(tiles[node][direction][info]),
                                   color="blue")

        A = nx.nx_agraph.to_agraph(g)
        A.draw(GUI.IMGNAME, prog="neato")  # "neato", "dot", "twopi", "circo", "fdp", "nop"

    def draw_path_debug(angle, pos):
        turtle.pendown()
        x, _, y = pos * 50
        turtle.setheading(-angle * 180 / np.pi)
        turtle.setpos((x, y))

    def draw_path(action):
        """
        This function draw the path of a DuckieBot in a new screen
        :return:
        """
        from_wierd2turtle = 1.1784411471630984  # magic constant
        action *= GUI.frame_skip
        turtle.left(action[1] / from_wierd2turtle)
        turtle.forward(action[0])
        bot_pos = turtle.pos()
        bot_heading = turtle.heading()
        # return bot_pos, bot_heading

    def draw_graph(g=None):
        img = cv2.imread(GUI.IMGNAME)
        cv2.imshow(GUI.IMGNAME, img)
