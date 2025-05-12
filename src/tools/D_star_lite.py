"""
D_star_Lite 2D
@author: huiming zhou
"""

import os
import sys
import math
import matplotlib.pyplot as plt
import numpy as np


class DStar:
    def __init__(self, s_start, s_goal, graph, heuristic_type):
        self.s_start, self.s_goal = s_start, s_goal
        self.heuristic_type = heuristic_type

        self.g, self.rhs, self.U, self.c = {}, {}, {}, {}
        self.km = 0
        self.dim = len(s_start)
        self.dim_divide_2 = self.dim//2

        self.graph = graph
        for v in graph.edges:
            self.rhs[v] = float("inf")
            self.g[v] = float("inf")
        # for edge in graph.E:
        #     self.c[edge] = float("inf")
        self.inf = float("inf")

        self.rhs[self.s_goal] = 0.0
        self.U[self.s_goal] = self.CalculateKey(self.s_goal)
        self.visited = set()
        self.count = 0
        # self.fig = plt.figure()

    def run(self):
        # self.Plot.plot_grid("D* Lite")
        self.ComputePath()
        self.extract_path()

        # self.plot_path()
        # self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        # plt.show()

    # def on_press(self, event):
    #     x, y = event.xdata, event.ydata
    #     if x < 0 or x > self.x - 1 or y < 0 or y > self.y - 1:
    #         print("Please choose right area!")
    #     else:
    #         x, y = int(x), int(y)
    #         print("Change position: s =", x, ",", "y =", y)
    #
    #         s_curr = self.s_start
    #         s_last = self.s_start
    #         i = 0
    #         path = [self.s_start]
    #
    #         while s_curr != self.s_goal:
    #             s_list = {}
    #
    #             for s in self.get_neighbor(s_curr):
    #                 s_list[s] = self.g[s] + self.cost(s_curr, s)
    #             s_curr = min(s_list, key=s_list.get)
    #             path.append(s_curr)
    #
    #             if i < 1:
    #                 self.km += self.h(s_last, s_curr)
    #                 s_last = s_curr
    #                 if (x, y) not in self.obs:
    #                     self.obs.add((x, y))
    #                     plt.plot(x, y, 'sk')
    #                     self.g[(x, y)] = float("inf")
    #                     self.rhs[(x, y)] = float("inf")
    #                 else:
    #                     self.obs.remove((x, y))
    #                     plt.plot(x, y, marker='s', color='white')
    #                     self.UpdateVertex((x, y))
    #                 for s in self.get_neighbor((x, y)):
    #                     self.UpdateVertex(s)
    #                 i += 1
    #
    #                 self.count += 1
    #                 self.visited = set()
    #                 self.ComputePath()
    #
    #         self.plot_visited(self.visited)
    #         self.plot_path(path)
    #         self.fig.canvas.draw_idle()

    def ComputePath(self):
        while True:
            if len(self.U) == 0:
                # break
                return False
                # return True
            s, v = self.TopKey()
            if v >= self.CalculateKey(self.s_start) and \
                    self.rhs[self.s_start] == self.g[self.s_start]:
                break

            k_old = v
            self.U.pop(s)
            # self.visited.add(s)

            if k_old < self.CalculateKey(s):
                self.U[s] = self.CalculateKey(s)
            elif self.g[s] > self.rhs[s]:
                self.g[s] = self.rhs[s]
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)
            else:
                self.g[s] = float("inf")
                self.UpdateVertex(s)
                for x in self.get_neighbor(s):
                    self.UpdateVertex(x)
        return True

    def UpdateVertex(self, s):
        if s != self.s_goal:
            self.rhs[s] = float("inf")
            tmp1 = self.get_neighbor(s)
            for x in self.get_neighbor(s):
                tmp = min(self.rhs[s], self.g[x] + self.cost(s, x))
                self.rhs[s] = tmp
        if s in self.U:
            self.U.pop(s)

        if self.g[s] != self.rhs[s]:
            self.U[s] = self.CalculateKey(s)

    def CalculateKey(self, s):
        return [min(self.g[s], self.rhs[s]) + self.h(self.s_start, s) + self.km,
                min(self.g[s], self.rhs[s])]

    def TopKey(self):
        """
        :return: return the min key and its value.
        """

        s = min(self.U, key=self.U.get)
        return s, self.U[s]

    def h(self, s_start, s_goal):
        heuristic_type = self.heuristic_type  # heuristic type

        if heuristic_type == "manhattan":
            return abs(s_goal[0] - s_start[0]) + abs(s_goal[1] - s_start[1])
        else:
            # return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
            return np.linalg.norm(np.array(s_start) - np.array(s_goal))

    def cost(self, s_start, s_goal):
        """
        Calculate Cost for this motion
        :param s_start: starting node
        :param s_goal: end node
        :return:  Cost for this motion
        :note: Cost function could be more complicate!
        """

        if self.is_collision(s_start, s_goal):
            return float("inf")

        if s_start + s_goal in self.c:
            return self.c[s_start + s_goal]
        elif s_goal + s_start in self.c:
            return self.c[s_goal + s_start]
        else:
            # dis = math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])
            dis = np.linalg.norm(np.array(s_start) - np.array(s_goal))
            self.c[s_start + s_goal] = dis
            self.c[s_goal + s_start] = dis
            return dis

    def update_cost(self, edges):
        # egdes, a list of tuple of (4,), to represent a list of edges
        # update self.c of edges which have changed states of collision
        for edge in edges:
            if self.graph.E[edge]:
                # dis = math.hypot(edge[0] - edge[2], edge[1] - edge[3])
                dis = np.linalg.norm(np.array(edge[:self.dim]) - np.array(edge[self.dim:]))
                self.c[edge] = dis
                self.c[edge[2:] + edge[:2]] = dis
            else:
                self.c[edge] = self.inf
                self.c[edge[2:] + edge[:2]] = self.inf

            self.UpdateVertex(edge[:self.dim])
            self.UpdateVertex(edge[self.dim:])

    def is_collision(self, s_start, s_end):
        return not self.graph.E[s_start + s_end]

    def get_neighbor(self, s):
        nei_list = set()
        for u in self.graph.edges[s]:
            if self.graph.E[s+u]:  # collision-free vertex
                nei_list.add(u)
        return nei_list

    def extract_path(self, max_nums=1000):
        """
        Extract the path based on the PARENT set.
        :return: The planning path
        """

        path = [self.s_start]
        s = self.s_start

        for k in range(max_nums):
            g_list = {}
            for x in self.get_neighbor(s):
                # if not self.is_collision(s, x):
                # g_list[x] = self.g[x] + self.c[s + x]
                g_list[x] = self.g[x]
            if len(g_list):
                s = min(g_list, key=g_list.get)
                path.append(s)
            if s == self.s_goal:
                break

        return list(path)

    # def plot_path(self, path):
    #     px = [x[0] for x in path]
    #     py = [x[1] for x in path]
    #     plt.plot(px, py, linewidth=2)
    #     plt.plot(self.s_start[0], self.s_start[1], "bs")
    #     plt.plot(self.s_goal[0], self.s_goal[1], "gs")

    def plot_visited(self, visited):
        color = ['gainsboro', 'lightgray', 'silver', 'darkgray',
                 'bisque', 'navajowhite', 'moccasin', 'wheat',
                 'powderblue', 'skyblue', 'lightskyblue', 'cornflowerblue']

        if self.count >= len(color) - 1:
            self.count = 0

        for x in visited:
            plt.plot(x[0], x[1], marker='s', color=color[self.count])


def main():
    s_start = (5, 5)
    s_goal = (45, 25)

    dstar = DStar(s_start, s_goal, "euclidean")
    dstar.run()


if __name__ == '__main__':
    main()
