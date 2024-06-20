class Node:
    def __init__(self):
        self.parent = None
        self.children = []
        self.state = []
        self.P = 0
        self.Q = 0
        self.N = 0
        self.W = 0
        self.action = None

    def score(self):
        return self.Q + self.P / (self.N + 1)