from abc import ABC, abstractmethod

class Board(ABC):
    def __init__(self):
        self.n = 0
        self.r = 0
        self.parent
        self.children = []

    @abstractmethod
    def getNextRandom():
        return Board
    
    @abstractmethod
    def getChildren():
        return []
    
    @abstractmethod
    def getNextState(action):
        return Board
    
    @abstractmethod
    def randomSimulate():
        return int
    
