import Board
import copy
import random as rand

class TicTacToe():
    def __init__(self):
        self.player1 = True
        self.n = 0
        self.r = 0
        self.parent = None
        self.children = None
        self.board = [["-","-","-"],
                      ["-","-","-"],
                      ["-","-","-"]]

    def getChildren(self):
        if self.children == None:
            self.children = []
            for row in range(len(self.board)):
                for col in range(len(self.board[row])):
                    if self.board[row][col] == "-":
                        newChild = TicTacToe()
                        newChild.board = copy.deepcopy(self.board)
                        newChild.board[row][col] = "X" if self.player1 else "O"
                        newChild.player1 = not self.player1
                        newChild.parent = self
                        self.children.append(newChild)
        return self.children
    
    def isFinished(self):
        # Check r->l Diagonal
        if self.board[0][0] != "-" and self.board[0][0] == self.board[1][1] and self.board[0][0] == self.board[2][2]:
            return self.board[0][0]

        # Check l->r Diagonal
        if self.board[0][2] != "-" and self.board[0][2] == self.board[1][1] and self.board[0][2] == self.board[2][0]:
            return self.board[0][2]
        
        noZeros = True
        for row in range(len(self.board)):
            # Check rows
            if self.board[row][0] != "-" and self.board[row][0] == self.board[row][1] and self.board[row][0] == self.board[row][2]:
                return self.board[row][0]
            
            # Check Columns
            if self.board[0][row] != "-" and self.board[0][row] == self.board[1][row] and self.board[0][row] == self.board[2][row]:
                return self.board[0][row]
            
            for col in range(len(self.board[row])):
                # Account for Tie
                if self.board[row][col] == "-":
                    noZeros = False

        if noZeros:
            return 0
        
    def getReward(self, value):
        if value == "X":
            return 1
        elif value == "O":
            return -1
        else:
            return value
    
    def getNextRandom(self):
        return rand.choice(self.children)
    
    def randomSimulate(self):
        currentState = self
        while currentState.isFinished() == None:
            # Could use a getChoices instead of creating the memory of every child
            currentState.getChildren()
            currentState = currentState.getNextRandom()
            currentState.printBoard()

        return self.getReward(currentState.isFinished())
    
    def printBoard(self):
        for row in self.board:
                print(row[0], row[1], row[2])

        print("N:", self.n, "R:", self.r)
        print()


ttt = TicTacToe()

r = ttt.randomSimulate()
ttt.printBoard()
print(r)