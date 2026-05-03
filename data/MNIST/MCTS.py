from TicTacToe import TicTacToe

class MCTS:
    def __init__(self, currentBoard):
        self.currentBoard = currentBoard

    def select(self):
        # "Expand" happens ust by calling getChildren
        return self.banditPick(self.currentBoard.getChildren())
    
    def banditPick(self, children):
        bestChild = children[0]
        for child in children:
            # Score these based off of their n and r values
            pass
        return bestChild

    def simulate(self):
        return self.currentBoard.randomSimulate()

    def backPropogate(self, reward):
        parent = self.currentBoard.parent
        while parent != None:
            parent.r += reward
            parent.n += 1
            parent = parent.parent
        
    def step(self):
        chosen = self.select()
        reward = self.simulate(chosen)
        self.backPropogate(reward)
        return chosen

board = TicTacToe()  
game = MCTS(board)
nextBoard = game.step()
print(nextBoard.printBoard())
