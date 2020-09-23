import numpy as np
import pickle

board_rows = 3
board_cols = 3

class State:
    def __init__(self, p1, p2):
        self.board = np.zeros((board_rows, board_cols))
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        self.playerSymbol = 1

    def getHash(self):
        self.boardHash = str(self.board.reshape(board_rows * board_cols))
        return self.boardHash

    def availablePositions(self):
        positions = []
        for i in range(board_rows):
            for j in range(board_cols):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1


    def winner(self):
        for i in range(board_rows):
            if sum(self.board[i, :]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[i, :]) == -3:
                self.isEnd = True
                return -1

        for i in range(board_cols):
            if sum(self.board[:, i]) == 3:
                self.isEnd = True
                return 1
            if sum(self.board[:, i]) == -3:
                self.isEnd = True
                return -1

        diag_sum1 = sum([self.board[i, i] for i in range(board_cols)])
        diag_sum2 = sum([self.board[i, board_cols - 1 - i] for i in range(board_cols)])
        diag_sum = max(abs(diag_sum1), abs(diag_sum2))
        if diag_sum == 3:
            self.isEnd = True
            if diag_sum1 == 3 or diag_sum2 == 3:
                return 1
            else:
                return -1


        if len(self.availablePositions()) == 0:
            self.isEnd = True
            return 0

        self.isEnd = False
        return None

    def availablePositions(self):
        positions = []
        for i in range(board_rows):
            for j in range(board_cols):
                if self.board[i, j] == 0:
                    positions.append((i, j))
        return positions

    def updateState(self, position):
        self.board[position] = self.playerSymbol
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def giveReward(self):
        result = self.winner()
        if result == 1:
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        elif result == -1:
            self.p1.feedReward(0)
            self.p2.feedReward(1)

        else:
            self.p1.feedReward(0.1)
            self.p2.feedReward(0.5)

    def reset(self):
        self.board= np.zeros((board_rows, board_cols))
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = -1

    def play(self, rounds = 100):
        for i in range(rounds):
            if i % 1000 == 0:
                print('Rounds {}'.format(i))
            while not self.isEnd:
                positions = self.availablePositions()
                p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
                self.updateState(p1_action)
                board_hash = self.getHash()
                self.p1.addState(board_hash)

                win = self.winner()
                if win is not None:
                    self.giveReward()
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    positions = self.availablePositions()
                    p2_action = self.p2.chooseAction(positions, self.board, self.playerSymbol)
                    self.updateState(p2_action)
                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
        p1.savePolicy()
    def play2(self):
        while not self.isEnd:
            positions = self.availablePositions()
            p1_action = self.p1.chooseAction(positions, self.board, self.playerSymbol)
            self.updateState(p1_action)
            self.showBoard()
            win = self.winner()
            if win is not None:
                if win == 1:
                    print(self.p1.name, ' wins !')
                else:
                    print('tie!')
                self.reset()
                break

            else:
                positions = self.availablePositions()
                p2_action = self.p2.chooseAction(positions)

                self.updateState(p2_action)
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.p2.name, ' wins !')
                    else:
                        print('tie!')
                    self.reset()
                    break

    def showBoard(self):
        for i in range(0, board_rows):
            print('-' * 12)
            out = '| '
            for j in range(0, board_cols):
                if self.board[i, j] == 1:
                    token = 'x'
                elif self.board[i, j] == -1:
                    token = 'o'
                elif self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-' * 12)



class Player:
    def __init__(self, name, exp_rate = 0.3):
        self.name = name
        self.states = []
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}

    def getHash(self, board):
        boardHash = str(board.reshape(board_rows * board_cols))
        return boardHash

    def chooseAction(self, positions, current_board, symbol):
        if np.random.uniform(0, 1) <= self.exp_rate:
            idx = np.random.choice(len(positions))
            action= positions[idx]

        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p] = symbol
                next_boardHash = self.getHash(next_board)
                value = 0 if self.states_value.get(next_boardHash) is None else self.states_value.get(next_boardHash)
                if value >= value_max:
                    value_max = value
                    action = p
        return action


    def addState(self, state):
        self.states.append(state)

    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

class HumanPlayer:
    def __init__(self, name):
        self.name = name

    def chooseAction(self, positions):
        while True:
            row = int(input('Input your action row: '))
            col = int(input('Input your action col: '))
            action = (row, col)
            if action in positions:
                return action


if __name__ == '__main__':
    
    p1 = Player('p1')
    p2 = Player('p2')

    st = State(p1, p2)
    print('training...')
    st.play(50000)

    p1 = Player('computer', exp_rate = 0)
    p1.loadPolicy('policy_p1')

    p2 = HumanPlayer('human')

    st = State(p1, p2)
    st.play2()






