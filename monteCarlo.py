# -*- coding: utf-8 -*-


import Reversi
from playerInterface import *
import math
import time
import random as rd


class myPlayerInternalBoard(Reversi.Board):
    """Improved board class to obtain new capabilities"""

    def __getitem__(self, i, j):
        return self._board[i][j]

    def nb_legal_moves_per_color(self, color):
        comp = 0
        for x in range(0, self._boardsize):
            for y in range(0, self._boardsize):
                if self.lazyTest_ValidMove(color, x, y):
                    comp += 1
        return max(1, comp)


class myPlayer(PlayerInterface):
    """Reversi AI player implementation"""

    @staticmethod
    def flipColor(color):
        """Invert the color"""
        if (color == 1):
            return 2
        elif (color == 2):
            return 1
        else:
            return None

    def __init__(self):
        """Init the player with an internal board, an horizon level and a color"""
        self._horizon = 1 # <= réglage de la profondeur d'exploration souhaitée ou du temps de recherche en secondes
        self._internalBoard = None
        self._myColor = None
        self._opponentColor = None

    def getPlayerName(self):
        """Retrieve the player's name"""
        return "monteCarlo"

    def getPlayerMove(self):
        """Compute and return a move"""
        if self._internalBoard.is_game_over():
            print(self.getPlayerName() + " >> Well.. I guess it's over")
            return (-1, -1)

        #################################################################################
        # Change strategy HERE

        move = self.monteCarloTime(self._horizon)

        #################################################################################

        self._internalBoard.push(move)
        print(self.getPlayerName() + " >> ", move, ", I choose Ya!")
        (c, x, y) = move
        assert (c == self._myColor)
        print(self.getPlayerName() + " >> This is my current board :", self._internalBoard)
        return (x, y)

    def playOpponentMove(self, x, y):
        """Register the opponent's last move"""
        assert (self._internalBoard.is_valid_move(self._opponentColor, x, y))
        # print(self.getPlayerName() + " >> Foe used move", (x, y))
        self._internalBoard.push([self._opponentColor, x, y])

    def newGame(self, color):
        """Init colors following a new game"""
        self._internalBoard = myPlayerInternalBoard(10)
        self._myColor = color
        self._opponentColor = myPlayer.flipColor(color)

    def endGame(self, winner):
        """Display final message following the winner"""
        if self._myColor == winner:
            print(self.getPlayerName() + " >> Yeahh!!!")
        else:
            print(self.getPlayerName() + " >> Oh, noooo :(!!")


    ########################################################################################
    ############################# Monte Carlo Tree Search ##################################
    ########################################################################################

    def monteCarloTime(self, timeSearch):
        """Strategy implementing the Monte Carlo Tree Search algorithm with our heuristic
        and a definite time search"""

        mct = MonteCarlo(self._internalBoard, self._myColor)
        start = time.time()
        while time.time() - start < timeSearch:
            mct.treeWalk(mct._tree)

        nodeList = mct._tree._childs
        bestMove = self._internalBoard.legal_moves()[0]
        bestScore = 0
        for n in nodeList:
            if n._visited > bestScore:
                bestScore = n._visited
                bestMove = n._move
        return bestMove




########################################################################################
########################### Structures for Monte Carlo #################################
########################################################################################

class Node:
    """class node useful for the representation of the tree of moves"""

    def __init__(self, parent, move):
        self._parent = parent
        self._move = move
        self._childs = []
        self._visited = 0
        self._successful = 0
        self._mu = float("inf")

    def championChild(self):
        champ = self._childs[0]
        muChamp = self._mu
        for c in self._childs:
            if c._mu > muChamp:
                muChamp = c._mu
                champ = c
        return champ

    def update(self, reward):
        self._visited += 1
        self._successful += reward
        self._mu = self._successful / self._visited + math.sqrt(2*math.log2(self._parent._visited+1)/self._visited) if self._parent != None else 0


class MonteCarlo:
    """Monte Carlo class with tree walk and random walk methods"""

    def __init__(self, board, color):
        self._board = board
        self._tree = Node(None, None)
        self._color = color

    def generateChilds(self, node):
        if self._board.is_game_over():
            return

        possible_moves = self._board.legal_moves()
        for m in possible_moves:
            node._childs.append(Node(node, m))

    def treeWalk(self, node):

        if node._childs != []:
            cc = node.championChild()
            self._board.push(cc._move)
            reward = self.treeWalk(cc)
        else:
            self.generateChilds(node)
            if node._childs == []:
                self._board.push(self._board.legal_moves()[0])
                reward = self.randomWalk()
            else:
                id = rd.randrange(len(node._childs))
                child = node._childs[id]
                self._board.push(child._move)
                reward = self.randomWalk()

        node.update(reward)
        self._board.pop()
        return reward


    def randomWalk(self):
        count = 0

        while not self._board.is_game_over():
            legal_moves = self._board.legal_moves()
            idrand = rd.randrange(len(legal_moves))
            randomMove = legal_moves[idrand]
            self._board.push(randomMove)
            count += 1

        nbWhite, nbBlack = self._board.get_nb_pieces()
        reward = 0

        if nbWhite > nbBlack and self._color == 2:
            reward = 1
        if nbWhite < nbBlack and self._color == 1:
            reward = 1

        for k in range(count):
            self._board.pop()

        return reward




