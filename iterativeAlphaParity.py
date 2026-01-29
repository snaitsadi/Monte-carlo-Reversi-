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
        return "iterativeAlphaParity"

    def getPlayerMove(self):
        """Compute and return a move"""
        if self._internalBoard.is_game_over():
            print(self.getPlayerName() + " >> Well.. I guess it's over")
            return (-1, -1)

        #################################################################################
        # Change strategy HERE

        move = self.iterativeDeepeningForTesting(self.firstAlphaBetaMax, self.computeParityScoreHeuristic, self._horizon)

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
    ################################   Heuristic Parity   ##################################
    ########################################################################################

    def computeParityScoreHeuristic(self, color):
        """Calculates the 'Parity-only' heuristic following the tokens parity in the current board"""
        (nbW, nbB) = self._internalBoard.get_nb_pieces()
        score = nbW - nbB if color == 2 else nbB - nbW
        return score


    ########################################################################################
    ##################################### AlphaBeta ########################################
    ########################################################################################

    def alphaBetaMax(self, heuristic, alpha, beta, l, lmax):
        """Strategy implementing the AlphaBeta MaxValue algorithm with our heuristic"""

        if self._internalBoard.is_game_over() or (l == lmax):
            return heuristic(self._myColor)

        possible_moves = self._internalBoard.legal_moves()
        for m in possible_moves:
            self._internalBoard.push(m)
            alpha = max(alpha, self.alphaBetaMin(heuristic, alpha, beta, l + 1, lmax))
            self._internalBoard.pop()
            if (alpha >= beta):
                return beta
        return alpha

    def alphaBetaMin(self, heuristic, alpha, beta, l, lmax):
        """Strategy implementing the AlphaBeta MinValue algorithm with our heuristic"""

        if self._internalBoard.is_game_over() or (l == lmax):
            return heuristic(self._myColor)

        possible_moves = self._internalBoard.legal_moves()
        for m in possible_moves:
            self._internalBoard.push(m)
            beta = min(beta, self.alphaBetaMax(heuristic, alpha, beta, l + 1, lmax))
            self._internalBoard.pop()
            if (alpha >= beta):
                return alpha
        return beta

    def firstAlphaBetaMax(self, heuristic, lmax):
        """Strategy implementing the first call of AlphaBeta MaxValue algorithm with our heuristic"""

        alpha = float("-inf")
        beta = float("inf")

        possible_moves = self._internalBoard.legal_moves()
        move = possible_moves[0]
        for m in possible_moves:
            self._internalBoard.push(m)
            prevAlpha = alpha
            alpha = max(alpha, self.alphaBetaMin(heuristic, alpha, beta, 1, lmax))
            if (prevAlpha != alpha):
                move = m
            self._internalBoard.pop()
            if (alpha >= beta):
                return move
        return move


    ########################################################################################
    ############################### Iterative Deepening ####################################
    ########################################################################################

    def iterativeDeepeningForTesting(self, explorer, heuristic, Texplo):
        """Strategy implementing the iterative deepening algorithm with our heuristic
        For the explorer, you have the choice between firstAlphaBetaMax and firstMaxMin
        This version is only here for tests and plots.
        Does not respect the time given but returns the move that should have been returned
        if the time would have been respected"""

        start = time.time()
        horizon = 1
        move = self._internalBoard.legal_moves()[0]

        while (time.time() - start < Texplo):
            newmove = explorer(heuristic, horizon)
            if(time.time() - start < Texplo):
                move = newmove
            horizon += 1
        return move