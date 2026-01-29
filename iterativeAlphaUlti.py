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

    _ultimateHeuristicMatrix = [
        [20, -3, 11, 8, 4, 4, 8, 11, -3, 20],
        [-3, -7, -4, 1, 0, 0, 1, -4, -7, -3],
        [11, -4, 2, 2, 2, 2, 2, 2, -4, 11],
        [8, 1, 2, 0, 0, 0, 0, 2, 1, 8],
        [4, 0, 2, 0, -3, -3, 0, 2, 0, 4],
        [4, 0, 2, 0, -3, -3, 0, 2, 0, 4],
        [8, 1, 2, 0, 0, 0, 0, 2, 1, 8],
        [11, -4, 2, 2, 2, 2, 2, 2, -4, 11],
        [-3, -7, -4, 1, 0, 0, 1, -4, -7, -3],
        [20, -3, 11, 8, 4, 4, 8, 11, -3, 20],
    ]

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
        return "iterativeAlphaUlti"

    def getPlayerMove(self):
        """Compute and return a move"""
        if self._internalBoard.is_game_over():
            print(self.getPlayerName() + " >> Well.. I guess it's over")
            return (-1, -1)

        #################################################################################
        # Change strategy HERE

        move = self.iterativeDeepeningForTesting(self.firstAlphaBetaMax, self.computeUltimateScoreHeuristic, self._horizon)

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
    #################################  Heuristic Ultimate  #################################
    ########################################################################################

    def computeUltimateScoreHeuristic(self, color):
        """Calculates the 'Ultimate' heuristic taking in count the parity, mobility, corner closeness"""

        my_color = color
        opp_color = self.flipColor(color)
        my_tiles = 0
        opp_tiles = 0
        my_front_tiles = 0
        opp_front_tiles = 0
        d = 0.0
        X1 = [-1, -1, 0, 1, 1, 1, 0, -1]
        Y1 = [0, 1, 1, 1, 0, -1, -1, -1]

        for i in range(0, 10):
            for j in range(0, 10):
                if (self._internalBoard.__getitem__(i, j) == my_color):
                    d += self._ultimateHeuristicMatrix[i][j]
                    my_tiles += 1
                elif (self._internalBoard.__getitem__(i, j) == opp_color):
                    d -= self._ultimateHeuristicMatrix[i][j]
                    opp_tiles += 1

                if (self._internalBoard.__getitem__(i, j) != 0):
                    for k in range(0, 8):
                        x = i + X1[k]
                        y = j + Y1[k]
                        if (x >= 0 and x < 8 and y >= 0 and y < 8 and self._internalBoard.__getitem__(x, y) == 0):
                            if (self._internalBoard.__getitem__(i, j) == my_color):
                                my_front_tiles += 1
                            else:
                                opp_front_tiles += 1
                            break

        # Parity
        if (my_tiles > opp_tiles):
            p = (100.0 * my_tiles) / (my_tiles + opp_tiles)
        elif (my_tiles < opp_tiles):
            p = -(100.0 * opp_tiles) / (my_tiles + opp_tiles)
        else:
            p = 0

        if (my_front_tiles > opp_front_tiles):
            f = -(100.0 * my_front_tiles) / (my_front_tiles + opp_front_tiles)
        elif (my_front_tiles < opp_front_tiles):
            f = (100.0 * opp_front_tiles) / (my_front_tiles + opp_front_tiles)
        else:
            f = 0

        # Corner occupancy
        my_tiles = 0
        opp_tiles = 0
        if (self._internalBoard.__getitem__(0, 0) == my_color):
            my_tiles += 1
        elif (self._internalBoard.__getitem__(0, 0) == opp_color):
            opp_tiles += 1
        if (self._internalBoard.__getitem__(0, 9) == my_color):
            my_tiles += 1
        elif (self._internalBoard.__getitem__(0, 9) == opp_color):
            opp_tiles += 1
        if (self._internalBoard.__getitem__(9, 0) == my_color):
            my_tiles += 1
        elif (self._internalBoard.__getitem__(9, 0) == opp_color):
            opp_tiles += 1
        if (self._internalBoard.__getitem__(9, 9) == my_color):
            my_tiles += 1
        elif (self._internalBoard.__getitem__(9, 9) == opp_color):
            opp_tiles += 1
        c = 25 * (my_tiles - opp_tiles)

        # Corner closeness
        my_tiles = 0
        opp_tiles = 0
        if (self._internalBoard.__getitem__(0, 0) == 0):
            if (self._internalBoard.__getitem__(0, 1) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(0, 1) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(1, 1) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(1, 1) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(1, 0) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(1, 0) == opp_color):
                opp_tiles += 1

        if (self._internalBoard.__getitem__(0, 9) == 0):
            if (self._internalBoard.__getitem__(0, 8) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(0, 8) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(1, 8) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(1, 8) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(1, 9) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(1, 9) == opp_color):
                opp_tiles += 1

        if (self._internalBoard.__getitem__(9, 0) == 0):
            if (self._internalBoard.__getitem__(9, 1) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(9, 1) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(8, 1) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(8, 1) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(8, 0) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(8, 0) == opp_color):
                opp_tiles += 1

        if (self._internalBoard.__getitem__(9, 9) == 0):
            if (self._internalBoard.__getitem__(8, 9) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(8, 9) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(8, 8) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(8, 8) == opp_color):
                opp_tiles += 1
            if (self._internalBoard.__getitem__(9, 8) == my_color):
                my_tiles += 1
            elif (self._internalBoard.__getitem__(9, 8) == opp_color):
                opp_tiles += 1

        l = -12.5 * (my_tiles - opp_tiles)

        # Mobility (number of possible movements you have)
        my_tiles = self._internalBoard.nb_legal_moves_per_color(my_color)
        opp_tiles = self._internalBoard.nb_legal_moves_per_color(opp_color)

        if (my_tiles > opp_tiles):
            m = (100.0 * my_tiles) / (my_tiles + opp_tiles)
        elif (my_tiles < opp_tiles):
            m = -(100.0 * opp_tiles) / (my_tiles + opp_tiles)
        else:
            m = 0

        # final weighted score
        score = (10 * p) + (801.724 * c) + (382.026 * l) + (78.922 * m) + (74.396 * f) + (10 * d)
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