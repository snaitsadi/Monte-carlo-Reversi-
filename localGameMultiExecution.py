import ColorfulReversi
import myPlayer as mp
import specialplayers.randomPlayer as rd
import time
from io import StringIO
import sys

nbGames = 0
nbWin = 0

# Players definition
player1 = mp.myPlayer()

# Choisir le joueur numero 2 entre les specialplayers et myPlayer
player2 = rd.randomPlayer()
# player2 = mp.myPlayer()

print("Black: ", player1.getPlayerName())
print("White: ", player2.getPlayerName())

for i in range(0, 100):
    nbGames += 1
    b = ColorfulReversi.Board(10)

    players = []
    player1.newGame(b._BLACK)
    players.append(player1)
    player2.newGame(b._WHITE)
    players.append(player2)

    totalTime = [0, 0]  # total real time for each player
    nextplayer = 0
    nextplayercolor = b._BLACK
    nbmoves = 1

    outputs = ["", ""]
    sysstdout = sys.stdout
    stringio = StringIO()
    # Problème : quand on est en fin de partie, le ID est relancé des millieurs de fois avec une profondeur max très grande
    while not b.is_game_over():
        nbmoves += 1
        otherplayer = (nextplayer + 1) % 2
        othercolor = b._BLACK if nextplayercolor == b._WHITE else b._WHITE

        currentTime = time.time()
        sys.stdout = stringio
        move = players[nextplayer].getPlayerMove()
        sys.stdout = sysstdout
        playeroutput = "\r" + stringio.getvalue()
        stringio.truncate(0)
        outputs[nextplayer] += playeroutput
        totalTime[nextplayer] += time.time() - currentTime
        (x, y) = move
        if not b.is_valid_move(nextplayercolor, x, y):
            break
        b.push([nextplayercolor, x, y])
        players[otherplayer].playOpponentMove(x, y)

        nextplayer = otherplayer
        nextplayercolor = othercolor

    (nbwhites, nbblacks) = b.get_nb_pieces()
    if nbwhites > nbblacks:
        print("\033[92mW\033[0m", end="")
    elif nbblacks > nbwhites:
        nbWin += 1
        print("\033[91mB\033[0m", end="")
    else:
        print(".", end="")
    sys.stdout.flush()

print("\n" + str(nbWin) + " of " + str(nbGames) + " -> " + str((nbWin / nbGames) * 100) + "%")
