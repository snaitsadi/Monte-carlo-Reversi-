import ColorfulReversi

import specialplayers.randomPlayer as rd

import specialplayers.iterativeAlphaParity as iap
import specialplayers.iterativeMinimaxParity as imp
import specialplayers.iterativeAlphaUlti as iau
import specialplayers.iterativeMinimaxUlti as imu
import specialplayers.monteCarlo as mc

import time
from io import StringIO
import sys
import matplotlib.pyplot as plt

# Players definition
player2 = rd.randomPlayer()

playersToTest = []

# playersToTest.append(iap.myPlayer())
playersToTest.append(imp.myPlayer())
playersToTest.append(iau.myPlayer())
# playersToTest.append(imu.myPlayer())
playersToTest.append(mc.myPlayer())

results = [[] for k in range(len(playersToTest))]

times = [0.00001 * 10**k for k in range(4)]

print(times)


for p in range(len(playersToTest)):
    player1 = playersToTest[p]

    for t in range(len(times)):

        player1._horizon = times[t]

        nbGames = 0
        nbWin = 0

        print("Black: ", player1.getPlayerName(), ", horizon: ", times[t])
        print("White: ", player2.getPlayerName())

        for i in range(10):
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
        results[p].append((nbWin / nbGames) * 100)


# valeur relevées pour le graphique présent dans le readme (sur 100 duels)
# plt.plot(times, [57.0, 64.0, 71.0, 81.0, 87.0, 95.0], label="iterativeMinimaxParity")
# plt.plot(times, [49.0, 53.0, 54.0, 80.0, 99.0, 100.0], label="iterativeAlphaUlti")
# plt.plot(times, [46.0, 55.0, 62.0, 68.0, 79.0, 100.0], label="monteCarlo")

for p in range(len(playersToTest)):
    plt.plot(times, results[p], label=playersToTest[p].getPlayerName())
    print(playersToTest[p].getPlayerName(), " : ", results[p])

plt.plot([50 for i in range(len(times))], label="random limit")

plt.xlabel("Horizon (en secondes)")
plt.xscale('log')
plt.ylabel("Pourcentage de victoires")
plt.ylim((40, 100))
plt.title("Pourcentage de victoires de différents joueurs contre un randomPlayer en fonction de l'horizon en secondes")
plt.legend()
plt.show()