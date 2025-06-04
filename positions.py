# positions.py

import json
import chess.pgn

pgn_path = "game.pgn"
position_data = []

with open(pgn_path, "r") as pgn_file:
    while True:
        game = chess.pgn.read_game(pgn_file)
        if game is None:
            break
        board = game.board()
        moves = []
        for move in game.mainline_moves():
            moves.append(move.uci())
            board.push(move)
            position_data.append({'moves': list(moves), 'fen': board.fen()})

# Save as JSON
with open("positions.json", "w") as outfile:
    json.dump(position_data, outfile, indent=2)
