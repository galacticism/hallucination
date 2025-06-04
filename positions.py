import json
import chess.pgn

# this takes pgn file (which is a bunch of chess games) and extracts positions from each game with the list of moves
# to get to that position and saves it to a json file used to prompt the LLM
def extract_positions_from_pgn(pgn_path):
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
    return position_data

# Training positions
train_positions = extract_positions_from_pgn("game.pgn")
with open("positions.json", "w") as outfile:
    json.dump(train_positions, outfile, indent=2)

# Test positions
test_positions = extract_positions_from_pgn("test.pgn")
with open("tests.json", "w") as outfile:
    json.dump(test_positions, outfile, indent=2)
