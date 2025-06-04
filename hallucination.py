import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import numpy as np
import re

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def make_move_and_logprobs(fen):
    prompt = (
        f"The following is a chess position in FEN format:\n{fen}\n"
        "What is the best move for the player to move? Output ONLY the move in UCI format (e.g., e2e4, g1f3). "
        "Do NOT include move numbers or any explanation.\n\n"
        "Examples:\n"
        "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
        "UCI: e2e4\n"
        "FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1\n"
        "UCI: c7c5\n"
        f"FEN: {fen}\n"
        "UCI:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )
    generated_tokens = output.sequences[0][inputs.input_ids.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().split()[0]
    token_logprobs = []
    for i, score_distribution in enumerate(output.scores):
        logprobs = torch.nn.functional.log_softmax(score_distribution, dim=-1)
        token_id = generated_tokens[i].item()
        token_logprobs.append(logprobs[0, token_id].item())
    return generated_text, token_logprobs

def get_legal_move_in_uci(board, move_str):
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move.uci(), True
    except ValueError:
        pass
    # Try SAN
    try:
        move = board.parse_san(move_str)
        if move in board.legal_moves:
            return move.uci(), True
    except Exception:
        pass
    return move_str, False

positions = [
    # Openings
    chess.STARTING_FEN,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppppppp/2n5/8/8/2N5/PPPPPPPP/R1BQKBNR w KQkq - 2 3",
    "rnbqkb1r/pp1p1ppp/2p1pn2/8/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
    "rnbqkbnr/pp2pppp/2p5/3p4/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 3",
    "r1bqkb1r/pppppppp/2n5/8/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 2 2",
    "rnbqk1nr/pppp1ppp/4p3/8/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqkbnr/pppppppp/2n5/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 3 3",
    "rnbqkb1r/pppppppp/5n2/8/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 3 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2",
    "rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2",

    # Early middlegame
    "rnbq1rk1/ppp2ppp/4pn2/3p4/3P4/2N1BN2/PPP1BPPP/R2QK2R w KQ - 2 7",
    "r1bq1rk1/pppp1ppp/2n2n2/4p3/3P4/2N1PN2/PPP1BPPP/R1BQK2R b KQ - 2 6",
    "rnbq1rk1/pp3ppp/2p1pn2/3p4/3P4/2N1PN2/PPP1BPPP/R1BQK2R w KQ - 0 7",
    "r1bq1rk1/ppp1ppbp/2n2np1/3p4/3P4/2N1PN2/PPP1BPPP/R1BQ1RK1 w - - 4 7",
    "r1bq1rk1/ppp1ppbp/2n2np1/3p4/3P4/2N1PN2/PPP1BPPP/R1BQ1RK1 b - - 4 7",
    "r2q1rk1/ppp2ppp/2nbbn2/3pp3/3PP3/2N1BN2/PPP2PPP/R1BQ1RK1 w - - 6 8",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/3PP3/2N1BN2/PPP2PPP/R1BQK2R w KQ - 0 7",
    "r2q1rk1/ppp1ppbp/2nb1np1/3p4/3P4/2N1PN2/PPP1BPPP/R1BQ1RK1 w - - 2 8",
    "r2q1rk1/ppp1ppbp/2nb1np1/3p4/3P4/2N1PN2/PPP1BPPP/R1BQ1RK1 b - - 3 8",
    "r1bq1rk1/ppp1ppbp/2n2np1/3p4/3P4/2N1PN2/PPP1BPPP/R1BQ1RK1 b - - 5 7",
    "r2q1rk1/ppp2ppp/2nbbn2/3pp3/3PP3/2N1BN2/PPP2PPP/R1BQ1RK1 w - - 8 8",

    # Middlegame
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b - - 5 6",
    "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N1PN2/PPP2PPP/R1BQ1RK1 w - - 8 7",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2N2N2/PPP2PPP/R1BQ1RK1 b - - 3 7",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 3 8",
    "r2q1rk1/ppp2ppp/2nbbn2/3pp3/3PP3/2N1BN2/PPP2PPP/R1BQ1RK1 b - - 7 8",
    "r1bq1rk1/pp1n1ppp/2n1pb2/2pp4/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 w - - 4 8",
    "r1bq1rk1/pp1n1ppp/2n1pb2/2pp4/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 b - - 5 8",
    "r1bq1rk1/pp1n1ppp/2n1pb2/2pp4/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 w - - 5 9",
    "r1bq1rk1/pp1n1ppp/2n1pb2/2pp4/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 b - - 6 9",
    "r1bq1rk1/pp1n1ppp/2n1pb2/2pp4/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 w - - 6 10",
    "r1bq1rk1/pp1n1ppp/2n1pb2/2pp4/8/2NP1NP1/PPP1PPBP/R1BQ1RK1 b - - 7 10",

    # Endgame (variety: pawn, rook, queen, etc.)
    "8/8/8/8/8/8/8/K6k w - - 0 1",
    "8/8/8/8/8/8/5k2/6K1 w - - 0 1",
    "8/8/8/8/8/6k1/7p/7K w - - 0 1",
    "8/8/8/5k2/8/8/5K2/8 w - - 0 1",
    "8/8/8/8/8/8/7k/7K w - - 0 1",
    "8/8/8/8/8/8/4k3/3K4 w - - 0 1",
    "8/8/8/8/8/8/2k5/2K5 w - - 0 1",
    "8/8/8/8/8/8/2K5/2k5 w - - 0 1",
    "8/8/8/8/8/8/1k6/1K6 w - - 0 1",
    "8/8/8/8/8/8/7K/7k w - - 0 1",

    # Some late endgames
    "8/5k2/8/8/8/8/6K1/8 w - - 0 1",
    "8/8/8/6k1/8/8/6K1/8 w - - 0 1",
    "8/8/5k2/8/8/8/5K2/8 w - - 0 1",
    "8/8/8/8/4k3/8/4K3/8 w - - 0 1",
    "8/8/8/8/8/2k5/8/2K5 w - - 0 1",
    "8/8/8/8/8/8/4K3/4k3 w - - 0 1",
    "8/8/8/8/8/8/7K/7k w - - 0 1",
    "8/8/8/8/8/8/8/K6k w - - 0 1",
    "8/8/8/8/8/8/6k1/7K w - - 0 1",
    "8/8/8/8/8/8/7K/6k1 w - - 0 1",
]

data = []

for fen in positions:
    board = chess.Board(fen)
    move, logprobs = make_move_and_logprobs(fen)
    move_uci, is_legal = get_legal_move_in_uci(board, move)
    num_pieces = len(board.piece_map())
    print(f"FEN: {fen}")
    print(f"Raw model move: {move}")
    print(f"Move (UCI): {move_uci}")
    print(f"Legal: {is_legal}")
    print(f"Logprobs: {logprobs}")
    print(f"Num pieces: {num_pieces}")
    print("---")
    data.append({
        "mean_logprob": np.mean(logprobs),
        "num_pieces": num_pieces,
        "legal": int(is_legal),
    })

# Convert to numpy arrays for sklearn
X = np.array([[d["mean_logprob"], d["num_pieces"]] for d in data])
y = np.array([d["legal"] for d in data])

if len(set(y)) > 1:
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print(f"Learned coefficients: {model.coef_}, intercept: {model.intercept_}")
    print(f"Features: mean_logprob, num_pieces")
else:
    print("Not enough data diversity for logistic regression (need both legal and illegal moves).")
