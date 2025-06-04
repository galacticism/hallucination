import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def make_move_and_logprobs(move_sequence, side_to_move, legal_moves):
    prompt = (
        "The following is a chess position after these moves (in UCI format):\n"
        + ", ".join(move_sequence)
        + f"\nIt is {side_to_move}'s turn to move."
        + "\nThe list of legal moves in this position is: " + ", ".join(legal_moves) + "."
        + "\nWhat is the best move for the player to move? Output ONLY the move in UCI format (e.g., e2e4, g1f3). "
        + "Do NOT include move numbers or any explanation.\n"
        + "Examples:\n"
        + "Moves: e2e4, e7e5, g1f3\nUCI: b8c6\n"
        + "Moves: e2e4, c7c5\nUCI: g1f3\n"
        + "Moves: " + ", ".join(move_sequence) + "\nUCI:"
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
    output_str = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    split_output = output_str.split()
    if split_output:
        generated_text = split_output[0]
    else:
        generated_text = ""
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

# Load positions data from positions.json
with open("positions.json", "r") as infile:
    position_data = json.load(infile)

data = []

for entry in position_data:
    move_sequence = entry['moves']
    fen = entry['fen']
    side_to_move = "White" if fen.split()[1] == "w" else "Black"
    side_to_move_num = 1 if fen.split()[1] == "w" else 0
    board = chess.Board(fen)
    legal_moves = [m.uci() for m in board.legal_moves]
    move, logprobs = make_move_and_logprobs(move_sequence, side_to_move, legal_moves)
    move_uci, is_legal = get_legal_move_in_uci(board, move)
    num_pieces = len(board.piece_map())
    avg_legal_move_length = np.mean([len(m) for m in legal_moves]) if legal_moves else 0
    num_moves_so_far = len(move_sequence)
    std_logprob = np.std(logprobs) if logprobs else 0
    # 1 if in check, checkmate, or stalemate; 0 otherwise
    terminal_status = int(board.is_check() or board.is_checkmate() or board.is_stalemate())

    print(f"Moves: {move_sequence}")
    print(f"FEN: {fen}")
    print(f"Legal moves: {legal_moves}")
    print(f"Raw model move: {move}")
    print(f"Move (UCI): {move_uci}")
    print(f"Legal: {is_legal}")
    print(f"Logprobs: {logprobs}")
    print(f"Num pieces: {num_pieces}")
    print(f"Avg legal move length: {avg_legal_move_length}")
    print(f"Num moves so far: {num_moves_so_far}")
    print(f"Std logprob: {std_logprob}")
    print(f"Terminal status: {terminal_status}")
    print("---")
    data.append({
        "mean_logprob": np.mean(logprobs),
        "std_logprob": std_logprob,
        "num_pieces": num_pieces,
        "avg_legal_move_length": avg_legal_move_length,
        "num_moves_so_far": num_moves_so_far,
        "side_to_move_num": side_to_move_num,
        "terminal_status": terminal_status,
        "legal": int(is_legal),
    })

# Convert to numpy arrays for sklearn
X = np.array([
    [
        d["mean_logprob"],
        d["std_logprob"],
        d["num_pieces"],
        d["avg_legal_move_length"],
        d["num_moves_so_far"],
        d["side_to_move_num"],
        d["terminal_status"]
    ]
    for d in data
])
y = np.array([d["legal"] for d in data])

if len(set(y)) > 1:
    model = LogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print(f"Learned coefficients: {model.coef_}, intercept: {model.intercept_}")
    print(f"Features: mean_logprob, std_logprob, num_pieces, avg_legal_move_length, num_moves_so_far, side_to_move_num, terminal_status")
else:
    print("Not enough data diversity for logistic regression (need both legal and illegal moves).")
