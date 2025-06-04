import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess
import numpy as np
import json

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
    # either UCI move or SAN move if the model outputs that way, otherwise it's a hallucination
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move.uci(), True
    except ValueError:
        pass
    try:
        move = board.parse_san(move_str)
        if move in board.legal_moves:
            return move.uci(), True
    except Exception:
        pass
    return move_str, False

def extract_features(entry):
    move_sequence = entry['moves']
    fen = entry['fen']
    side_to_move = "White" if fen.split()[1] == "w" else "Black"
    side_to_move_num = 1 if fen.split()[1] == "w" else 0
    board = chess.Board(fen)
    legal_moves = [m.uci() for m in board.legal_moves]
    move, logprobs = make_move_and_logprobs(move_sequence, side_to_move, legal_moves)
    move_uci, is_legal = get_legal_move_in_uci(board, move)
    num_pieces = len(board.piece_map())
    num_legal_moves = len(legal_moves)
    num_moves_so_far = len(move_sequence)
    std_logprob = np.std(logprobs) if logprobs else 0
    terminal_status = int(board.is_check() or board.is_checkmate() or board.is_stalemate())
    # I used this to debug while I was doing the analysis
    '''
    print(f"Moves: {move_sequence}")
    print(f"FEN: {fen}")
    print(f"Legal moves: {legal_moves}")
    print(f"Raw model move: {move}")
    print(f"Move (UCI): {move_uci}")
    print(f"Legal: {is_legal}")
    print(f"Logprobs: {logprobs}")
    print(f"Num pieces: {num_pieces}")
    print(f"Avg legal move length: {num_legal_moves}")
    print(f"Num moves so far: {num_moves_so_far}")
    print(f"Std logprob: {std_logprob}")
    print(f"Terminal status: {terminal_status}")
    print("---")
    return {
        "mean_logprob": np.mean(logprobs),
        "std_logprob": std_logprob,
        "num_pieces": num_pieces,
        "num_legal_moves": num_legal_moves,
        "num_moves_so_far": num_moves_so_far,
        "side_to_move_num": side_to_move_num,
        "terminal_status": terminal_status,
        "legal": int(is_legal),
    }
    '''

# to figure out how often the model outputs a legal move
def compute_llm_legal_move_rate(data):
    total = len(data)
    legal = sum(d["legal"] for d in data)
    return legal / total

# to figure out how often a random UCI move is legal using rejection sampling using positions from training data
def random_uci_legal_rate(position_data, trials=1000):
    legal_count = 0
    for entry in random.sample(position_data, min(len(position_data), trials)):
        board = chess.Board(entry["fen"])
        legal_moves = set(m.uci() for m in board.legal_moves)
        squares = [chess.SQUARE_NAMES[i] for i in range(64)]
        move = random.choice(squares) + random.choice(squares)
        if move in legal_moves:
            legal_count += 1
    return legal_count / trials

# Logistic Regression Stuff Starts Here
with open("positions.json", "r") as infile:
    position_data = json.load(infile)

train_data = [extract_features(entry) for entry in position_data]

# this function finds whether the feature is actually useful by doing rejection sampling on both sides of the conditional
# and seeing if there is any noticeable difference in the probability that the model outputs a legal move
def analyze_feature_correlations(data):
    print("\n--- Feature Correlation Checks (non-binned) ---")

    # mean_logprob
    mean_logprobs_legal = [d["mean_logprob"] for d in data if d["legal"] == 1]
    mean_logprobs_illegal = [d["mean_logprob"] for d in data if d["legal"] == 0]
    print(f"mean_logprob (legal):   mean={np.mean(mean_logprobs_legal):.3f}, std={np.std(mean_logprobs_legal):.3f}")
    print(f"mean_logprob (illegal): mean={np.mean(mean_logprobs_illegal):.3f}, std={np.std(mean_logprobs_illegal):.3f}")

    # std_logprob
    stds_legal = [d["std_logprob"] for d in data if d["legal"] == 1]
    stds_illegal = [d["std_logprob"] for d in data if d["legal"] == 0]
    print(f"std_logprob (legal):    mean={np.mean(stds_legal):.3f}, std={np.std(stds_legal):.3f}")
    print(f"std_logprob (illegal):  mean={np.mean(stds_illegal):.3f}, std={np.std(stds_illegal):.3f}")

    # num_pieces
    pieces_legal = [d["num_pieces"] for d in data if d["legal"] == 1]
    pieces_illegal = [d["num_pieces"] for d in data if d["legal"] == 0]
    print(f"num_pieces (legal):     mean={np.mean(pieces_legal):.1f}")
    print(f"num_pieces (illegal):   mean={np.mean(pieces_illegal):.1f}")

    # num_moves_so_far
    moves_legal = [d["num_moves_so_far"] for d in data if d["legal"] == 1]
    moves_illegal = [d["num_moves_so_far"] for d in data if d["legal"] == 0]
    print(f"num_moves_so_far (legal):   mean={np.mean(moves_legal):.1f}")
    print(f"num_moves_so_far (illegal): mean={np.mean(moves_illegal):.1f}")

    # num_legal_moves
    lengths_legal = [d["num_legal_moves"] for d in data if d["legal"] == 1]
    lengths_illegal = [d["num_legal_moves"] for d in data if d["legal"] == 0]
    print(f"num_legal_moves (legal):   mean={np.mean(lengths_legal):.3f}")
    print(f"num_legal_moves (illegal): mean={np.mean(lengths_illegal):.3f}")

    # side_to_move
    white_legal = sum(1 for d in data if d["side_to_move_num"] == 1 and d["legal"] == 1)
    white_total = sum(1 for d in data if d["side_to_move_num"] == 1)
    black_legal = sum(1 for d in data if d["side_to_move_num"] == 0 and d["legal"] == 1)
    black_total = sum(1 for d in data if d["side_to_move_num"] == 0)
    print(f"P(legal | white to move): {white_legal / white_total:.3f}")
    print(f"P(legal | black to move): {black_legal / black_total:.3f}")

    # terminal_status
    terminal_legal = sum(1 for d in data if d["terminal_status"] == 1 and d["legal"] == 1)
    terminal_total = sum(1 for d in data if d["terminal_status"] == 1)
    normal_legal = sum(1 for d in data if d["terminal_status"] == 0 and d["legal"] == 1)
    normal_total = sum(1 for d in data if d["terminal_status"] == 0)
    print(f"P(legal | terminal_status=1): {terminal_legal / terminal_total:.3f} ({terminal_total} samples)")
    print(f"P(legal | terminal_status=0): {normal_legal / normal_total:.3f} ({normal_total} samples)")

X_train = np.array([
    [
        d["mean_logprob"],
        d["std_logprob"],
        d["num_pieces"],
        d["num_legal_moves"],
        d["num_moves_so_far"],
        d["side_to_move_num"],
        d["terminal_status"]
    ]
    for d in train_data
])
y_train = np.array([d["legal"] for d in train_data])

# test data
with open("tests.json", "r") as infile:
    test_position_data = json.load(infile)

test_data = [extract_features(entry) for entry in test_position_data]

X_test = np.array([
    [
        d["mean_logprob"],
        d["std_logprob"],
        d["num_pieces"],
        d["num_legal_moves"],
        d["num_moves_so_far"],
        d["side_to_move_num"],
        d["terminal_status"]
    ]
    for d in test_data
])
y_test = np.array([d["legal"] for d in test_data])

# Logistic Regression as implemented in Pset 8
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

np.random.seed(42)
w = np.zeros(X_train.shape[1])
b = 0
learning_rate = 0.01
epochs = 2000

for epoch in range(epochs):
    z = np.dot(X_train, w) + b
    p = sigmoid(z)
    dw = np.dot(X_train.T, (p - y_train)) / len(y_train)
    db = np.sum(p - y_train) / len(y_train)
    w -= learning_rate * dw
    b -= learning_rate * db

print("\nLearned weights (manual logistic regression):", w)
print("Intercept:", b)
print(f"Features: mean_logprob, std_logprob, num_pieces, num_legal_moves, num_moves_so_far, side_to_move_num, terminal_status")

# finding accuracies on training and testing data
z_train = np.dot(X_train, w) + b
p_train = sigmoid(z_train)
y_pred_train = (p_train > 0.5).astype(int)
train_acc = np.mean(y_pred_train == y_train)
print(f"\nTrain accuracy: {train_acc:.3f}")

z_test = np.dot(X_test, w) + b
p_test = sigmoid(z_test)
y_pred_test = (p_test > 0.5).astype(int)
test_acc = np.mean(y_pred_test == y_test)
print(f"Test accuracy: {test_acc:.3f}")


TP = np.sum((y_test == 1) & (y_pred_test == 1))
TN = np.sum((y_test == 0) & (y_pred_test == 0))
FP = np.sum((y_test == 0) & (y_pred_test == 1))
FN = np.sum((y_test == 1) & (y_pred_test == 0))

confusion_matrix = np.array([[TN, FP],
                             [FN, TP]])

print("Confusion Matrix:")
print("                Predicted")
print("               0       1")
print(f"Actual  0    {TN:5}   {FP:5}")
print(f"        1    {FN:5}   {TP:5}")

# --- Conditional Probability Analysis ---
llm_legal_move_rate = compute_llm_legal_move_rate(test_data)
print(f"\nLLM legal move rate on test set: {llm_legal_move_rate:.3f}")

rand_legal_rate = random_uci_legal_rate(test_position_data)
print(f"Random UCI move legal rate: {rand_legal_rate:.4f}")

def compute_bayes_estimates(data, logprob_threshold):
    high_confidence = [d for d in data if d["mean_logprob"] > logprob_threshold]
    legal_and_high_conf = [d for d in high_confidence if d["legal"] == 1]

    P_high_conf = len(high_confidence) / len(data) if data else 0
    P_legal = sum(d["legal"] for d in data) / len(data) if data else 0
    P_legal_given_high_conf = len(legal_and_high_conf) / len(high_confidence) if high_confidence else 0

    # bayes
    if P_legal > 0:
        P_high_conf_given_legal = (P_legal_given_high_conf * P_high_conf) / P_legal
    else:
        P_high_conf_given_legal = 0

    return {
        "P(high_confidence)": P_high_conf,
        "P(legal)": P_legal,
        "P(legal | high_confidence)": P_legal_given_high_conf,
        "P(high_confidence | legal)": P_high_conf_given_legal
    }

# logprob threshold to see if conditionals are different from each other below and above this threshold
threshold = -0.5 
bayes_results = compute_bayes_estimates(test_data, threshold)

print("\n--- Bayes' Rule Estimates ---")
for k, v in bayes_results.items():
    print(f"{k}: {v:.3f}")

analyze_feature_correlations(train_data)
