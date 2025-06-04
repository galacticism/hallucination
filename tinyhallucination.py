import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import chess

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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
    """
    Attempts to interpret move_str as UCI, then as SAN.
    Returns (uci_move_str, is_legal).
    """
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

# Example usage
positions = [
    chess.STARTING_FEN,
    # Add more FENs if you like
]

for fen in positions:
    move, logprobs = make_move_and_logprobs(fen)
    board = chess.Board(fen)
    move_uci, is_legal = get_legal_move_in_uci(board, move)
    print(f"FEN: {fen}")
    print(f"Raw model move: {move}")
    print(f"Move (UCI): {move_uci}")
    print(f"Legal: {is_legal}")
    print(f"Logprobs: {logprobs}")
    print("---")
