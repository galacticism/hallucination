import chess
import csv
import os
import time

# Uncomment and set your API keys here
OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
ANTHROPIC_API_KEY = 'YOUR_ANTHROPIC_API_KEY'
COHERE_API_KEY = 'YOUR_COHERE_API_KEY'

# Choose your provider: "openai", "anthropic", "cohere"
PROVIDER = "openai"

# Load SDKs
if PROVIDER == "openai":
    import openai
    openai.api_key = OPENAI_API_KEY
elif PROVIDER == "anthropic":
    import anthropic
    client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
elif PROVIDER == "cohere":
    import cohere
    co = cohere.Client(COHERE_API_KEY)
else:
    raise ValueError("Unknown provider!")

def get_move_openai(fen):
    prompt = (
        f"The following is a chess position in FEN format:\n{fen}\n"
        "What is the best move for the player to move? Give the move in UCI format (e.g., e2e4, g1f3):\n"
    )
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=5,
        temperature=0.7,
        logprobs=5,
        n=1,
        stop=["\n"]
    )
    move_str = response['choices'][0]['text'].strip().split()[0]
    logprob = response['choices'][0]['logprobs']['token_logprobs'][0]
    return move_str, logprob

def get_move_anthropic(fen):
    prompt = (
        f"The following is a chess position in FEN format:\n{fen}\n"
        "What is the best move for the player to move? Give the move in UCI format (e.g., e2e4, g1f3):"
    )
    # Claude requires a system and user message for best results
    message = client_anthropic.messages.create(
        model="claude-3-opus-20240229",  # Or "claude-3-sonnet-20240229" for lower cost
        max_tokens=5,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ],
        top_p=1,
        stop_sequences=["\n"],
        logprobs=True,  # This returns token-level logprobs
    )
    text = message.content[0].text.strip().split()[0]
    logprob = message.usage.output_tokens[0].logprob if message.usage.output_tokens else None
    # Above line gets the logprob for the first output token. Adjust as needed for your analysis.
    return text, logprob

def get_move_cohere(fen):
    prompt = (
        f"The following is a chess position in FEN format:\n{fen}\n"
        "What is the best move for the player to move? Give the move in UCI format (e.g., e2e4, g1f3):"
    )
    response = co.generate(
        model="command-r", # or "command" or your preferred model
        prompt=prompt,
        max_tokens=5,
        temperature=0.7,
        return_likelihoods="GENERATION",
        stop_sequences=["\n"]
    )
    move_str = response.generations[0].text.strip().split()[0]
    logprob = response.generations[0].token_likelihoods[0].likelihood if response.generations[0].token_likelihoods else None
    # Cohere returns actual probabilities; you may want to log-transform them if needed.
    return move_str, logprob

def make_move(fen):
    if PROVIDER == "openai":
        return get_move_openai(fen)
    elif PROVIDER == "anthropic":
        return get_move_anthropic(fen)
    elif PROVIDER == "cohere":
        return get_move_cohere(fen)
    else:
        raise ValueError("Unknown provider!")

# List of FENs to try
positions = [
    chess.STARTING_FEN,
    # Add more positions as needed
]

results = []

for fen in positions:
    board = chess.Board(fen)
    move_str, logprob = make_move(fen)
    is_legal = False
    try:
        move = chess.Move.from_uci(move_str)
        is_legal = move in board.legal_moves
    except Exception:
        is_legal = False

    results.append({
        "fen": fen,
        "move": move_str,
        "logprob": logprob,
        "is_legal": is_legal,
    })
    print(f"FEN: {fen}\nMove: {move_str}\nLegal: {is_legal}\nLogprob: {logprob}\n---")
    time.sleep(1)  # Be polite to the API!

with open("llm_chess_results.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["fen", "move", "logprob", "is_legal"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)
