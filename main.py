import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hey there! How's it going?"
tokens = enc.encode(text)

print(f"Text: {text}")
print(f"Tokens: {tokens}")

decode = enc.decode([25216, 1354, 0, 3253, 885, 480, 2966, 30])
print(f"Decoded: {decode}")