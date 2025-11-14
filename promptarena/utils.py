import re

def trim_lines(string: str, n_chars: int):
    newline = string.find('\n')
    if 0 <= newline < n_chars:
        return string[:newline]
    return string[:n_chars-3] + "..." if len(string) > n_chars else string

def make_filename_safe(string: str):
    return re.sub(r'[^A-Za-z0-9._-]', '_', string)