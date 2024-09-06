# Lexical Analyzer

This project implements a lexical analyzer (lexer) that can tokenize input based on regular expression specifications.

## Author

Rodolfo Lopez

## Date

November 24, 2021

## Files

- `pa4.py`: Main implementation of the lexical analyzer
- `test_pa4.py`: Test script to validate the lexer functionality
- `regex*.txt`: Regular expression specification files
- `src*.txt`: Source input files to tokenize
- `correct*.txt`: Expected tokenization results

## Usage

To run the lexer on a specific input:

```python
reg_ex_filename = "regex1.txt"
source_filename = "src1.txt"
lex = Lex(reg_ex_filename, source_filename)
try:
    while True:
        token = lex.next_token()
        print(token)
except EOFError:
    pass
except InvalidToken:
    print("Invalid token")
```

## Key Components

- `Lex`: Main lexer class that reads regex specs and tokenizes input
- `RegEx`: Converts regex to NFA to DFA
- `NFA`: Non-deterministic finite automaton implementation
- `DFA`: Deterministic finite automaton for efficient token matching

## Testing

Run `test_pa4.py` to execute the test suite, which validates the lexer against multiple input/output pairs.

## Notes

- Supports common regex operations like union, concatenation, and Kleene star
- Handles epsilon transitions when converting NFA to DFA
- Raises `InvalidToken` exception for unrecognized input

## Acknowledgments

Professor John Glick wrote all test scripts.
