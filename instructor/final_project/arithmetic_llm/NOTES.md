# Arithmetic LLM Notes

Consolidated technical notes for tokenizer behavior, data loading, and evaluation.

## Tokenizer (arithmetic_tokenizer.py)

### Special tokens and atomic symbols
- Special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<think>`, `</think>`
- Atomic symbols (never merged): `+`, `-`, `(`, `)`, `:`

### Pre-tokenization and BPE merging rules
- Input is pre-tokenized by inserting spaces around atomic symbols, so operators and parentheses are always separate tokens.
- BPE merge statistics skip any adjacent pair that includes an atomic symbol, preventing merges like `((` or `(1`.

### BOS/EOS handling
- `encode(text, add_special_tokens=True)` adds `<bos>` and `<eos>` by default.
- `decode(token_ids, skip_special_tokens=True)` truncates at the first `<eos>` and removes special tokens from output.

### Compatibility note
- Any tokenizer behavior change (BOS/EOS or atomic symbol handling) requires retraining the tokenizer and downstream models.

## Data loading and masking (data_loader.py)

### Foundational mode
- Input sequence is `problem + solution`.
- All non-padding tokens contribute to loss.

### Instruction mode
- Prompt is `problem + " <think>"`; target is `solution`.
- `prompt_length` is computed **without** special tokens, then adjusted by +1 to account for `<bos>`.
- Loss is masked (`-100`) over the prompt tokens so only the solution contributes.

### Padding and labels
- Dynamic padding to the longest sequence in each batch.
- Padding uses `<pad>` token; padding positions are ignored in loss via `-100` labels.

## EOS truncation during generation
- `transformer_model.generate()` stops when `<eos>` is produced (or when max length is reached).
- Tokenizer `decode()` truncates anything after the first `<eos>`.

## Evaluation updates (evaluator.py / run_evaluation.py)

- Evaluation runs in batches via `_generate_batch()` for faster inference.
- CLI supports `--batch-size` and `--max-gen-length`.
- Single-sample generation remains available for interactive use.

## Expression evaluator fix (evaluator.py)

- Expressions with consecutive numbers separated by whitespace are rejected (e.g., `"6 18"`).
- Detection uses a pre-check like `\d\s+\d` before stripping spaces.

## Token table and inspection

- Print basic table: `python homeowrk/arithmetic_llm/print_token_table.py [N]`
- Export CSV/JSON: `python homeowrk/arithmetic_llm/print_token_table.py csv|json [N]`
- Detailed view and stats: `python homeowrk/arithmetic_llm/show_token_table.py`
