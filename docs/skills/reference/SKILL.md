---
name: reference
description: Reference code folder conventions — storing external code and documentation for AI-assisted development context
---

# Reference Code

`docs/reference/` stores reference external code and documentation. These files serve as helpful context for AI-assisted code generation and development — you can attach them to your conversation so the AI can draw on external codebases.

## Conventions

- The `check-added-large-files` pre-commit hook is disabled in this folder, so large files can be committed freely.
- Always include an accompanying `README.md` entry in `docs/reference/` to document each file:

```
- file-name.txt
   - Source: name of external source
   - Content: brief description of what this contains
```
