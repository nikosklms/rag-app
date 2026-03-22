# 🎯 Features TODO — Κάνε τα μόνος σου!

Τρία features για να γράψεις **εσύ** τον κώδικα και να μάθεις σε βάθος το project.

---

## 1. Multi-format Support (🟢 Εύκολο)

**Στόχος:** Πρόσθεσε υποστήριξη `.md` (markdown) αρχείων.

**Πού πειράζεις:**
- `src/ingestion/parser.py` — Νέα function `parse_markdown()`
- `src/main.py` — Πρόσθεσε `.md` στα accepted file types

**Βήματα:**
1. Γράψε `parse_markdown()` στο `parser.py` — διάβασε το αρχείο, αφαίρεσε markdown syntax (`#`, `**`, `- `, `` ` `` κλπ). Μπορείς με regex ή απλά `.replace()`
2. Πρόσθεσε `".md"` στο `parse_file()` dispatcher
3. Στο `main.py`, πρόσθεσε `".md"` στη λίστα accepted extensions
4. Δοκίμασε: ανέβασε ένα `.md` αρχείο και ρώτα κάτι

**Hints:**
- Κοίτα πώς δουλεύει η `parse_txt()` — η δική σου θα είναι σχεδόν ίδια
- `import re` → `re.sub(r'#{1,6}\s', '', text)` αφαιρεί headings

---

## 2. Chat History / Conversation Memory (🟡 Μεσαίο)

**Στόχος:** Ο LLM να θυμάται τα προηγούμενα μηνύματα. Π.χ. "Πες μου για το project X" → "Εξήγησέ μου περισσότερα" → ο LLM ξέρει ποιο "X".

**Πού πειράζεις:**
- `src/models/schemas.py` — Πρόσθεσε πεδίο `history` στο `QueryRequest`
- `src/generation/generator.py` — Βάλε τα previous messages στο prompt
- `frontend/app.py` — Στείλε τα messages μαζί με κάθε ερώτηση
- `src/main.py` — Πέρνα το history στον generator

**Βήματα:**
1. Στο `schemas.py`: πρόσθεσε `history: list[dict] = []` στο `QueryRequest`
2. Στο `generator.py`: φτιάξε function που παίρνει `history` (list of `{"role": "user"/"assistant", "content": "..."}`) και τα βάζει στα messages πριν το τελευταίο query
3. Στο `main.py`: πέρνα `request.history` στο `generate()`
4. Στο `app.py`: κάθε φορά που στέλνεις ερώτηση, βάλε και `st.session_state.messages` στο JSON body

**Hints:**
- Κοίτα πώς δημιουργείται το prompt στο `_build_prompt()` — θα πρέπει τα previous messages να μπλέπονται ΠΡΙΝ το τελευταίο query
- Πρόσεξε: μην στέλνεις πάρα πολλά messages (π.χ. κράτα μόνο τα τελευταία 10)

---

## 3. Streaming Responses — SSE (🔴 Δύσκολο)

**Στόχος:** Η απάντηση εμφανίζεται token-by-token (σαν ChatGPT), αντί να περιμένεις να τελειώσει.

**Πού πειράζεις:**
- `src/generation/generator.py` — Streaming call στο Ollama API
- `src/main.py` — Νέο endpoint `/query/ask/stream` με `StreamingResponse`
- `frontend/app.py` — Progressive display

**Βήματα:**
1. Στο `generator.py`: φτιάξε `generate_stream()` που καλεί Ollama με `"stream": True`. To Ollama API επιστρέφει line-by-line JSON — κάθε line έχει ένα token. Χρησιμοποίησε `yield` (Python generator)
2. Στο `main.py`: φτιάξε νέο endpoint `POST /query/ask/stream` που επιστρέφει `StreamingResponse` (from `fastapi.responses`)
3. Στο `app.py`: αντί `httpx.post()`, χρησιμοποίησε `httpx.stream()` και γράψε κάθε token σε `st.write_stream()` ή `st.empty()` + progressive update

**Hints:**
- Docs Ollama streaming: https://github.com/ollama/ollama/blob/main/docs/api.md — δες πώς δουλεύει `"stream": true`
- `from fastapi.responses import StreamingResponse`
- `yield f"data: {token}\n\n"` ← αυτό είναι Server-Sent Events (SSE) format
- Το Streamlit `st.write_stream()` δέχεται generator — ιδανικό

---

## Σειρά εκτέλεσης: 1 → 2 → 3

Κάθε feature χτίζει πάνω στην εμπειρία του προηγούμενου. Καλή τύχη! 💪
