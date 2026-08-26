# PDF Explainer Chat App

Upload a text-based PDF and get a structured explanation from Gemini. This small Version 1 app runs entirely on your computer; it does not use Docker, RAG, a database, or a separate frontend server.

## Run it locally

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1`.

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey). Copy `.env.example` to `.env`, then replace `your_gemini_key_here` with that key. Do not commit `.env`.

4. Start the app:

   ```bash
   uvicorn main:app --reload
   ```

5. Open http://127.0.0.1:8000 in your browser, choose a PDF, and send an optional instruction. The app keeps that PDF in local server memory for the current browser chat, so later questions do not need another upload. Uploading another PDF replaces it. Restarting the server or refreshing the browser starts a new chat.

Scanned PDFs need OCR before they can be explained because this Version 1 extracts only selectable PDF text.

The app uses `gemini-3.6-flash` through Gemini's free tier and does not enable billing-only tools such as Google Search grounding. It sends up to 500,000 extracted PDF characters per request. This avoids failures with exceptionally large PDFs while allowing far more text than the earlier 60,000-character limit. Free-tier use has rate limits; if you reach one, wait for the quota to reset instead of enabling billing.

## Multimodal RAG, agent, and evaluation

Each uploaded PDF follows this in-memory pipeline:

```
PDF → text chunks / detected tables / small embedded figures
    → Gemini Embedding 2 vectors → in-memory cosine vector store
    → query-routing agent → text/table/image retrieval
    → Gemini multimodal answer → charts, citations, and RAG evaluation
```

There is no Docker, disk-backed vector database, local embedding model, or uploaded-file cache. The extracted content, figure bytes, vectors, and chat history reside only in RAM and disappear when the server restarts. To keep memory use modest, a document is capped at 500,000 text characters, 120 text chunks, 20 tables, 8 supported figures, and 2 MB of extracted figure data.

The routing agent selects text RAG by default, adds table RAG for data-oriented questions, and adds image RAG for questions about figures, diagrams, or visuals. Answers list the retrieved PDF sources they cite and show a lightweight grounding evaluation based on retrieval strength and citation coverage.

## Chat and charts

The app retains the eight most recent user/assistant entries for the current browser session, which makes questions such as “explain that more simply” work naturally. Gemini is asked to create at most two bar or line chart specifications only when the retrieved PDF sources contain clear, explicit numerical data. The browser renders those charts locally as SVG; it never calls a charting service or makes up data.
