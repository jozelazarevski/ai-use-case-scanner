# AI Needs Brainstorm Tool

A lightweight questionnaire app to run with different people and teams to discover
what tasks fill their week and where AI agents / AI tools could help them — so you
can build the right agents for the right people.

## How it works

1. **Participants join** with just their name and team — no password needed.
2. **They brainstorm one idea at a time** across five guided categories:
   - **Front Stage / Back Stage** — the work where they shine vs. what buries it
   - **No-Go Zones** — what should always stay human
   - **Repetitive Task** — repetitive weekly work (with hours/week)
   - **Iron Man Moment** — expert decisions buried under prep time
   - **Human + Agent** — augmentation ideas: work they'd keep, with an AI agent alongside
     (my part vs. the agent's part). This screen also **auto-suggests ideas**: every
     repetitive task the participant submitted is turned into a suggested human/agent
     split (keyword rules pick a fitting split — emails, scheduling, reports, invoices,
     data entry, quotes, follow-ups, paperwork — with a generic split as fallback).
     Clicking a suggestion pre-fills the form so they can adjust and submit it.
   - **Who Would You Hire?** — the one thing they'd delegate forever (with a success measure)
3. **Final idea summary** — at the end, each participant sees a "My Idea Summary" page
   that compiles everything they submitted, with headline stats (total ideas, AI-agent
   candidates, copilot candidates, hours/week recoverable) and a printable layout.
4. **Facilitator overview** — `/overview` shows every participant's answers grouped by
   team, with a CSV export (`/overview.csv`) for further analysis.

## AI task discovery (Gemini)

On the join page participants can optionally describe their job in their own
words. If a Gemini API key is configured, the Repetitive Task screen then shows
an **"Are you also doing this?"** panel: Gemini reads the job description (plus
everything already reported) and suggests recurring tasks the person likely
also does but forgot to mention. Each suggestion can be accepted — which
pre-fills the form so they add it with their own hours — or dismissed with
"Not me". Suggestions are cached per participant and only regenerated when the
job description changes; accepted/dismissed ones are never suggested again.

To enable it, set one of these in the environment (or `.env` of the main app):

- `GOOGLE_API_KEY` or `GEMINI_API_KEY` — your Gemini API key
- `GEMINI_MODEL` — optional, defaults to `gemini-2.0-flash`

Without a key the app works normally — the panel simply doesn't appear.

## Run it

```bash
python brainstorm/app.py
```

Then open http://localhost:5001 and share the link with your teams.

Options (environment variables):

- `BRAINSTORM_PORT` — port to listen on (default `5001`)
- `BRAINSTORM_DB` — path to the SQLite database file (default `brainstorm/brainstorm.db`)
- `BRAINSTORM_SECRET_KEY` — Flask session secret (set this in production)

No extra dependencies beyond Flask (already in `requirements.txt`); data is stored
in a local SQLite database created automatically on first run.
