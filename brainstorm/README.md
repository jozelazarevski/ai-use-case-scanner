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
   - **Who Would You Hire?** — the one thing they'd delegate forever (with a success measure)
3. **Final idea summary** — at the end, each participant sees a "My Idea Summary" page
   that compiles everything they submitted, with headline stats (total ideas, AI-agent
   candidates, copilot candidates, hours/week recoverable) and a printable layout.
4. **Facilitator overview** — `/overview` shows every participant's answers grouped by
   team, with a CSV export (`/overview.csv`) for further analysis.

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
