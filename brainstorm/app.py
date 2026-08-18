"""
AI Needs Brainstorm Tool
========================
A lightweight questionnaire app to run with different people and teams to
discover what tasks they do today and where AI agents / AI tools could help.

Participants log in with just their name + team, answer a set of guided
brainstorm questions (one idea at a time, as many as they like), and finish
on a "My Summary" page that compiles everything they submitted into a final
idea summary.

A facilitator overview at /overview shows every participant's answers grouped
by team, with a CSV export, so you can decide which agents / AI tools to
build for which people.

Run it standalone (separate from the main app):

    python brainstorm/app.py            # http://localhost:5001

No extra dependencies: Flask + stdlib sqlite3 only.
"""

import csv
import hashlib
import io
import json
import os
import re
import sqlite3
from datetime import datetime

from flask import (
    Flask, g, jsonify, redirect, render_template, request, session, url_for,
    Response, abort
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("BRAINSTORM_DB", os.path.join(BASE_DIR, "brainstorm.db"))

# Pick up API keys from the main app's .env (repo root) when available.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = os.environ.get("BRAINSTORM_SECRET_KEY", "ai-needs-brainstorm-dev-key")


# ---------------------------------------------------------------------------
# Questionnaire definition
# ---------------------------------------------------------------------------
# Each category is one screen. Fields are combined into a single readable
# idea line when displayed ("format" below controls how).

CATEGORIES = [
    {
        "slug": "front-back-stage",
        "nav": "Front Stage / Back Stage",
        "title": "What's your front stage?",
        "subtitle": "Front Stage = the work where you shine. Back Stage = everything that buries it.",
        "examples": [
            "closing deals in person, buried by data entry after every call",
            "designing new products, buried by status meetings",
        ],
        "fields": [
            {"name": "front_stage", "label": "Front stage", "placeholder": "My front stage (the brilliant work)...", "type": "textarea"},
            {"name": "back_stage", "label": "Back stage", "placeholder": "My back stage (what buries it)...", "type": "textarea"},
        ],
        "format": "{front_stage} — buried by: {back_stage}",
    },
    {
        "slug": "no-go-zones",
        "nav": "No-Go Zones",
        "title": "What's always human?",
        "subtitle": "The relationship, moment, or decision that should never be handed to AI.",
        "examples": [
            "letting someone go",
            "apologizing to a customer in person",
        ],
        "fields": [
            {"name": "always_human", "label": "Always human", "placeholder": "Always human in my role...", "type": "textarea"},
        ],
        "format": "{always_human}",
    },
    {
        "slug": "robot-task",
        "nav": "Repetitive Task",
        "title": "What's your repetitive task?",
        "subtitle": "Repetitive, rule-based, high-volume, or low-judgment work you do every week.",
        "examples": [
            "copying orders into the system, 4 hrs a week",
            "sending the same confirmation emails over and over",
        ],
        "fields": [
            {"name": "task", "label": "The task", "placeholder": "My repetitive task...", "type": "textarea"},
            {"name": "hours_per_week", "label": "Hours per week", "placeholder": "e.g. 3", "type": "text"},
        ],
        "format": "{task} ({hours_per_week} hrs/week)",
    },
    {
        "slug": "iron-man",
        "nav": "Iron Man Moment",
        "title": "Where's your Iron Man moment?",
        "subtitle": "A task where you're the expert but spend most time on prep before the decision.",
        "examples": [
            "pricing big jobs: hours of prep for a 15 min decision",
            "reviewing agreements: 40 pages for the 3 clauses that matter",
        ],
        "fields": [
            {"name": "task", "label": "The task", "placeholder": "The task where prep buries my judgment...", "type": "textarea"},
            {"name": "prep_time", "label": "Prep time", "placeholder": "e.g. 3 hours", "type": "text"},
            {"name": "decision_time", "label": "Decision / performance time", "placeholder": "e.g. 10 minutes", "type": "text"},
        ],
        "format": "{task} — {prep_time} prep → {decision_time} decision",
    },
    {
        "slug": "augmentation",
        "nav": "Human + Agent",
        "title": "Where would you and an agent team up?",
        "subtitle": "Work you'd keep doing yourself, but where an AI agent working alongside you would make you faster or better.",
        "examples": [
            "drafting proposals: the agent writes the first draft, I add the judgment",
            "customer visits: the agent preps the briefing, I run the meeting",
        ],
        "fields": [
            {"name": "task", "label": "The work", "placeholder": "The work we'd do together...", "type": "textarea"},
            {"name": "human_part", "label": "My part", "placeholder": "I'd keep doing...", "type": "textarea"},
            {"name": "agent_part", "label": "The agent's part", "placeholder": "The AI agent would...", "type": "textarea"},
        ],
        "format": "{task} — me: {human_part} · agent: {agent_part}",
    },
    {
        "slug": "who-would-you-hire",
        "nav": "Who Would You Hire?",
        "title": "Who would you hire?",
        "subtitle": "If you could hire one person to handle something you never want to think about again.",
        "examples": [
            "chase every quote we sent and never heard back on",
            "keep every listing and price up to date everywhere",
        ],
        "fields": [
            {"name": "what_they_would_do", "label": "What they would do", "placeholder": "I'd hire someone to...", "type": "textarea"},
            {"name": "how_to_measure", "label": "How to measure", "placeholder": "I'd know they're doing well if...", "type": "textarea"},
        ],
        "format": "{what_they_would_do} — {how_to_measure}",
    },
]

CATEGORY_BY_SLUG = {c["slug"]: c for c in CATEGORIES}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            team TEXT NOT NULL DEFAULT '',
            job_description TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            UNIQUE(name, team)
        );
        CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL REFERENCES participants(id),
            category TEXT NOT NULL,
            answers TEXT NOT NULL,   -- JSON of field name -> value
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ai_suggestions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            participant_id INTEGER NOT NULL REFERENCES participants(id),
            task TEXT NOT NULL,
            reason TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'new',   -- new | used | dismissed
            jd_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    # Migrate pre-existing databases created before job_description existed.
    try:
        db.execute("ALTER TABLE participants ADD COLUMN job_description TEXT NOT NULL DEFAULT ''")
    except sqlite3.OperationalError:
        pass
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def current_participant():
    pid = session.get("participant_id")
    if not pid:
        return None
    row = get_db().execute(
        "SELECT * FROM participants WHERE id = ?", (pid,)
    ).fetchone()
    return row


def format_submission(category, answers):
    """Render a submission's answers as one readable idea line."""
    try:
        return category["format"].format(**{k: (v or "?") for k, v in answers.items()})
    except (KeyError, IndexError):
        return " — ".join(v for v in answers.values() if v)


def submissions_for(participant_id):
    rows = get_db().execute(
        "SELECT * FROM submissions WHERE participant_id = ? ORDER BY id",
        (participant_id,),
    ).fetchall()
    grouped = {c["slug"]: [] for c in CATEGORIES}
    for row in rows:
        cat = CATEGORY_BY_SLUG.get(row["category"])
        if not cat:
            continue
        answers = json.loads(row["answers"])
        grouped[row["category"]].append(
            {
                "id": row["id"],
                "answers": answers,
                "text": format_submission(cat, answers),
                "time": row["created_at"],
            }
        )
    return grouped


def counts_for(participant_id):
    grouped = submissions_for(participant_id)
    return {slug: len(items) for slug, items in grouped.items()}


HOURS_RE = re.compile(r"(\d+(?:[.,]\d+)?)")


def parse_hours(value):
    if not value:
        return None
    m = HOURS_RE.search(str(value))
    if not m:
        return None
    return float(m.group(1).replace(",", "."))


def build_summary_stats(grouped):
    """Aggregate the participant's answers into final-summary stats."""
    total_ideas = sum(len(items) for items in grouped.values())
    robot_hours = 0.0
    robot_hours_known = False
    for item in grouped.get("robot-task", []):
        h = parse_hours(item["answers"].get("hours_per_week"))
        if h is not None:
            robot_hours += h
            robot_hours_known = True
    return {
        "total_ideas": total_ideas,
        "robot_hours": round(robot_hours, 1) if robot_hours_known else None,
        "automation_candidates": len(grouped.get("robot-task", []))
        + len(grouped.get("who-would-you-hire", [])),
        "copilot_candidates": len(grouped.get("iron-man", []))
        + len(grouped.get("augmentation", [])),
        "no_go_count": len(grouped.get("no-go-zones", [])),
    }


# ---------------------------------------------------------------------------
# Human + Agent idea suggestions
# ---------------------------------------------------------------------------
# Each repetitive task the participant submitted is turned into a suggested
# human/agent split. Keyword rules pick a split that fits the kind of work;
# anything unmatched gets a sensible generic split.

AUGMENTATION_RULES = [
    {
        "keywords": ["email", "e-mail", "inbox", "reply", "replying", "respond"],
        "human_part": "review and send the replies that matter, handle the sensitive ones myself",
        "agent_part": "sort the inbox, draft replies for me to approve, and answer the routine ones",
    },
    {
        "keywords": ["meeting", "schedul", "calendar", "appointment", "booking"],
        "human_part": "decide which meetings are worth my time",
        "agent_part": "find slots, send invites, reschedule conflicts, and prepare a short agenda",
    },
    {
        "keywords": ["report", "dashboard", "summar", "status update", "presentation"],
        "human_part": "check the numbers and add my conclusions and recommendations",
        "agent_part": "gather the data and produce the first full draft",
    },
    {
        "keywords": ["invoice", "billing", "purchase order", " po ", "pos ", "payment"],
        "human_part": "approve them and handle disputes",
        "agent_part": "prepare and enter them, chase what's missing, and flag mismatches",
    },
    {
        "keywords": ["data entry", "copying", "copy ", "entering", "typing", "crm", "update the system", "spreadsheet"],
        "human_part": "spot-check the exceptions it flags",
        "agent_part": "do the entry end to end and keep every system in sync",
    },
    {
        "keywords": ["quote", "proposal", "offer", "tender", "pricing"],
        "human_part": "set the price and add the judgment and relationship touch",
        "agent_part": "draft it from our templates and past examples with all the standard parts filled in",
    },
    {
        "keywords": ["follow up", "follow-up", "chase", "chasing", "remind"],
        "human_part": "step in when a reply needs a human touch",
        "agent_part": "send and track every follow-up until it's answered",
    },
    {
        "keywords": ["document", "paperwork", "admin", "filing", "form", "contract"],
        "human_part": "review and sign off",
        "agent_part": "fill in, file, and organize the documents, flagging anything unusual",
    },
]

GENERIC_AUGMENTATION = {
    "human_part": "handle the exceptions and give final approval",
    "agent_part": "do the routine part end to end and flag anything unusual",
}


def _normalize(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def suggest_augmentations(grouped):
    """Build Human + Agent idea suggestions from the repetitive tasks."""
    already_used = {
        _normalize(item["answers"].get("task"))
        for item in grouped.get("augmentation", [])
    }
    suggestions = []
    for item in grouped.get("robot-task", []):
        task = (item["answers"].get("task") or "").strip()
        if not task or _normalize(task) in already_used:
            continue
        haystack = " " + _normalize(task) + " "
        split = GENERIC_AUGMENTATION
        for rule in AUGMENTATION_RULES:
            if any(kw in haystack for kw in rule["keywords"]):
                split = rule
                break
        suggestions.append(
            {
                "task": task,
                "human_part": split["human_part"],
                "agent_part": split["agent_part"],
            }
        )
    return suggestions


# ---------------------------------------------------------------------------
# AI task discovery (Gemini)
# ---------------------------------------------------------------------------
# From the participant's job description (plus the tasks they already
# reported), Gemini suggests recurring tasks they very likely also do but
# did not report ("Are you also doing this?"). Suggestions are cached per
# participant and regenerated only when the job description changes.

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

DISCOVERY_PROMPT = """You help companies discover employee tasks that could be supported by AI.
A person described their job like this:

JOB DESCRIPTION:
{job_description}

TASKS THEY ALREADY REPORTED DOING:
{reported_tasks}

List 5 to 8 OTHER concrete, recurring work tasks that someone with this job
very likely ALSO does but did not report — think of admin work, communication,
coordination, chasing people, reporting, data upkeep, preparation work.
Do NOT repeat or rephrase any reported task. Keep each task short (max 12
words), written in first person, e.g. "Preparing the weekly sales report".

Reply ONLY with a JSON array, no other text, in this exact shape:
[{{"task": "...", "reason": "one short sentence on why people in this role usually do this"}}]
"""


def gemini_api_key():
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""


def call_gemini(prompt):
    """Send one prompt to Gemini and return the raw response text.

    Kept as a separate function so tests can stub it out.
    """
    import google.generativeai as genai

    genai.configure(api_key=gemini_api_key())
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)
    return response.text


def parse_suggestion_json(text):
    """Extract the JSON array of suggestions from a model response."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return []
    suggestions = []
    for item in data:
        if isinstance(item, dict) and (item.get("task") or "").strip():
            suggestions.append(
                {
                    "task": str(item["task"]).strip(),
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
    return suggestions[:8]


def jd_hash(job_description):
    return hashlib.md5(_normalize(job_description).encode("utf-8")).hexdigest()


def all_reported_tasks(grouped):
    """Every idea line the participant submitted, across all categories."""
    lines = []
    for items in grouped.values():
        for item in items:
            lines.append(item["text"])
    return lines


def get_task_discoveries(participant):
    """Return cached (or freshly generated) task suggestions for a participant.

    Returns (suggestions, error_message). suggestions is a list of rows with
    id/task/reason; error_message is set when generation was attempted but
    failed.
    """
    job_description = (participant["job_description"] or "").strip()
    if not job_description or not gemini_api_key():
        return [], None

    db = get_db()
    current_hash = jd_hash(job_description)
    cached = db.execute(
        "SELECT * FROM ai_suggestions WHERE participant_id = ? AND jd_hash = ?",
        (participant["id"], current_hash),
    ).fetchall()
    if cached:
        return [dict(row) for row in cached if row["status"] == "new"], None

    # Job description is new or changed: drop stale pending suggestions
    # (keep used/dismissed history so we don't re-suggest those tasks).
    db.execute(
        "DELETE FROM ai_suggestions WHERE participant_id = ? AND status = 'new'",
        (participant["id"],),
    )

    grouped = submissions_for(participant["id"])
    reported = all_reported_tasks(grouped)
    handled = db.execute(
        "SELECT task FROM ai_suggestions WHERE participant_id = ?",
        (participant["id"],),
    ).fetchall()
    reported += [row["task"] for row in handled]

    prompt = DISCOVERY_PROMPT.format(
        job_description=job_description,
        reported_tasks="\n".join(f"- {t}" for t in reported) or "- (none reported yet)",
    )
    try:
        suggestions = parse_suggestion_json(call_gemini(prompt))
    except Exception:
        return [], "Could not reach the AI service right now — please try again later."

    already_known = {_normalize(t) for t in reported}
    for items in grouped.values():
        for item in items:
            for value in item["answers"].values():
                if value:
                    already_known.add(_normalize(value))
    already_known.discard("")

    def is_duplicate(task):
        t = _normalize(task)
        return any(t in known or known in t for known in already_known)

    now = datetime.now().isoformat(timespec="seconds")
    fresh = []
    for s in suggestions:
        if is_duplicate(s["task"]):
            continue
        cur = db.execute(
            "INSERT INTO ai_suggestions (participant_id, task, reason, status, jd_hash, created_at)"
            " VALUES (?, ?, ?, 'new', ?, ?)",
            (participant["id"], s["task"], s["reason"], current_hash, now),
        )
        fresh.append({"id": cur.lastrowid, "task": s["task"], "reason": s["reason"], "status": "new"})
    db.commit()
    return fresh, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        team = (request.form.get("team") or "").strip()
        job_description = (request.form.get("job_description") or "").strip()
        if not name:
            return render_template("login.html", error="Please enter your name.")
        db = get_db()
        row = db.execute(
            "SELECT id FROM participants WHERE name = ? AND team = ?", (name, team)
        ).fetchone()
        if row:
            pid = row["id"]
            if job_description:
                db.execute(
                    "UPDATE participants SET job_description = ? WHERE id = ?",
                    (job_description, pid),
                )
                db.commit()
        else:
            cur = db.execute(
                "INSERT INTO participants (name, team, job_description, created_at)"
                " VALUES (?, ?, ?, ?)",
                (name, team, job_description, datetime.now().isoformat(timespec="seconds")),
            )
            db.commit()
            pid = cur.lastrowid
        session["participant_id"] = pid
        return redirect(url_for("category", slug=CATEGORIES[0]["slug"]))
    if current_participant():
        return redirect(url_for("category", slug=CATEGORIES[0]["slug"]))
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.pop("participant_id", None)
    return redirect(url_for("login"))


@app.route("/q/<slug>", methods=["GET", "POST"])
def category(slug):
    participant = current_participant()
    if not participant:
        return redirect(url_for("login"))
    cat = CATEGORY_BY_SLUG.get(slug)
    if not cat:
        abort(404)

    if request.method == "POST":
        answers = {}
        has_content = False
        for field in cat["fields"]:
            value = (request.form.get(field["name"]) or "").strip()
            answers[field["name"]] = value
            if value:
                has_content = True
        if has_content:
            db = get_db()
            db.execute(
                "INSERT INTO submissions (participant_id, category, answers, created_at)"
                " VALUES (?, ?, ?, ?)",
                (
                    participant["id"],
                    slug,
                    json.dumps(answers),
                    datetime.now().strftime("%I:%M %p"),
                ),
            )
            db.commit()
        return redirect(url_for("category", slug=slug))

    grouped = submissions_for(participant["id"])
    suggestions = suggest_augmentations(grouped) if slug == "augmentation" else []
    return render_template(
        "category.html",
        participant=participant,
        categories=CATEGORIES,
        counts=counts_for(participant["id"]),
        cat=cat,
        items=grouped[slug],
        suggestions=suggestions,
        active=slug,
    )


@app.route("/q/<slug>/delete/<int:submission_id>", methods=["POST"])
def delete_submission(slug, submission_id):
    participant = current_participant()
    if not participant:
        return redirect(url_for("login"))
    db = get_db()
    db.execute(
        "DELETE FROM submissions WHERE id = ? AND participant_id = ?",
        (submission_id, participant["id"]),
    )
    db.commit()
    return redirect(url_for("category", slug=slug))


@app.route("/api/discover")
def api_discover():
    """AI-suggested tasks from the participant's job description."""
    participant = current_participant()
    if not participant:
        return jsonify({"available": False, "message": "Not logged in."}), 401
    if not (participant["job_description"] or "").strip():
        return jsonify({"available": False, "message": "no-job-description"})
    if not gemini_api_key():
        return jsonify({"available": False, "message": "no-api-key"})
    suggestions, error = get_task_discoveries(participant)
    if error:
        return jsonify({"available": False, "message": error})
    return jsonify(
        {
            "available": True,
            "suggestions": [
                {"id": s["id"], "task": s["task"], "reason": s["reason"]}
                for s in suggestions
            ],
        }
    )


@app.route("/api/discover/<int:suggestion_id>/<action>", methods=["POST"])
def api_discover_action(suggestion_id, action):
    """Mark an AI task suggestion as used (added) or dismissed (not me)."""
    participant = current_participant()
    if not participant:
        return jsonify({"ok": False}), 401
    if action not in ("used", "dismissed"):
        abort(404)
    db = get_db()
    db.execute(
        "UPDATE ai_suggestions SET status = ? WHERE id = ? AND participant_id = ?",
        (action, suggestion_id, participant["id"]),
    )
    db.commit()
    return jsonify({"ok": True})


@app.route("/summary")
def summary():
    """Final idea summary for the logged-in participant."""
    participant = current_participant()
    if not participant:
        return redirect(url_for("login"))
    grouped = submissions_for(participant["id"])
    stats = build_summary_stats(grouped)
    return render_template(
        "summary.html",
        participant=participant,
        categories=CATEGORIES,
        counts=counts_for(participant["id"]),
        grouped=grouped,
        stats=stats,
        active="summary",
    )


@app.route("/overview")
def overview():
    """Facilitator view: everyone's answers grouped by team."""
    db = get_db()
    participants = db.execute(
        "SELECT * FROM participants ORDER BY team, name"
    ).fetchall()
    teams = {}
    total_ideas = 0
    for p in participants:
        grouped = submissions_for(p["id"])
        stats = build_summary_stats(grouped)
        total_ideas += stats["total_ideas"]
        teams.setdefault(p["team"] or "No team", []).append(
            {"participant": p, "grouped": grouped, "stats": stats}
        )
    return render_template(
        "overview.html",
        categories=CATEGORIES,
        teams=teams,
        participant_count=len(participants),
        total_ideas=total_ideas,
    )


@app.route("/overview.csv")
def overview_csv():
    db = get_db()
    rows = db.execute(
        """
        SELECT p.name, p.team, s.category, s.answers, s.created_at
        FROM submissions s JOIN participants p ON p.id = s.participant_id
        ORDER BY p.team, p.name, s.id
        """
    ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Team", "Name", "Category", "Idea", "Details", "Submitted"])
    for row in rows:
        cat = CATEGORY_BY_SLUG.get(row["category"])
        if not cat:
            continue
        answers = json.loads(row["answers"])
        details = "; ".join(
            f"{f['label']}: {answers.get(f['name'], '')}" for f in cat["fields"]
        )
        writer.writerow(
            [
                row["team"],
                row["name"],
                cat["nav"],
                format_submission(cat, answers),
                details,
                row["created_at"],
            ]
        )
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=ai-brainstorm-ideas.csv"},
    )


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("BRAINSTORM_PORT", 5001)), debug=False)
