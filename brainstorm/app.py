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
import io
import os
import re
import sqlite3
from datetime import datetime

from flask import (
    Flask, g, redirect, render_template, request, session, url_for,
    Response, abort
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("BRAINSTORM_DB", os.path.join(BASE_DIR, "brainstorm.db"))

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
        """
    )
    db.commit()
    db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import json


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
        "copilot_candidates": len(grouped.get("iron-man", [])),
        "no_go_count": len(grouped.get("no-go-zones", [])),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        team = (request.form.get("team") or "").strip()
        if not name:
            return render_template("login.html", error="Please enter your name.")
        db = get_db()
        row = db.execute(
            "SELECT id FROM participants WHERE name = ? AND team = ?", (name, team)
        ).fetchone()
        if row:
            pid = row["id"]
        else:
            cur = db.execute(
                "INSERT INTO participants (name, team, created_at) VALUES (?, ?, ?)",
                (name, team, datetime.now().isoformat(timespec="seconds")),
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
    return render_template(
        "category.html",
        participant=participant,
        categories=CATEGORIES,
        counts=counts_for(participant["id"]),
        cat=cat,
        items=grouped[slug],
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
