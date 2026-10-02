"""Build a small, entirely fictional person directory database.

    python examples/person_directory/build_example_db.py data/example_directory.db

Then set PERSON_DIRECTORY_DB to that path. All persons, units and
addresses are invented.
"""

import json
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).parent

PERSONS = [
    # pid, given, family, title, status, email, consent
    (1, "Mira", "Okonkwo", "Prof. Dr.", "Professor", "mira.okonkwo@example.edu", 1),
    (2, "Mira", "Lindqvist", "Dr.", "Research associate", "mira.lindqvist@example.edu", 1),
    (3, "Tomas", "Brandt", "", "Staff", "tomas.brandt@example.edu", 1),
    (4, "Jonas", "Weller", "Dr.", "Research associate", "jonas.weller@example.edu", 0),
]
ORGS = [
    ("O1", "Institute of Computational Linguistics",
     "Faculty of Humanities → Institute of Computational Linguistics"),
    ("O2", "Media Services", "Central Services → Media Services"),
]
PERSON_ORGS = [
    (1, "O1", "Head of institute", "Corpus linguistics", "Corpus linguistics", "+49 30 000-1001", "2.14", "Main building"),
    (2, "O1", "Researcher", "Speech technology", "Speech technology", "+49 30 000-1002", "2.20", "Main building"),
    (3, "O2", "Web developer", "Web services", "Web services", "+49 30 000-2001", "0.05", "Annex"),
    (4, "O1", "Researcher", "Language models", "Language models", "+49 30 000-1003", "2.21", "Main building"),
]
PAGES = [
    (10, 1, "https://www.example.edu/people/okonkwo", "Mira Okonkwo — profile",
     "Mira Okonkwo researches corpus linguistics and annotation methods.", "homepage"),
    (11, 4, "https://www.example.edu/people/weller", "Jonas Weller — profile",
     "Jonas Weller works on language models.", "homepage"),
]


def build(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    conn = sqlite3.connect(p)
    conn.executescript((HERE / "schema.sql").read_text(encoding="utf-8"))
    for pid, given, family, title, status, email, consent in PERSONS:
        conn.execute(
            "INSERT INTO persons (pid, given_name, family_name, academic_title, status, "
            "email, homepage_url, memberships, consent) VALUES (?,?,?,?,?,?,?,?,?)",
            (pid, given, family, title, status, email,
             f"https://www.example.edu/people/{family.lower()}",
             json.dumps(["Library committee"]) if pid == 1 else "", consent))
    conn.executemany("INSERT INTO orgs VALUES (?,?,?)", ORGS)
    conn.executemany(
        "INSERT INTO person_orgs (pid, org_id, role, subject_area, subject_area_en, "
        "phone, room, building) VALUES (?,?,?,?,?,?,?,?)", PERSON_ORGS)
    conn.executemany("INSERT INTO pages VALUES (?,?,?,?,?,?)", PAGES)
    # Index as the sync job would: persons and their pages, consent ignored
    # on purpose so that the connector's clean-up can be observed.
    for pid, given, family, *_ in PERSONS:
        org = next(o for o in PERSON_ORGS if o[0] == pid)
        path_ = next(x[2] for x in ORGS if x[0] == org[1])
        conn.execute(
            "INSERT INTO search_index (pid, page_id, person_name, org_path, subject_area, "
            "title, content) VALUES (?, NULL, ?, ?, ?, '', ?)",
            (pid, f"{given} {family}", path_, org[3], f"{given} {family} {org[2]} {org[3]}"))
    for page_id, pid, url, title, content, _ in PAGES:
        person = next(x for x in PERSONS if x[0] == pid)
        conn.execute(
            "INSERT INTO search_index (pid, page_id, person_name, org_path, subject_area, "
            "title, content) VALUES (?, ?, ?, '', '', ?, ?)",
            (pid, page_id, f"{person[1]} {person[2]}", title, content))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "data/example_directory.db")
    print("ok")
