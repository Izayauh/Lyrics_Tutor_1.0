"""Bulk-download lyrics from the free LRCLIB API for RAG ingestion."""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "reference_lyrics"
API_URL = "https://lrclib.net/api/get"
DELAY_SECONDS = 1.5  # respect rate limits

# ---------------------------------------------------------------------------
# Song list – (Title, Artist)
# Populate the remaining songs as needed; the script handles any length.
# ---------------------------------------------------------------------------

SONGS: list[tuple[str, str]] = [
    # ── 5 seed songs (already downloaded) ─────────────────────────────────
    ("Yesterday", "The Beatles"),
    ("Passionfruit", "Drake"),
    ("Saint Pablo", "Kanye West"),
    ("Is This Love", "Bob Marley"),
    ("Ain't No Sunshine", "Bill Withers"),
    # ── remaining songs ────────────────────────────────────────────────────
    ("Head Over Heels", "Tears for Fears"),
    ("Marvin's Room", "Drake"),
    # "Everything I've Ever Wanted" by Tiffany Day is unreleased — saved manually
    ("Yukon", "Justin Bieber"),
    ("One More Light", "Linkin Park"),
    ("I Write Sins Not Tragedies", "Panic! at the Disco"),
    ("Churchill Downs", "Jack Harlow"),
    ("Alright", "Kendrick Lamar"),
    ("Depression & Obsession", "XXXTENTACION"),
    ("Save That Shit", "Lil Peep"),
    ("Alexander Hamilton", "Lin-Manuel Miranda"),
    ("The Violence", "Childish Gambino"),
    ("Free Mind", "Tems"),
    ("Gone", "Chiiild"),
    ("Too Sweet", "Hozier"),
    ("Dear Sahana", "Sid Sriram"),
    ("Runaway", "Kanye West"),
    ("Heartless", "Kanye West"),
    ("Pound Cake / Paris Morton Music 2", "Drake"),
    ("Somewhere Over the Rainbow", "Israel Kamakawiwo'ole"),
    ("Somewhere Only We Know", "Keane"),
    ("Fast Car", "Tracy Chapman"),
    ("Sweatpants", "Childish Gambino"),
    ("Primetime", "Jay-Z"),
    ("HYFR (Hell Ya Fucking Right)", "Drake"),
    ("Rocket Man", "Elton John"),
    ("Ms. Jackson", "OutKast"),
    ("Roses", "OutKast"),
    ("Mr. Brightside", "The Killers"),
    ("My Body", "Young the Giant"),
    ("I Wanna Be Yours", "Arctic Monkeys"),
    ("Chocolate", "The 1975"),
    ("Prey", "The Neighbourhood"),
    ("Sweater Weather", "The Neighbourhood"),
    ("Stop This Train", "John Mayer"),
    ("Hundred", "The Fray"),
    ("How to Save a Life", "The Fray"),
    ("Say When", "The Fray"),
    ("All Time Low", "Jon Bellion"),
    ("Guillotine", "Jon Bellion"),
    ("Me and Your Mama", "Childish Gambino"),
    ("Like Whoa", "Logic"),
    ("Pink Matter", "Frank Ocean"),
    ("When the Party's Over", "Billie Eilish"),
    ("Sex", "Eden"),
    ("Non-Stop", "Lin-Manuel Miranda"),
    ("Ultralight Beam", "Kanye West"),
    ("That Funny Feeling", "Bo Burnham"),
    ("Sparks", "Coldplay"),
    ("The Room Where It Happens", "Lin-Manuel Miranda"),
    ("Trust Me", "The Fray"),
    ("Love", "Keyshia Cole"),
    ("Tennessee Whiskey", "Chris Stapleton"),
    ("Vienna", "Billy Joel"),
    ("Habits (Stay High)", "Tove Lo"),  # "Habits" by Gary Clark Jr. not found; closest match
    ("Waving Through a Window", "Ben Platt"),
    ("Beautiful", "Yana"),
    ("Wash", "Jon Bellion"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Replace spaces with underscores and strip filesystem-unsafe chars."""
    name = name.replace(" ", "_")
    # Remove characters that are problematic on Windows / macOS / Linux
    name = re.sub(r'[\\/:*?"<>|]', "", name)
    return name


def fetch_lyrics(title: str, artist: str) -> str | None:
    """Query LRCLIB for plain lyrics. Returns None on 404 or missing data."""
    try:
        resp = requests.get(
            API_URL,
            params={"track_name": title, "artist_name": artist},
            timeout=15,
        )

        if resp.status_code == 404:
            return None

        resp.raise_for_status()
        data = resp.json()
        plain = data.get("plainLyrics")
        return plain if plain else None

    except requests.RequestException as exc:
        print(f"  ⚠  Network error for '{title}' by {artist}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    failed = 0
    total = len(SONGS)

    print(f"Fetching lyrics for {total} songs → {OUTPUT_DIR}\n")

    for idx, (title, artist) in enumerate(SONGS, start=1):
        filename = f"{sanitize_filename(title)}_{sanitize_filename(artist)}.txt"
        filepath = OUTPUT_DIR / filename

        # Resume support: skip files that already exist
        if filepath.exists():
            print(f"[{idx}/{total}] Already exists, skipping: {filename}")
            skipped += 1
            continue

        print(f"[{idx}/{total}] Fetching: {title} – {artist} ... ", end="", flush=True)
        lyrics = fetch_lyrics(title, artist)

        if lyrics is None:
            print("NOT FOUND")
            failed += 1
        else:
            # Write with the required RAG header
            filepath.write_text(
                f"Title: {title}\nArtist: {artist}\n\n{lyrics}\n",
                encoding="utf-8",
            )
            print("OK")
            downloaded += 1

        # Rate-limit delay (skip after the last song)
        if idx < total:
            time.sleep(DELAY_SECONDS)

    # Summary
    print(f"\n{'='*40}")
    print(f"Done!  Downloaded: {downloaded}  |  Skipped: {skipped}  |  Failed: {failed}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
