"""
mock_schedule.py

Mock doctor availability data for the Appointment Agent to check against.
In the real deployment (Phase 5) this gets replaced by an n8n webhook call
that reads/writes a live Google Sheet — but keeping it as a simple in-memory
structure here means the Appointment Agent's logic can be built and tested
in isolation first, same philosophy as building RAG before wiring voice.

Departments match the ones already referenced in the FAQ/Billing knowledge
base (cardiology, dental, etc.) so the whole demo stays internally consistent.
"""

DEPARTMENTS = ["Cardiology", "Dental", "Pediatrics", "General Medicine", "Orthopedics"]

# Each slot: date (YYYY-MM-DD), time, doctor name, and whether it's already booked.
# Dates are set relative to mid-July 2026 so the demo always looks "current."
SCHEDULE = {
    "Cardiology": [
        {"date": "2026-07-15", "time": "10:00 AM", "doctor": "Dr. Ayesha Raza", "booked": False},
        {"date": "2026-07-15", "time": "2:00 PM", "doctor": "Dr. Ayesha Raza", "booked": True},
        {"date": "2026-07-16", "time": "11:30 AM", "doctor": "Dr. Farhan Malik", "booked": False},
        {"date": "2026-07-17", "time": "9:00 AM", "doctor": "Dr. Ayesha Raza", "booked": False},
    ],
    "Dental": [
        {"date": "2026-07-14", "time": "9:00 AM", "doctor": "Dr. Sara Khan", "booked": False},
        {"date": "2026-07-14", "time": "3:00 PM", "doctor": "Dr. Sara Khan", "booked": False},
        {"date": "2026-07-18", "time": "10:00 AM", "doctor": "Dr. Bilal Ahmed", "booked": True},
    ],
    "Pediatrics": [
        {"date": "2026-07-15", "time": "1:00 PM", "doctor": "Dr. Nadia Iqbal", "booked": False},
        {"date": "2026-07-16", "time": "4:00 PM", "doctor": "Dr. Nadia Iqbal", "booked": False},
    ],
    "General Medicine": [
        {"date": "2026-07-14", "time": "11:00 AM", "doctor": "Dr. Omar Sheikh", "booked": False},
        {"date": "2026-07-15", "time": "9:30 AM", "doctor": "Dr. Omar Sheikh", "booked": False},
        {"date": "2026-07-15", "time": "2:30 PM", "doctor": "Dr. Hina Tariq", "booked": False},
    ],
    "Orthopedics": [
        {"date": "2026-07-17", "time": "10:30 AM", "doctor": "Dr. Usman Ghani", "booked": False},
    ],
}


def normalize_department(raw: str):
    """
    Caller might say 'cardiologist', 'cardiology', 'dental department', etc.
    This does a loose case-insensitive match against known departments instead
    of requiring an exact string — keeps the demo forgiving of natural speech.
    """
    if not raw:
        return None
    raw_lower = raw.lower()
    for dept in DEPARTMENTS:
        if dept.lower() in raw_lower or raw_lower in dept.lower():
            return dept
    # Handle common variants not caught by substring match
    if "heart" in raw_lower or "cardiologist" in raw_lower:
        return "Cardiology"
    if "tooth" in raw_lower or "teeth" in raw_lower or "dentist" in raw_lower:
        return "Dental"
    if "kid" in raw_lower or "child" in raw_lower or "pediatrician" in raw_lower:
        return "Pediatrics"
    if "bone" in raw_lower or "joint" in raw_lower or "orthopedist" in raw_lower or "orthopedic" in raw_lower:
        return "Orthopedics"
    if "physician" in raw_lower or "general practitioner" in raw_lower or "gp" in raw_lower:
        return "General Medicine"
    return None


def get_available_slots(department: str):
    """Returns all NOT booked slots for a department, sorted by date."""
    slots = SCHEDULE.get(department, [])
    available = [s for s in slots if not s["booked"]]
    return sorted(available, key=lambda s: (s["date"], s["time"]))


def find_slot_on_date(department: str, date_str: str):
    """
    Looks for an available slot on a specific date. Expects date_str already
    normalized to "YYYY-MM-DD" (the appointment agent's extraction step handles
    turning "the 16th of July" etc. into that format) so this can do an exact
    match. Falls back to substring matching in case a partial date slips through,
    so the demo degrades gracefully instead of just failing.
    """
    available = get_available_slots(department)
    for slot in available:
        if slot["date"] == date_str:
            return slot
    for slot in available:
        if date_str in slot["date"] or slot["date"] in date_str:
            return slot
    return None


def book_slot(department: str, date_str: str, patient_name: str):
    """
    Marks a matching available slot as booked. Returns the booked slot dict
    on success, or None if no matching available slot was found.
    """
    for slot in SCHEDULE.get(department, []):
        if not slot["booked"] and (slot["date"] == date_str or date_str in slot["date"] or slot["date"] in date_str):
            slot["booked"] = True
            slot["patient_name"] = patient_name
            return slot
    return None
