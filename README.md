# NeuroFrame

NeuroFrame is a Streamlit app that predicts your daily energy curve and suggests a practical schedule from:

- sleep timing
- caffeine doses
- daily workload
- subjective clarity feedback

It is not a medical treatment app. It is a day-design tool based on a lightweight heuristic model.

## Features

- User signup/login
- Per-user baseline setup (onboarding wizard)
- Daily input editing (sleep override, caffeine doses, workload, clarity)
- 24h net-energy curve visualization
- Zone interpretation:
  - Prime Zone (deep work)
  - Crash Zone (low-load tasks)
  - Sleep Gate (wind-down)
- Basic adaptation (`baseline_offset`) from daily self-rating
- Admin page for user baseline editing and log inspection

## Tech Stack

- Python
- Streamlit
- Google Sheets (via `gspread`) as storage backend
- Matplotlib for plotting

## Project Structure

```text
neuroframe_app.py         # Main Streamlit app
admin.py                  # Admin dashboard logic
pages/1_Admin.py          # Streamlit multipage admin entry
auth.py                   # Auth/session/repo wiring
neuroframe/engine.py      # Core prediction model
neuroframe/coach.py       # Zone interpretation + schedule suggestions
neuroframe/plots.py       # Plot rendering
storage/gsheets.py        # Google Sheets client
storage/repo.py           # Backend-agnostic app repository interface
storage/security.py       # Password hashing/verification helpers
shared/today_input.py     # Daily input parsing/serialization helpers
tests/                    # Unit tests
```

## Setup

1. Create and activate a virtual environment:

```bash
python3 -m venv venv_neuro
source venv_neuro/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.streamlit/secrets.toml` with:

```toml
admin_pass = "your-admin-password"

[gcp_service_account]
type = "service_account"
project_id = "..."
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "..."
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "..."
```

4. Prepare Google Spreadsheet named `NeuroFrame_DB` with worksheets:

- `users`
- `daily_logs`

Headers are auto-initialized when worksheets are empty.

## Run

Main app:

```bash
streamlit run neuroframe_app.py
```

Admin page:

```bash
streamlit run admin.py
```

or from multipage mode in the app sidebar (`pages/1_Admin.py`).

## Data Model (Google Sheets)

`users` columns:

- `username`, `password`, `created_at`, `last_login`
- `baseline_sleep_start`, `baseline_wake`, `chronotype_shift_hours`
- `caffeine_half_life_hours`, `caffeine_sensitivity`
- `baseline_offset`
- `circadian_weight`, `sleep_pressure_weight`, `drug_weight`, `load_weight`
- `onboarded`

`daily_logs` columns:

- `date`, `username`
- `sleep_override_on`, `sleep_cross_midnight`
- `sleep_start`, `wake_time`
- `doses_json`
- `workload_level`
- `subjective_clarity`
- `updated_at`

## Security Notes

- Passwords are stored as PBKDF2-SHA256 hashes (not plaintext).
- Legacy plaintext rows are migrated to hashed passwords automatically on successful login.
- Keep `.streamlit/secrets.toml` out of git (already ignored).

## Tests

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

