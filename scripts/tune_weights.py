#!/usr/bin/env python3
"""
scripts/tune_weights.py

Standalone script to run automatic model personalization (weight tuning)
for all users who have sufficient data. Can be run via a cron job or manually.
"""
import sys
import os

# Add parent directory to path to import neuroframe modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from storage.gsheets import NeuroGSheets
from storage.repo import NeuroRepo
from auth import make_gspread_client_from_secrets
from neuroframe.engine import tune_user_weights

def main():
    print("Starting automated weight tuning...")
    
    # Needs valid streamlit secrets in environment or `.streamlit/secrets.toml`
    try:
        gc = make_gspread_client_from_secrets()
    except Exception as e:
        print(f"Failed to authenticate with Google Sheets: {e}")
        print("Make sure you run this from the project root where .streamlit/secrets.toml is accessible.")
        return

    repo = NeuroRepo(NeuroGSheets(gc))
    
    # Get all users
    try:
        db = repo.db
        all_users = db.users.get_all_records()
    except Exception as e:
        print(f"Failed to fetch users: {e}")
        return
        
    for u in all_users:
        username = u.get("username", "")
        if not username:
            continue
            
        print(f"Analyzing user: {username}")
        uobj = repo.get_user(username)
        if not uobj:
            print(f"  -> Could not parse user info.")
            continue
            
        # Get last 7 days of logs
        recent_logs = repo.get_daily_logs_for_user(username, limit=7)
        if not recent_logs:
            print(f"  -> No recent logs found. Skipping.")
            continue
            
        # Tune weights
        updated_weights = tune_user_weights(uobj.baseline, recent_logs)
        if not updated_weights:
            print(f"  -> Not enough valid data for tuning. Skipping.")
            continue
            
        # Patch the repo
        old_sens = uobj.baseline.caffeine_sensitivity
        old_circ = uobj.baseline.circadian_weight
        
        new_sens = updated_weights.get("caffeine_sensitivity", old_sens)
        new_circ = updated_weights.get("circadian_weight", old_circ)
        
        if abs(new_sens - old_sens) < 0.001 and abs(new_circ - old_circ) < 0.001:
            print(f"  -> No change in weights.")
            continue
            
        patch = {
            "caffeine_sensitivity": str(new_sens),
            "circadian_weight": str(new_circ)
        }
        
        ok = repo.update_user_baseline(username, patch)
        if ok:
            print(f"  -> SUCCESS: Updated weights. Caffeine Sens: {old_sens} -> {new_sens}, Circadian: {old_circ} -> {new_circ}")
        else:
            print(f"  -> ERROR: Failed to update DB for {username}.")

    print("Finished automated weight tuning.")

if __name__ == "__main__":
    main()
