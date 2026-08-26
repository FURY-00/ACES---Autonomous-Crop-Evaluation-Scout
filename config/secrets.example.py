"""
Copy this file to config/secrets.py and fill it in.

config/secrets.py is in .gitignore. NEVER commit a bot token: anyone with it
can read and post to your group, and GitHub's scanners will find it within
minutes of a public push.
"""

# --- Telegram -----------------------------------------------------------
# 1. Message @BotFather on Telegram, send /newbot, follow the prompts.
# 2. Copy the token it gives you here.
# 3. Add the bot to your group, then make it an admin (otherwise it cannot
#    read or post reliably in groups).
# 4. Send any message in the group, then open:
#       https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
#    and copy the "chat":{"id":-100...} value. Group IDs are NEGATIVE.
TELEGRAM_TOKEN   = "0000000000:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
TELEGRAM_CHAT_ID = "-1000000000000"

# --- Google Sheets ------------------------------------------------------
# Google Cloud Console -> new project -> enable Sheets API and Drive API ->
# Credentials -> Service Account -> Keys -> Add key -> JSON.
# Save the JSON as config/google_credentials.json, then SHARE your sheet with
# the service account's client_email (it behaves like a separate user, and
# will get a 403 on a sheet nobody shared with it).
GOOGLE_CREDENTIALS = "config/google_credentials.json"
