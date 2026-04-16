import json
import os

def save_session(data, filename):
    """
    Save the user's session (original text, summary, settings, etc.)
    into a JSON file chosen by the user.

    Parameters:
    - data (dict): A dictionary containing session information.
    - filename (str): The name of the file to save (e.g., "session1.json")

    Returns:
    - bool: True if saved successfully, False otherwise.
    """

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return True

    except Exception as e:
        print(f"Error saving session: {e}")
        return False


def load_session(filename):
    """
    Load a previously saved session from a JSON file.

    Parameters:
    - filename (str): The name of the file to load (e.g., "session1.json")

    Returns:
    - dict: The loaded session data if successful.
    - None: If the file does not exist or cannot be read.
    """

    if not os.path.exists(filename):
        print("File not found.")
        return None

    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    except Exception as e:
        print(f"Error loading session: {e}")
        return None
