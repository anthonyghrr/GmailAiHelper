from gpt4all import GPT4All
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from colorama import Fore, Style, init
import pickle
import os
import json
import re

init(autoreset=True)

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Connect to Gmail
def connect_to_gmail():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service

# Get the list of emails
def list_emails(service):
    try:
        results = service.users().messages().list(userId='me', maxResults=100).execute()
        messages = results.get('messages', [])
        return messages
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

# Fetch email details (subject and sender)
def get_email_details(service, message_id):
    try:
        message = service.users().messages().get(userId='me', id=message_id, format='metadata').execute()
        headers = message.get('payload', {}).get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
        sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown Sender")
        return subject, sender
    except Exception as e:
        print(f"Error fetching email details: {e}")
        return "Error", "Error"

# Function to clean and extract valid JSON from the model's response
def clean_json_response(response):
    try:
        # Match JSON block even with extra unrelated text
        json_response = re.search(r'{.*}', response, re.DOTALL)
        if json_response:
            json_data = json_response.group(0)
            return json.loads(json_data)
        else:
            print("Error: No valid JSON found in the response.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding the response JSON: {e}")
        print(f"Raw response: {response}")
        return None

# Summarize email and categorize
def process_email(model, subject, sender):
    try:
        prompt = f"""
        You are an assistant that categorizes and summarizes emails. Below is the email information:

        Email Subject: {subject}
        Sender: {sender}

        Respond **only** with a valid JSON object in the following format do not output anything else but the following:
        {{
            "summary": "<summary>",
            "category": "<category>",
            "priority": "<priority>",
            "response_required": "<Yes/No> if spam say no"
        }}

        Constraints: You must always give me one the following answers dont leave anything blank 
        - "category" must be one of the following: 'School', 'Shopping', 'Spam', 'Social Media', 'Personal'.
        - "priority" must be one of the following: 'High', 'Medium', 'Low'.
        - "response_required" must be either 'Yes' or 'No'.
        """

        # Limiting the token count
        response = model.generate(prompt, max_tokens=45)  

        # Log the raw response for debugging
        print(f"Raw model response: {response}")

        # Clean and parse the JSON from the response
        response_data = clean_json_response(response)
        if response_data:
            required_keys = {"summary", "category", "priority", "response_required"}
            if required_keys.issubset(response_data.keys()):
                summary = response_data.get('summary', 'No Summary')
                category = response_data.get('category', 'No Category')
                priority = response_data.get('priority', 'No Priority')
                response_required = response_data.get('response_required', 'No Response Required')
                return summary, category, priority, response_required
            else:
                print("Error: Response JSON is missing required keys.")
        else:
            print(f"Failed to process email from {sender}. Response parsing failed.")

        # Log invalid responses for debugging
        # with open('debug_responses.log', 'a') as log_file:
        #     log_file.write(f"Raw response: {response}\n")

        return "Error summarizing email", "Error", "Error", "Error"
    except Exception as e:
        print(f"An error occurred while processing the email: {e}")
        return "Error summarizing email", "Error", "Error", "Error"

# Display Results with Colors
def display_result(sender, summary, category, priority, response_required):
    # Color mapping for categories
    category_colors = {
        'School': Fore.BLUE,
        'Shopping': Fore.MAGENTA,
        'Spam': Fore.RED,
        'Social Media': Fore.CYAN,
        'Work': Fore.YELLOW,
        'Personal': Fore.GREEN,
    }
    color = category_colors.get(category, Fore.LIGHTWHITE_EX)

    print(f"{Style.BRIGHT}{color}Sender: {sender}")
    print(f"{Fore.LIGHTBLACK_EX}Summary: {summary}")
    print(f"{Fore.LIGHTWHITE_EX}Category: {color}{category}")
    print(f"{Fore.LIGHTWHITE_EX}Priority: {Fore.LIGHTMAGENTA_EX}{priority}")
    print(f"{Fore.LIGHTWHITE_EX}Response Required: {Fore.LIGHTGREEN_EX if response_required == 'Yes' else Fore.RED}{response_required}")
    print(f"{Fore.LIGHTBLACK_EX}{'-' * 50}")

# Main function to process emails
def main():
    print(f"{Fore.LIGHTGREEN_EX}Connecting to Gmail...")
    service = connect_to_gmail()
    print(f"{Fore.LIGHTGREEN_EX}Connected to Gmail successfully!")

    print(f"{Fore.LIGHTBLUE_EX}Retrieving emails...")
    emails = list_emails(service)
    print(f"{Fore.LIGHTBLUE_EX}Retrieved {len(emails)} emails.")

    if not emails:
        print(f"{Fore.RED}No emails found.")
        return

    # Load the GPT model once
    print(f"{Fore.LIGHTCYAN_EX}Loading GPT model...")
    model = GPT4All("Llama-3.2-3B-Instruct-Q4_0.gguf")  
    print(f"{Fore.LIGHTCYAN_EX}GPT model loaded successfully!")

    for email in emails[:20]: 
        message_id = email['id']
        subject, sender = get_email_details(service, message_id)
        print(f"{Fore.LIGHTCYAN_EX}Processing email from {sender}...")
        summary, category, priority, response_required = process_email(model, subject, sender)
        display_result(sender, summary, category, priority, response_required)

if __name__ == "__main__":
    main()