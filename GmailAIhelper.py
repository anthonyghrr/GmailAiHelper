from gpt4all import GPT4All
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from colorama import Fore, Style, init
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import os
import json
import re
import redis
import time

init(autoreset=True)

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Connect to Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

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
def list_emails(service, query=None):
    try:
        results = service.users().messages().list(userId='me', q=query, maxResults=100).execute()
        if 'messages' in results:
            messages = results['messages']
            return messages
        else:
            return []
    except Exception as e:
        return []

# Fetch email details (subject, sender, date, message_id_header)
def get_email_details(service, message_id):
    try:
        message = service.users().messages().get(userId='me', id=message_id, format='metadata').execute()
        headers = message.get('payload', {}).get('headers', [])
        subject = None
        for h in headers:
            if h['name'] == 'Subject':
                subject = h['value']
                break
        if subject is None:
            subject = "No Subject"

        sender = None
        for h in headers:
            if h['name'] == 'From':
                sender = h['value']
                break
        if sender is None:
            sender = "Unknown Sender"

        date = None
        for h in headers:
            if h['name'] == 'Date':
                date = h['value']
                break

        message_id_header = None
        for h in headers:
            if h['name'] == 'Message-ID':
                message_id_header = h['value']
                break
        
        if not message_id_header:
            return None  # Skip emails without Message-ID
        
        if not date:
            date = 'Unknown Date'

        # Store the email details in Redis
        email_data = {
            'subject': subject,
            'sender': sender,
            'date': date,
            'message_id_header': message_id_header
        }
        redis_client.set(f"email:{message_id}", json.dumps(email_data))
        
        return subject, sender, date, message_id_header
    except Exception as e:
        return None

# Summarize email and categorize
def process_email(model, subject, sender, date, message_id_header):
    try:
        if not message_id_header:
            return "Error summarizing email", "Error", "Error", "Error"
        
        email_key = f"summary:{message_id_header}"
        
        cached_summary = redis_client.get(email_key)
        
        if cached_summary:
            response_data = json.loads(cached_summary.decode('utf-8'))
        else:
            prompt = f"""
            You are an assistant tasked with summarizing and categorizing emails. Respond **only** with the following JSON format:

            {{
                "summary": "<summary>",
                "category": "<category>",
                "priority": "<priority>",
                "response_required": "<Yes/No>"
            }}

            ### Constraints:
            - "category" must be one of: 'Work', 'School', 'Shopping', 'Social', 'Personal'.
            - "priority" must be one of: 'Urgent', 'Important', 'Normal'.
            - "response_required" must be one of: 'Yes', 'No'.

            ### Input:
            - Subject: {subject}
            - Sender: {sender}

            ### Output:
            Return strictly the JSON format as specified above. Do not include any extra text or explanations.
            """
            
            response = model.generate(prompt, max_tokens=100, top_p=0.9)
            response_data = clean_json_response(response)
            
            if response_data:
                required_keys = {"summary", "category", "priority", "response_required"}
                if required_keys.issubset(response_data.keys()):
                    redis_client.set(email_key, json.dumps(response_data), ex=14400)  # Cache for 4 hours
                else:
                    return "Error summarizing email", "Error", "Error", "Error"
            else:
                return "Error summarizing email", "Error", "Error", "Error"
        
        summary = response_data.get('summary', 'No Summary')
        category = response_data.get('category', 'No Category')
        priority = response_data.get('priority', 'No Priority')
        response_required = response_data.get('response_required', 'No Response Required')
        
        return summary, category, priority, response_required
    except Exception as e:
        return "Error summarizing email", "Error", "Error", "Error"

# Function to clean and extract valid JSON from the model's response
def clean_json_response(response):
    try:
        json_response = re.search(r'{.*}', response, re.DOTALL)
        if json_response:
            json_data = json_response.group(0)
            return json.loads(json_data)
        else:
            return None
    except json.JSONDecodeError as e:
        return None

# Display results with colors
def display_result(sender, summary, category, priority, response_required):
    category_colors = {
        'Work': Fore.BLUE,
        'School': Fore.MAGENTA,
        'Shopping': Fore.YELLOW,
        'Social': Fore.CYAN,
        'Personal': Fore.GREEN,
    }
    color = category_colors.get(category, Fore.LIGHTWHITE_EX)

    print(f"{Style.BRIGHT}{color}Sender: {sender}")
    print(f"{Fore.LIGHTBLACK_EX}Summary: {summary}")
    print(f"{Fore.LIGHTWHITE_EX}Category: {color}{category}")
    print(f"{Fore.LIGHTWHITE_EX}Priority: {Fore.LIGHTMAGENTA_EX}{priority}")
    print(f"{Fore.LIGHTWHITE_EX}Response Required: {Fore.LIGHTGREEN_EX if response_required == 'Yes' else Fore.RED}{response_required}")
    print(f"{Fore.LIGHTBLACK_EX}{'-' * 50}")

# Generate only the visualizations
def generate_visualizations(processed_emails):
    category_counts = defaultdict(int)
    response_counts_yes = defaultdict(int)
    response_counts_no = defaultdict(int)

    for email in processed_emails:
        category_counts[email['category']] += 1
        if email['response_required'] == 'Yes':
            response_counts_yes[email['priority']] += 1
        else:
            response_counts_no[email['priority']] += 1

    # Pie Chart for Category Distribution
    plt.figure(figsize=(8, 6))
    plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Email Category Distribution')
    plt.show()

    # Stacked Bar Chart for Response Requirement by Priority
    response_priorities = ['Urgent', 'Important', 'Normal']
    response_counts_yes_list = [response_counts_yes[priority] for priority in response_priorities]
    response_counts_no_list = [response_counts_no[priority] for priority in response_priorities]

    x = range(len(response_priorities))
    plt.figure(figsize=(10, 6))
    plt.bar(x, response_counts_yes_list, label='Requires Response', color='lightblue')
    plt.bar(x, response_counts_no_list, bottom=response_counts_yes_list, label='No Response Needed', color='lightgrey')
    plt.xlabel('Priority Level')
    plt.ylabel('Number of Emails')
    plt.title('Response Requirement by Priority Level')
    plt.xticks(x, response_priorities)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function to process emails
def main():
    print(f"{Fore.LIGHTGREEN_EX}Connecting to Gmail...")
    service = connect_to_gmail()
    print(f"{Fore.LIGHTGREEN_EX}Connected to Gmail successfully!")

    print(f"{Fore.LIGHTBLUE_EX}Retrieving emails...")

    query = ""  # Empty query to retrieve all available emails
    emails = list_emails(service, query=query)
    print(f"{Fore.LIGHTBLUE_EX}Retrieved {len(emails)} emails.")

    if not emails:
        print(f"{Fore.RED}No new emails found.")
        return

    print(f"{Fore.LIGHTCYAN_EX}Loading GPT model...")
    model = GPT4All("Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf")
    print(f"{Fore.LIGHTCYAN_EX}GPT model loaded successfully!")

    processed_emails = []

    for email in emails[:100]:
        message_id = email['id']
        email_details = get_email_details(service, message_id)
        
        # Skip emails that have errors in fetching details
        if email_details is None:
            continue

        subject, sender, date, message_id_header = email_details
        summary, category, priority, response_required = process_email(model, subject, sender, date, message_id_header)
        if summary != "Error summarizing email":
            display_result(sender, summary, category, priority, response_required)
            processed_emails.append({
                'sender': sender,
                'summary': summary,
                'category': category,
                'priority': priority,
                'response_required': response_required,
                'date': date
            })

    # Generate visualizations: pie chart and stacked bar chart
    generate_visualizations(processed_emails)

    current_ts = int(time.time())
    redis_client.set('last_processed_ts', current_ts)

if __name__ == "__main__":
    main()