!pip install streamlit
!pip install streamlit streamlit-option-menu pyngrok
!pip install google-generativeai
!pip install PyPDF2
!pip install pandas
!pip install streamlit pyngrok --quiet

import os
import random
import google.generativeai as genai
import PyPDF2 # Keep the import for potential future use or manual adaptation
import pandas as pd
import json
from datetime import datetime
import re
import time # Added for simulating delays


# Configure Google Gemini AI
# Replace with your actual API key
# It's recommended to use environment variables in a real application
os.environ["API_KEY"] = 'YOUR_API_KEY' # Replace with your actual API key
genai.configure(api_key=os.environ["API_KEY"])

# --- User Authentication Functions ---

USERS_FILE = 'users.json'
INTERVIEW_HISTORY_DIR = 'interview_data'

def load_users():
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def validate_user_input(input_text, input_type):
    """Validate user input with specific rules"""
    if not input_text.strip():
        return False, f"{input_type} cannot be empty."

    if ' ' in input_text:
        return False, f"{input_type} cannot contain spaces."

    if input_type == "Username" and len(input_text) < 3:
        return False, "Username must be at least 3 characters long."

    if input_type == "Password" and len(input_text) < 6:
        return False, "Password must be at least 6 characters long."

    return True, "Input validated successfully."

def handle_user_authentication():
    while True:
        print("\n--- User Authentication ---")
        print("1. Login")
        print("2. Register")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            username = input("Enter username: ")
            password = input("Enter password: ")
            users = load_users()

            if username not in users:
                print("Error: Username not found.")
            elif users[username]['password'] != password: # In a real app, use hashed passwords
                print("Error: Incorrect password.")
            else:
                print("Login successful!")
                # Update last login time
                users[username]['last_login'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                save_users(users)
                return username # Return username on successful login

        elif choice == '2':
            username = input("Choose a username: ")
            password = input("Choose a password: ")
            users = load_users()

            valid, message = validate_user_input(username, "Username")
            if not valid:
                print(f"Registration Error: {message}")
                continue
            valid, message = validate_user_input(password, "Password")
            if not valid:
                 print(f"Registration Error: {message}")
                 continue

            if username in users:
                print("Error: Username already exists.")
            else:
                users[username] = {
                    'password': password, # In a real app, use hashed passwords
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'last_login': None,
                    'interview_history': [] # This will be handled in separate files
                }
                save_users(users)
                print("Account created successfully!")

        elif choice == '3':
            print("Exiting authentication.")
            return None

        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

# --- Utility Functions ---

def generate_ai_response(prompt, model_name='gemini-1.5-flash-latest'):
    """Generic function to generate AI responses"""
    try:
        model = genai.GenerativeModel(model_name)
        # Add a small delay to avoid hitting API rate limits too quickly
        time.sleep(1)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            print("API quota exceeded. Please try again later or check your API usage.")
        else:
            print(f"Error generating response: {e}")
        return "Error generating response. Please try again."

def extract_text_from_pdf(file_path):
    """Extract text from PDF file (requires PyPDF2)"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
    except FileNotFoundError:
        print(f"Error: PDF file not found at {file_path}")
        return ""
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

# Simplified extract_text for command-line (assuming text is provided directly)
def extract_text_from_source(source):
    """Extract text from uploaded file or direct text input"""
    # In a command-line version, we'll assume the user provides text directly or from a file path
    if os.path.exists(source):
        try:
            # Attempt to read as text file
            with open(source, 'r') as f:
                return f.read()
        except Exception:
            # If not a text file, try PDF (requires PyPDF2)
            return extract_text_from_pdf(source)
    else:
        # Assume direct text input
        return source


def evaluate_answer(user_answer, ideal_answer=""):
    """Evaluate user's answer (simplified for command-line)"""
    if not user_answer.strip():
        return 0

    # Use AI to evaluate if an ideal answer or detailed feedback is needed
    if ideal_answer:
         prompt = f"""
         Evaluate the following user answer based on the ideal answer.
         Provide a score out of 10. Consider relevance, completeness, and clarity.
         User Answer: {user_answer}
         Ideal Answer: {ideal_answer}
         Score: [Score out of 10, e.g., 7/10]
         """
         response = generate_ai_response(prompt)
         try:
             score_match = re.search(r'Score:\s*(\d+)/10', response)
             if score_match:
                 return int(score_match.group(1))
             else:
                 print("Could not parse score from AI feedback. Providing a default score.")
                 # Fallback to basic word overlap if AI scoring fails
                 user_words = set(user_answer.lower().split())
                 ideal_words = set(ideal_answer.lower().split())
                 overlap = len(user_words.intersection(ideal_words))
                 total_unique_words = len(ideal_words)
                 return round(min(10, (overlap / max(1, total_unique_words)) * 10), 0) # Ensure division by zero is avoided
         except Exception as e:
             print(f"Error during AI evaluation: {e}. Falling back to basic scoring.")
             # Fallback to basic word overlap if AI scoring fails
             user_words = set(user_answer.lower().split())
             ideal_words = set(ideal_answer.lower().split())
             overlap = len(user_words.intersection(ideal_words))
             total_unique_words = len(ideal_words)
             return round(min(10, (overlap / max(1, total_unique_words)) * 10), 0)


    else:
        # Basic evaluation if no ideal answer is provided (e.g., for generic questions)
        score = min(10, len(user_answer.split()) / 15) # Score based on length
        key_terms = ['experience', 'project', 'team', 'solution', 'challenge', 'success', 'learn', 'improve']
        score += sum(1 for term in key_terms if term.lower() in user_answer.lower()) # Bonus for key terms
        return round(min(10, score), 0)


def load_questions(company):
    """Load and select random questions based on company and difficulty levels."""
    try:
        if company == "TCS":
            file_path = "tcs1.csv"
            question_col = "Question"
        elif company == "Amazon":
            file_path = "amazon1.csv"
            question_col = "Question"
        elif company == "Accenture":
            file_path = "acc.csv"
            question_col = "questions"
        elif company == "Microsoft":
            file_path = "micro.csv"
            question_col = "Question"
        else:
            print(f"Error: Unknown company: {company}")
            return None, None

        # Load the CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}")
            return None, None

        # Check if the question column exists
        if question_col not in df.columns:
            print(f"Error: Column '{question_col}' not found in {file_path}. Available columns: {', '.join(df.columns)}")
            return None, None

        # Extract difficulty and clean question text
        try:
            # Handle potential NaN values before string operations
            df[question_col] = df[question_col].astype(str).fillna('')
            df['difficulty'] = df[question_col].str.extract(r'\((.*?)\)$', expand=False)
            df['question'] = df[question_col].str.replace(r'\(.*?\)$', '', regex=True)
        except Exception as e:
            print(f"Error extracting difficulty from questions: {str(e)}")
            return None, None

        # Clean and standardize difficulty values
        df['difficulty'] = df['difficulty'].str.lower().str.strip()
        # Replace any remaining NaN or empty strings in difficulty with a default like 'unknown'
        df['difficulty'] = df['difficulty'].replace('', 'unknown').fillna('unknown')


        # Remove any empty rows based on cleaned question text
        df = df.dropna(subset=['question']).reset_index(drop=True)
        # Remove rows where cleaned question is just whitespace
        df = df[df['question'].str.strip() != ''].reset_index(drop=True)


        # Get questions by difficulty, handling potential missing groups gracefully
        low_questions = df[df['difficulty'] == 'low']
        medium_questions = df[df['difficulty'] == 'medium']
        high_questions = df[df['difficulty'] == 'high']
        unknown_questions = df[df['difficulty'] == 'unknown']


        # Check if we have enough questions of each difficulty
        required = {'low': 3, 'medium': 4, 'high': 3}
        selected_low, selected_medium, selected_high = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        current_seed = int(datetime.now().timestamp()) # Use timestamp as seed

        try:
            if len(low_questions) >= required['low']:
                selected_low = low_questions.sample(n=required['low'], random_state=current_seed)
            else:
                print(f"Warning: Not enough low difficulty questions in {company} file. Found {len(low_questions)}, Required {required['low']}. Using all available low questions.")
                selected_low = low_questions

            if len(medium_questions) >= required['medium']:
                selected_medium = medium_questions.sample(n=required['medium'], random_state=current_seed)
            else:
                 print(f"Warning: Not enough medium difficulty questions in {company} file. Found {len(medium_questions)}, Required {required['medium']}. Using all available medium questions.")
                 selected_medium = medium_questions

            if len(high_questions) >= required['high']:
                selected_high = high_questions.sample(n=required['high'], random_state=current_seed)
            else:
                print(f"Warning: Not enough high difficulty questions in {company} file. Found {len(high_questions)}, Required {required['high']}. Using all available high questions.")
                selected_high = high_questions

            # Combine questions
            selected_questions = pd.concat([selected_low, selected_medium, selected_high]).reset_index(drop=True)

            # If we still don't have 10 questions, pad with unknown difficulty questions or sample from all available
            if len(selected_questions) < 10:
                remaining_needed = 10 - len(selected_questions)
                available_for_padding = df[~df.index.isin(selected_questions.index)] # Questions not already selected
                if len(available_for_padding) >= remaining_needed:
                    selected_padding = available_for_padding.sample(n=remaining_needed, random_state=current_seed)
                    selected_questions = pd.concat([selected_questions, selected_padding]).reset_index(drop=True)
                else:
                     print(f"Warning: Only found {len(selected_questions)} unique questions of specified difficulties. Using all available unique questions.")
                     # If not enough unique questions, just use what we have or add duplicates if necessary (less ideal)
                     if len(df) >= 10:
                         selected_questions = df.sample(n=10, random_state=current_seed).reset_index(drop=True)
                     else:
                          selected_questions = df.copy() # Use all available if less than 10


            # Ensure exactly 10 questions if possible
            selected_questions = selected_questions.head(10)


            # Verify we have exactly 10 questions if the original data allowed for it
            if len(selected_questions) != 10:
                 print(f"Warning: Could not select exactly 10 questions. Found {len(selected_questions)}.")


            return selected_questions, 'question'

        except Exception as e:
            print(f"Error selecting questions: {str(e)}")
            # Fallback to selecting any 10 questions if specific difficulty selection fails
            if len(df) >= 10:
                print("Falling back to selecting any 10 questions.")
                return df.sample(n=10, random_state=int(datetime.now().timestamp())).reset_index(drop=True), 'question'
            elif not df.empty:
                 print(f"Falling back to using all {len(df)} available questions.")
                 return df.reset_index(drop=True), 'question'
            else:
                 print("No questions found in the file.")
                 return None, None


    except Exception as e:
        print(f"Error loading questions for {company}: {str(e)}")
        return None, None


def get_role_specific_questions(role):
    """Generate role-specific interview questions"""
    role_prompts = {
        "Software Developer": [
            "Generate a technical question about software design patterns.",
            "Ask about experience with agile development methodologies.",
            "Create a question about debugging complex issues.",
            "Ask about code optimization and performance.",
            "Generate a question about version control and collaboration."
        ],
        "Data Scientist": [
            "Ask about experience with machine learning algorithms.",
            "Generate a question about data preprocessing techniques.",
            "Create a question about statistical analysis methods.",
            "Ask about big data technologies.",
            "Generate a question about model evaluation metrics."
        ],
        "DevOps Engineer": [
            "Ask about CI/CD pipeline implementation.",
            "Generate a question about container orchestration.",
            "Create a question about infrastructure automation.",
            "Ask about monitoring and logging solutions.",
            "Generate a question about cloud services."
        ],
        "Frontend Developer": [
            "Ask about modern JavaScript frameworks.",
            "Generate a question about responsive design.",
            "Create a question about web performance optimization.",
            "Ask about state management in web applications.",
            "Generate a question about UI/UX best practices."
        ],
        "Backend Developer": [
            "Ask about API design and implementation.",
            "Generate a question about database optimization.",
            "Create a question about scalability solutions.",
            "Ask about microservices architecture.",
            "Generate a question about server security."
        ]
    }

    questions = []
    prompts = role_prompts.get(role, role_prompts["Software Developer"])
    print(f"Generating 5 role-specific questions for {role}...")
    for prompt in prompts[:5]: # Take the first 5 technical prompts
        question = generate_ai_response(f"{prompt} for a {role} position. Make the question specific and technical.")
        if question and '?' in question:
             questions.append(question)
        else:
             # Fallback if AI fails to generate a valid question
             print(f"Warning: AI failed to generate question for prompt '{prompt}'. Using fallback.")
             questions.append(f"Discuss a technical challenge related to {role} development.")


    # Generate 5 more behavioral questions specific to the role
    print(f"Generating 5 behavioral questions for {role}...")
    behavioral_prompt_template = "Generate a behavioral interview question for a {role} position that focuses on real work scenarios, such as teamwork, problem-solving, or handling pressure."
    for _ in range(5):
        prompt = behavioral_prompt_template.format(role=role)
        question = generate_ai_response(prompt)
        if question and '?' in question:
             questions.append(question)
        else:
             # Fallback if AI fails to generate a valid question
             print(f"Warning: AI failed to generate behavioral question. Using fallback.")
             questions.append(f"Tell me about a time you demonstrated a key behavioral skill for a {role} role.")


    # Ensure exactly 10 questions if possible
    return questions[:10]

def generate_resume_questions(resume_text):
    """Generate exactly 10 questions covering all resume aspects"""
    question_distribution = [
        {
            "topic": "Technical Skills",
            "count": 2,
            "prompt": "Based on the technical skills mentioned in the resume, generate specific questions about their practical application and proficiency level."
        },
        {
            "topic": "Internships/Work Experience",
            "count": 2,
            "prompt": "Looking at the internship/work experience section, ask about specific responsibilities, challenges faced, and solutions implemented."
        },
        {
            "topic": "Projects",
            "count": 2,
            "prompt": "Regarding the projects mentioned in the resume, ask about technical implementation, role, and impact."
        },
        {
            "topic": "Certifications",
            "count": 2,
            "prompt": "Based on the certifications listed, ask about their relevance to the role and practical application of learned skills."
        },
        {
            "topic": "Achievements",
            "count": 2,
            "prompt": "Looking at achievements and accomplishments, ask about the impact, metrics, and process of achieving these results."
        }
    ]

    all_questions = []
    print("Generating resume-based questions...")

    try:
        for category in question_distribution:
            prompt = f"""
Based on this resume content:\n{resume_text}\n

Generate exactly {category['count']} questions about the candidate's {category['topic']}.
{category['prompt']}

Requirements:
- Questions should be specific to the content in the resume.
- Focus on practical experience and implementation.
- Ask about concrete examples and results.
- Questions should require detailed responses.
- Provide the questions as a numbered list.
"""

            response = generate_ai_response(prompt)
            # Parse numbered list
            questions = [q.split('.', 1)[1].strip() for q in response.split('\n') if q.strip() and q.strip().startswith(tuple(str(i) + '.' for i in range(1, category['count'] + 1)))]

            # If we didn't get enough questions for this category from parsing, try splitting by lines or generating generic ones
            if len(questions) < category['count']:
                print(f"Warning: Could not parse {category['count']} questions for '{category['topic']}' from AI response. Trying alternative parsing or fallbacks.")
                questions = [q.strip() for q in response.split('\n') if '?' in q][:category['count']] # Fallback: split by lines and check for '?'

            # If still not enough, generate generic ones
            while len(questions) < category['count']:
                generic_q = f"Tell me about your experience with {category['topic'].lower()}?"
                questions.append(generic_q)
                print(f"Warning: Generating generic question for '{category['topic']}'.")


            all_questions.extend(questions)
            print(f"Generated {len(questions)} questions for {category['topic']}.")


    except Exception as e:
        print(f"Error generating questions: {str(e)}. Providing backup questions.")
        # Provide backup questions if generation fails
        all_questions = [
            "Describe your most relevant technical skills for this position.",
            "Tell me about your most challenging project.",
            "What was your most significant achievement in your last role?",
            "How have you applied your certifications in practical scenarios?",
            "Describe your role in your most recent internship.",
            "What technical challenges did you face in your projects?",
            "How do you keep your technical skills updated?",
            "Tell me about a situation where you used your technical expertise to solve a problem.",
            "What was your most impactful contribution in your previous role?",
            "How have your certifications helped you in your career development?"
        ]

    return all_questions[:10]  # Ensure exactly 10 questions


def save_interview_data(username, interview_type, data):
    """
    Save interview data for a user.
    """
    try:
        # Create interview_data directory if it doesn't exist
        if not os.path.exists(INTERVIEW_HISTORY_DIR):
            os.makedirs(INTERVIEW_HISTORY_DIR)

        # Construct the file path
        file_path = os.path.join(INTERVIEW_HISTORY_DIR, f'{username}_history.json')

        # Load existing history or create new
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {file_path}. Starting with empty history.")
                history = {}
        else:
            history = {}

        # Ensure the interview type exists in history
        if interview_type not in history:
            history[interview_type] = []

        # Create interview entry with proper ideal answers handling
        interview_entry = {
            'timestamp': data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'total_score': data.get('total_score', 0),
            'average_score': data.get('average_score', 0),
            'questions': data.get('questions', []),
            'answers': data.get('answers', []),
            'scores': data.get('scores', []),
            'feedback': data.get('feedback', ''),
            'ideal_answers': data.get('ideal_answers', [])
        }

         # Add company name if it's a company interview
        if interview_type == 'company' and 'company' in data:
            interview_entry['company'] = data['company']

        # Add role if it's a professional interview
        if interview_type == 'professional' and 'role' in data:
            interview_entry['role'] = data['role']


        # Add the new interview to history
        history[interview_type].append(interview_entry)

        # Save updated history
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=4)

        print("Interview results saved successfully!")
        return True

    except Exception as e:
        print(f"Error saving interview data: {str(e)}")
        return False

def get_user_interview_history(username):
    """
    Retrieve interview history for a user from the JSON file.
    """
    try:
        # Construct the file path
        file_path = os.path.join(INTERVIEW_HISTORY_DIR, f'{username}_history.json')

        # Check if history file exists
        if not os.path.exists(file_path):
            return []

        # Load the history file
        with open(file_path, 'r') as f:
            history = json.load(f)

        # Combine all interview types into a single list
        all_interviews = []
        for interview_type, interviews in history.items():
            for interview in interviews:
                # Add interview type to each record if not already present
                if 'interview_type' not in interview:
                    interview['interview_type'] = interview_type
                all_interviews.append(interview)

        # Sort interviews by timestamp (most recent first)
        all_interviews.sort(key=lambda x: x.get('timestamp', ''), reverse=True) # Use .get for safety

        return all_interviews

    except Exception as e:
        print(f"Error retrieving interview history: {str(e)}")
        return []


# --- Interview Flow Functions (Command-Line Adapted) ---

def run_interview(questions, interview_type, username, company=None, role=None):
    """Generic function to run an interview session in command line."""
    scores = []
    answers = []
    ideal_answers = []

    print(f"\n--- Starting {interview_type.replace('_', ' ').title()} Interview ---")
    if company:
        print(f"Company: {company}")
    if role:
        print(f"Role: {role}")

    num_questions = len(questions)

    for i in range(num_questions):
        print(f"\n--- Question {i + 1} of {num_questions} ---")
        question = questions[i]
        print(f"Question: {question}")

        user_answer = input("Your Answer: ")
        answers.append(user_answer)

        print("Evaluating your answer...")

        # Generate ideal answer based on interview type
        if interview_type == 'company' and company:
             ideal_answer = generate_company_answer(question, company)
        elif interview_type == 'professional' and role:
             ideal_answer = generate_ai_response(
                f"""Provide a concise but complete technical answer for this {role} question:
                Question: {question}

                Requirements:
                - Focus on key technical points
                - Include one practical example or implementation detail
                - Keep it clear and precise
                - Maximum 3-4 sentences
                - Highlight best practices

                Make it specific to {role} role."""
            )
        elif interview_type == 'behavioral':
             ideal_answer = generate_ai_response(
                f"""Provide a brief but effective STAR method answer for this behavioral question:
                Question: {question}

                Requirements:
                - One short sentence each for Situation, Task, Action, and Result
                - Focus on key points only
                - Be specific and measurable
                - Maximum 4 sentences total
                - Highlight the impact

                Format:
                S: [brief situation]
                T: [clear task]
                A: [specific action]
                R: [measurable result]"""
            )
        elif interview_type == 'resume':
             # For resume interview, we need the resume text to generate ideal answers
             # This is a limitation in the command-line version, so we'll generate a generic ideal answer
             ideal_answer = generate_ai_response(
                 f"Provide a concise, brief ideal answer (max 2-3 sentences) for: {question}"
             )

        else:
             ideal_answer = generate_ai_response(f"Provide a concise ideal answer for: {question}")


        ideal_answers.append(ideal_answer)
        score = evaluate_answer(user_answer, ideal_answer)
        scores.append(score)

        print(f"Your Score: {score}/10")
        print(f"Ideal Answer: {ideal_answer}")


    print("\n--- Interview Completed ---")

    total_score = sum(scores)
    average_score = total_score / num_questions if num_questions > 0 else 0

    print(f"\nTotal Score: {total_score}/{num_questions * 10}")
    print(f"Average Score: {average_score:.1f}/10")

    # Generate comprehensive feedback
    print("\nGenerating detailed feedback...")
    qa_details = ''.join([
        f"Q{i+1}: {questions[i]}\nA: {answers[i]}\nIdeal: {ideal_answers[i]}\nScore: {scores[i]}/10\n\n"
        for i in range(num_questions)
    ])
    feedback_prompt = f"""
Analyze this {interview_type.replace('_', ' ').title()} interview performance:
Total Score: {total_score}/{num_questions * 10}
Average Score: {average_score:.1f}/10

Questions and Answers:
{qa_details}

Provide detailed feedback covering:
1. Overall performance assessment
2. Key strengths demonstrated
3. Areas for improvement
4. Specific recommendations
5. Next steps for preparation
"""
    feedback = generate_ai_response(feedback_prompt)
    print("\n--- Interview Feedback ---")
    print(feedback)

    # Save interview data
    save_interview_data(
        username,
        interview_type,
        {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_score': total_score,
            'average_score': average_score,
            'questions': questions,
            'answers': answers,
            'scores': scores,
            'feedback': feedback,
            'ideal_answers': ideal_answers,
            'company': company if company else None,
            'role': role if role else None
        }
    )


def company_interview_flow(username):
    """Command-line flow for company-specific interviews."""
    print("\n--- Company Interview ---")
    print("Choose your company:")
    print("1. TCS")
    print("2. Accenture")
    print("3. Amazon")
    print("4. Microsoft")
    print("5. Back to Main Menu")

    while True:
        choice = input("Enter your choice: ")
        if choice == '1':
            company = "TCS"
            break
        elif choice == '2':
            company = "Accenture"
            break
        elif choice == '3':
            company = "Amazon"
            break
        elif choice == '4':
            company = "Microsoft"
            break
        elif choice == '5':
            return # Go back to main menu
        else:
            print("Invalid choice.")

    questions_df, column_name = load_questions(company)

    if questions_df is not None and not questions_df.empty:
        questions = questions_df[column_name].tolist()
        run_interview(questions, 'company', username, company=company)
    else:
        print(f"Could not load questions for {company}. Please check the file and try again.")


def behavioral_interview_flow(username):
    """Command-line flow for behavioral interviews."""
    behavioral_questions = [
        "Tell me about a time when you had to work with a difficult team member.",
        "Describe a situation where you had to meet a tight deadline.",
        "Share an experience where you had to handle multiple priorities.",
        "Tell me about a time when you demonstrated leadership skills.",
        "Describe a situation where you had to resolve a conflict.",
        "Share an example of how you dealt with failure.",
        "Tell me about a time when you went above and beyond.",
        "Describe a situation where you had to learn something quickly.",
        "Share an experience where you had to make a difficult decision.",
        "Tell me about a time when you showed initiative."
    ]
    run_interview(behavioral_questions, 'behavioral', username)

def professional_interview_flow(username):
    """Command-line flow for professional interviews."""
    print("\n--- Professional Interview ---")
    print("Select your role:")
    roles = [
        "Software Developer",
        "Data Scientist",
        "DevOps Engineer",
        "Frontend Developer",
        "Backend Developer"
    ]
    for i, role in enumerate(roles):
        print(f"{i + 1}. {role}")
    print(f"{len(roles) + 1}. Back to Main Menu")

    while True:
        try:
            choice = int(input("Enter your choice: "))
            if 1 <= choice <= len(roles):
                selected_role = roles[choice - 1]
                break
            elif choice == len(roles) + 1:
                return # Go back to main menu
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    questions = get_role_specific_questions(selected_role)
    if questions:
        run_interview(questions, 'professional', username, role=selected_role)
    else:
        print("Could not generate professional questions. Please try again.")


def resume_interview_flow(username):
    """Command-line flow for resume-based interviews."""
    print("\n--- Resume-based Interview ---")
    print("Please provide your resume content (you can paste text or provide a file path).")
    resume_source = input("Enter resume text or file path: ")

    resume_text = extract_text_from_source(resume_source)

    if not resume_text.strip():
        print("Could not get resume text. Cannot start resume interview.")
        return

    questions = generate_resume_questions(resume_text)

    if questions:
        # Store resume text temporarily for ideal answer generation in run_interview
        # In a real application, you might pass this explicitly or store it in a class
        global current_resume_text
        current_resume_text = resume_text # Using a global variable as a simple workaround

        run_interview(questions, 'resume', username)

        # Clean up the global variable
        current_resume_text = ""
    else:
        print("Could not generate resume questions. Please try again.")

def display_history_flow(username):
    """Command-line flow to display interview history."""
    print("\n--- Interview History ---")
    history = get_user_interview_history(username)

    if not history:
        print("No interview history found.")
        return

    print(f"Found {len(history)} interviews for {username}.")

    for i, record in enumerate(history):
        print(f"\n--- Interview {i + 1} ---")
        print(f"Type: {record.get('interview_type', 'N/A').title()}")
        print(f"Date: {record.get('timestamp', 'N/A')}")
        if 'company' in record:
            print(f"Company: {record['company']}")
        if 'role' in record:
            print(f"Role: {record['role']}")
        print(f"Total Score: {record.get('total_score', 0)}/{len(record.get('questions', [])) * 10 if record.get('questions') else 0}")
        print(f"Average Score: {record.get('average_score', 0):.1f}/10")

        # Option to view details
        view_details = input("View details (yes/no)? ").lower()
        if view_details == 'yes':
            print("\nQuestions and Answers:")
            questions = record.get('questions', [])
            answers = record.get('answers', [])
            ideal_answers = record.get('ideal_answers', [])
            scores = record.get('scores', [])

            min_length = min(len(questions), len(answers), len(scores), len(ideal_answers) if ideal_answers else len(questions))

            for j in range(min_length):
                print(f"Q{j+1} (Score: {scores[j]}/10): {questions[j]}")
                print(f"Your Answer: {answers[j]}")
                if ideal_answers and j < len(ideal_answers) and ideal_answers[j] and ideal_answers[j].strip():
                     print(f"Ideal Answer: {ideal_answers[j]}")
                print("-" * 20)

            print("\nFeedback:")
            print(record.get('feedback', 'No feedback available'))


# --- Main Application Loop ---

def main_menu(username):
    """Displays the main menu and handles user choices."""
    while True:
        print("\n--- Main Menu ---")
        print(f"Welcome, {username}!")
        print("1. Company Interview")
        print("2. Behavioral Interview")
        print("3. Professional Interview")
        print("4. Resume-based Interview")
        print("5. View Interview History")
        print("6. Logout")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            company_interview_flow(username)
        elif choice == '2':
            behavioral_interview_flow(username)
        elif choice == '3':
            professional_interview_flow(username)
        elif choice == '4':
            resume_interview_flow(username)
        elif choice == '5':
            display_history_flow(username)
        elif choice == '6':
            print("Logging out.")
            return False # Signal to go back to authentication
        elif choice == '7':
            print("Exiting program.")
            return True # Signal to exit completely
        else:
            print("Invalid choice. Please enter a number between 1 and 7.")


if _name_ == "_main_":
    while True:
        logged_in_username = handle_user_authentication()
        if logged_in_username:
            if main_menu(logged_in_username): # If main_menu returns True (exit), break loop
                break
        elif logged_in_username is None: # User chose to exit from authentication
            break
