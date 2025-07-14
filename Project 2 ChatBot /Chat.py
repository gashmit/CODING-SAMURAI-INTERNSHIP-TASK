import nltk
import re
import random
import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class RuleBasedChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.user_name = None
        self.conversation_state = "normal"
        
        # Predefined Rules and Responses
        self.rules = {
            'greeting': {
                'patterns': [
                    r'\b(hello|hi|hey|good morning|good afternoon|good evening|greetings)\b',
                    r'\b(howdy|sup|what\'s up|whats up)\b'
                ],
                'responses': [
                    "Hello! How can I help you today?",
                    "Hi there! Nice to meet you!",
                    "Hello! I'm your friendly chatbot. How are you doing?",
                    "Hey! What's on your mind today?",
                    "Greetings! How can I assist you?"
                ]
            },
            
            'time_query': {
                'patterns': [
                    r'\b(what time|current time|time now|what\'s the time|whats the time)\b',
                    r'\b(tell me the time|show me time|time please)\b'
                ],
                'responses': [
                    "The current time is: {time}",
                    "Right now it's: {time}",
                    "The time is currently: {time}",
                    "It's {time} right now"
                ]
            },
            
            'date_query': {
                'patterns': [
                    r'\b(what date|current date|today\'s date|todays date)\b',
                    r'\b(what day|what\'s today|whats today)\b'
                ],
                'responses': [
                    "Today is: {date}",
                    "The current date is: {date}",
                    "Today's date: {date}"
                ]
            },
            
            'datetime_query': {
                'patterns': [
                    r'\b(date and time|time and date|current date and time)\b',
                    r'\b(what\'s the date and time|whats the date and time)\b'
                ],
                'responses': [
                    "Current date and time: {datetime}",
                    "Right now it's: {datetime}",
                    "The current date and time is: {datetime}"
                ]
            },
            
            'day_query': {
                'patterns': [
                    r'\b(what day is it|what day of the week|which day)\b',
                    r'\b(what day today|today is what day)\b'
                ],
                'responses': [
                    "Today is {day}",
                    "It's {day} today",
                    "Today is a {day}"
                ]
            },
            
            'month_year_query': {
                'patterns': [
                    r'\b(what month|current month|which month)\b',
                    r'\b(what year|current year|which year)\b'
                ],
                'responses': [
                    "It's {month_year}",
                    "We're in {month_year}",
                    "Current month and year: {month_year}"
                ]
            },
            
            'age_calculation': {
                'patterns': [
                    r'\b(i was born in|born in|my birth year)\s+(\d{4})\b',
                    r'\b(calculate my age|how old am i|my age)\b.*(\d{4})\b'
                ],
                'responses': [
                    "Based on birth year {birth_year}, you are {age} years old!",
                    "You are {age} years old (born in {birth_year})",
                    "If you were born in {birth_year}, you're {age} years old!"
                ]
            },
            
            'math_help': {
                'patterns': [
                    r'\b(calculate|math|add|subtract|multiply|divide)\b',
                    r'\b(\d+)\s*[\+\-\*\/]\s*(\d+)\b'
                ],
                'responses': [
                    "I can help with basic math! Try asking: 'What is 5 + 3?' or 'Calculate 10 * 4'",
                    "I can do simple calculations! Give me a math problem like '15 + 7' or '20 / 4'",
                    "Ask me math questions like 'What is 25 - 8?' and I'll calculate it for you!"
                ]
            },
            
            'math_calculation': {
                'patterns': [
                    r'\b(what is|calculate|solve)\s+(\d+)\s*[\+]\s*(\d+)\b',
                    r'\b(what is|calculate|solve)\s+(\d+)\s*[\-]\s*(\d+)\b',
                    r'\b(what is|calculate|solve)\s+(\d+)\s*[\*x]\s*(\d+)\b',
                    r'\b(what is|calculate|solve)\s+(\d+)\s*[\/]\s*(\d+)\b'
                ],
                'responses': [
                    "The answer is: {result}",
                    "That equals: {result}",
                    "The result is: {result}"
                ]
            },
            
            'compliment': {
                'patterns': [
                    r'\b(you are good|you are great|you are awesome|you are amazing)\b',
                    r'\b(you\'re good|you\'re great|you\'re awesome|you\'re amazing)\b',
                    r'\b(good job|well done|nice work|great work)\b'
                ],
                'responses': [
                    "Thank you so much! That's very kind of you to say!",
                    "I appreciate the compliment! You're pretty awesome too!",
                    "Thanks! I'm just trying to be helpful. You're great to talk to!",
                    "That means a lot! Thank you for being so nice!"
                ]
            },
            
            'capabilities': {
                'patterns': [
                    r'\b(what can you do|your capabilities|what do you know)\b',
                    r'\b(what are you good at|what can you help with)\b'
                ],
                'responses': [
                    "I can: tell time/date, do basic math, have conversations, remember your name, calculate age, and chat about various topics!",
                    "My capabilities include: time/date information, simple calculations, friendly conversation, basic help, and remembering details about our chat!",
                    "I'm good at: providing current time/date, solving basic math problems, chatting, answering questions, and being a helpful companion!"
                ]
            },
            
            'joke': {
                'patterns': [
                    r'\b(tell me a joke|joke|funny|make me laugh)\b',
                    r'\b(something funny|humor|amusing)\b'
                ],
                'responses': [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "I told my computer a joke about UDP, but it didn't get it... oh wait, maybe it did!",
                    "Why do programmers prefer dark mode? Because light attracts bugs!",
                    "What do you call a chatbot that can sing? A Dell! (Get it? Like Adele but... Dell computers?)",
                    "Why did the chatbot break up with the database? It couldn't handle the relationship!"
                ]
            },
            
            'motivation': {
                'patterns': [
                    r'\b(motivate me|inspiration|encourage me|i need motivation)\b',
                    r'\b(feeling down|feeling sad|need encouragement)\b'
                ],
                'responses': [
                    "You're capable of amazing things! Every small step forward is progress worth celebrating!",
                    "Remember: you're stronger than you think and braver than you believe!",
                    "Challenges are just opportunities in disguise. You've got this!",
                    "Every expert was once a beginner. Keep going, you're doing great!",
                    "Your potential is limitless. Believe in yourself because I believe in you!"
                ]
            },
            
            'fun_facts': {
                'patterns': [
                    r'\b(fun fact|interesting fact|tell me something interesting)\b',
                    r'\b(did you know|random fact|cool fact)\b'
                ],
                'responses': [
                    "Fun fact: A group of flamingos is called a 'flamboyance'!",
                    "Did you know? Honey never spoils! Archaeologists have found edible honey in ancient Egyptian tombs!",
                    "Interesting fact: Octopuses have three hearts and blue blood!",
                    "Fun fact: The word 'set' has the most different meanings in the English language!",
                    "Did you know? A day on Venus is longer than its year!"
                ]
            },
            
            'learning': {
                'patterns': [
                    r'\b(how do you learn|do you learn|can you learn)\b',
                    r'\b(are you learning|do you remember)\b'
                ],
                'responses': [
                    "I work with predefined rules and patterns! I can remember things during our conversation, like your name.",
                    "I'm a rule-based chatbot, so I follow programmed patterns. I can remember details from our current chat!",
                    "I don't learn like humans do, but I can remember information you share with me during our conversation!",
                    "I use pattern matching and rules to respond. I can keep track of our conversation context!"
                ]
            },
            
            'location': {
                'patterns': [
                    r'\b(where are you|your location|where do you live)\b',
                    r'\b(where are you from|what\'s your location)\b'
                ],
                'responses': [
                    "I exist in the digital world! I'm running on a computer, so I don't have a physical location.",
                    "I'm a chatbot, so I live in the code! I don't have a physical location like humans do.",
                    "I exist wherever this program is running! I'm digital, so no physical location for me.",
                    "I'm in cyberspace! As a chatbot, I don't have a geographical location."
                ]
            },
            
            'how_are_you': {
                'patterns': [
                    r'\b(how are you|how do you do|how are you doing|how\'s it going|hows it going)\b',
                    r'\b(how are things|how you doing|what\'s up|whats up)\b'
                ],
                'responses': [
                    "I'm doing great! Thanks for asking. How about you?",
                    "I'm fantastic! How are you today?",
                    "I'm doing well, thank you! How can I help you?",
                    "All good here! How are you feeling?",
                    "I'm excellent! What about you?"
                ]
            },
            
            'name_query': {
                'patterns': [
                    r'\b(what\'s your name|whats your name|your name|who are you)\b',
                    r'\b(what do i call you|what should i call you)\b'
                ],
                'responses': [
                    "I'm a rule-based chatbot! You can call me ChatBot.",
                    "My name is ChatBot. I'm here to help you!",
                    "I'm ChatBot, your friendly AI assistant!",
                    "You can call me ChatBot. What's your name?"
                ]
            },
            
            'user_name': {
                'patterns': [
                    r'\b(my name is|i am|i\'m|call me)\s+(\w+)',
                    r'\b(name\'s|names)\s+(\w+)'
                ],
                'responses': [
                    "Nice to meet you, {name}!",
                    "Hello {name}! Great to know your name.",
                    "Pleasure to meet you, {name}!",
                    "Hi {name}! Thanks for telling me your name."
                ]
            },
            
            'weather': {
                'patterns': [
                    r'\b(weather|rain|sunny|cloudy|hot|cold|temperature)\b',
                    r'\b(how\'s the weather|hows the weather|weather like)\b'
                ],
                'responses': [
                    "I can't check the weather right now, but I hope it's nice where you are!",
                    "I don't have access to weather data, but I hope you're having good weather!",
                    "Sorry, I can't check the weather, but I hope it's pleasant outside!",
                    "I wish I could tell you about the weather, but I don't have that information."
                ]
            },
            
            'feelings': {
                'patterns': [
                    r'\b(i feel|i am feeling|feeling)\s+(good|great|bad|sad|happy|excited|tired|angry)\b',
                    r'\b(i\'m\s+(good|great|bad|sad|happy|excited|tired|angry))\b'
                ],
                'responses': [
                    "I understand you're feeling {emotion}. Thanks for sharing that with me!",
                    "It's good that you're sharing how you feel. I hope things get better if you're not feeling great!",
                    "I hear you're feeling {emotion}. Is there anything I can help you with?",
                    "Thanks for letting me know how you're feeling!"
                ]
            },
            
            'goodbye': {
                'patterns': [
                    r'\b(bye|goodbye|see you|farewell|take care|gotta go|have to go)\b',
                    r'\b(see you later|talk to you later|catch you later|until next time)\b'
                ],
                'responses': [
                    "Goodbye! It was nice chatting with you!",
                    "Take care! Have a great day!",
                    "See you later! Thanks for the conversation!",
                    "Farewell! Come back anytime!",
                    "Bye! Hope to talk to you again soon!"
                ]
            },
            
            'thanks': {
                'patterns': [
                    r'\b(thank you|thanks|thank u|thx|appreciate)\b',
                    r'\b(thanks a lot|thank you very much|much appreciated)\b'
                ],
                'responses': [
                    "You're welcome! Happy to help!",
                    "No problem at all!",
                    "Glad I could help!",
                    "You're very welcome!",
                    "Anytime! That's what I'm here for!",
                    "My pleasure! Feel free to ask me anything else!",
                    "Don't mention it! I'm here to help!"
                ]
            },
            
            'sorry': {
                'patterns': [
                    r'\b(sorry|apologize|my bad|my mistake)\b',
                    r'\b(i\'m sorry|im sorry|forgive me)\b'
                ],
                'responses': [
                    "No worries at all! No need to apologize!",
                    "It's totally fine! Don't worry about it!",
                    "No problem! These things happen!",
                    "All good! No need to say sorry!",
                    "Don't worry about it! We're all good!",
                    "It's okay! No harm done!"
                ]
            },
            
            'please': {
                'patterns': [
                    r'\b(please|pls|plz)\b',
                    r'\b(can you please|could you please|would you please)\b'
                ],
                'responses': [
                    "Of course! I'd be happy to help!",
                    "Absolutely! What can I do for you?",
                    "Sure thing! How can I assist you?",
                    "Certainly! I'm here to help!",
                    "No problem! What do you need?"
                ]
            },
            
            'yes_no': {
                'patterns': [
                    r'\b(yes|yeah|yep|yup|sure|okay|ok|alright)\b',
                    r'\b(no|nope|nah|not really|don\'t think so)\b'
                ],
                'responses': [
                    "I understand! Is there anything else I can help you with?",
                    "Got it! What else would you like to know?",
                    "Okay! Feel free to ask me anything else!",
                    "Alright! I'm here if you need anything more!",
                    "I see! Anything else on your mind?"
                ]
            },
            
            'excuse_me': {
                'patterns': [
                    r'\b(excuse me|pardon|pardon me)\b',
                    r'\b(can i ask|may i ask|i have a question)\b'
                ],
                'responses': [
                    "Of course! What can I help you with?",
                    "Yes! Go ahead, I'm listening!",
                    "Absolutely! What's your question?",
                    "Sure! I'm here to help!",
                    "Yes, please! What would you like to know?"
                ]
            },
            
            'welcome': {
                'patterns': [
                    r'\b(welcome|you\'re welcome|your welcome)\b',
                    r'\b(don\'t mention it|no worries|no problem)\b'
                ],
                'responses': [
                    "Thank you! You're very kind!",
                    "That's so nice of you! I appreciate it!",
                    "Thanks! You're great to talk to!",
                    "How thoughtful! Thank you!",
                    "I appreciate your kindness!"
                ]
            },
            
            'confusion': {
                'patterns': [
                    r'\b(what|huh|what do you mean|i don\'t understand)\b',
                    r'\b(confused|confusing|unclear|what are you saying)\b'
                ],
                'responses': [
                    "Let me clarify! What specifically would you like me to explain?",
                    "I can help clear that up! What part was confusing?",
                    "No problem! What would you like me to explain better?",
                    "I understand the confusion! What can I clarify for you?",
                    "Let me help explain! What specific part didn't make sense?"
                ]
            },
            
            'agreement': {
                'patterns': [
                    r'\b(i agree|exactly|absolutely|definitely|for sure)\b',
                    r'\b(you\'re right|that\'s right|that\'s correct|true)\b'
                ],
                'responses': [
                    "Great! I'm glad we're on the same page!",
                    "Awesome! It's nice when we agree!",
                    "Exactly! We think alike!",
                    "Perfect! Glad I could help clarify that!",
                    "Yes! We're definitely thinking the same way!"
                ]
            },
            
            'disagreement': {
                'patterns': [
                    r'\b(i disagree|disagree|not really|i don\'t think so)\b',
                    r'\b(that\'s wrong|you\'re wrong|incorrect|not right)\b'
                ],
                'responses': [
                    "I understand we might see things differently! That's okay!",
                    "Thanks for sharing your perspective! Everyone has different views!",
                    "I appreciate your input! Different opinions make conversations interesting!",
                    "Fair enough! We don't always have to agree!",
                    "I respect your viewpoint! It's good to have different perspectives!"
                ]
            },
            
            'politeness': {
                'patterns': [
                    r'\b(good morning|good afternoon|good evening|good night)\b',
                    r'\b(have a good day|have a nice day|take care|see you)\b'
                ],
                'responses': [
                    "Thank you! You have a wonderful day too!",
                    "Same to you! Take care!",
                    "That's very kind! Wishing you all the best!",
                    "Thank you! Hope your day is amazing!",
                    "So thoughtful! Have a fantastic day!"
                ]
            },
            
            'small_talk': {
                'patterns': [
                    r'\b(nice weather|beautiful day|lovely day)\b',
                    r'\b(how\'s your day|having a good day|good day)\b'
                ],
                'responses': [
                    "That sounds lovely! I hope you're enjoying it!",
                    "It's always nice to hear about good weather!",
                    "That's wonderful! Perfect day for a chat!",
                    "Sounds like a great day! I hope you make the most of it!",
                    "How nice! I hope your day continues to be wonderful!"
                ]
            },
            
            'acknowledgment': {
                'patterns': [
                    r'\b(i see|i understand|got it|makes sense)\b',
                    r'\b(that makes sense|i get it|ah i see|oh i see)\b'
                ],
                'responses': [
                    "Great! I'm glad that was helpful!",
                    "Perfect! Anything else you'd like to know?",
                    "Excellent! Is there anything else I can help with?",
                    "Wonderful! Feel free to ask me anything else!",
                    "Awesome! I'm here if you need more help!"
                ]
            },
            
            'help': {
                'patterns': [
                    r'\b(help|assist|support|what can you do)\b',
                    r'\b(how can you help|what do you do)\b'
                ],
                'responses': [
                    "I can help with: time/date info, basic math, conversations, jokes, motivation, fun facts, and more! Try asking 'what can you do?' for details.",
                    "I'm here to chat and help! I can tell time, solve math problems, share jokes, give motivation, and have friendly conversations!",
                    "I can assist with: current time/date, calculations, chatting, fun facts, jokes, encouragement, and answering questions!",
                    "I'm a helpful chatbot! Ask me about time, give me math problems, request jokes, or just chat with me!"
                ]
            },
            
            'calculator_help': {
                'patterns': [
                    r'\b(how to calculate|how do i calculate|calculator)\b',
                    r'\b(math help|how to do math)\b'
                ],
                'responses': [
                    "Just ask me math questions like: 'What is 15 + 7?', 'Calculate 20 * 3', or 'Solve 100 / 4'",
                    "I can do addition (+), subtraction (-), multiplication (*), and division (/). Try: 'What is 25 - 8?'",
                    "For math help, just ask! Examples: 'Calculate 50 + 25', 'What is 12 * 6?', 'Solve 45 / 9'"
                ]
            }
        }
    
    def preprocess_text(self, text):
        """Preprocess the input text using NLTK"""
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and token.isalpha()
        ]
        
        return ' '.join(processed_tokens), text
    
    def get_current_time(self):
        """Get current time formatted"""
        now = datetime.datetime.now()
        return now.strftime("%I:%M %p")
    
    def get_current_date(self):
        """Get current date formatted"""
        now = datetime.datetime.now()
        return now.strftime("%A, %B %d, %Y")
    
    def get_current_datetime(self):
        """Get current date and time together"""
        now = datetime.datetime.now()
        return now.strftime("%A, %B %d, %Y at %I:%M %p")
    
    def get_day_of_week(self):
        """Get current day of the week"""
        now = datetime.datetime.now()
        return now.strftime("%A")
    
    def get_month_year(self):
        """Get current month and year"""
        now = datetime.datetime.now()
        return now.strftime("%B %Y")
    
    def calculate_age(self, birth_year):
        """Calculate age from birth year"""
        current_year = datetime.datetime.now().year
        return current_year - birth_year
    
    def solve_math(self, expression):
        """Solve basic math expressions"""
        try:
            # Extract numbers and operator
            import re
            match = re.search(r'(\d+)\s*([+\-*/])\s*(\d+)', expression)
            if match:
                num1, operator, num2 = match.groups()
                num1, num2 = int(num1), int(num2)
                
                if operator == '+':
                    return num1 + num2
                elif operator == '-':
                    return num1 - num2
                elif operator == '*' or operator == 'x':
                    return num1 * num2
                elif operator == '/':
                    if num2 != 0:
                        return round(num1 / num2, 2)
                    else:
                        return "Error: Cannot divide by zero!"
            return None
        except:
            return None
    
    def match_pattern(self, user_input):
        """Match user input against predefined rules"""
        processed_input, original_input = self.preprocess_text(user_input)
        
        for intent, rule in self.rules.items():
            for pattern in rule['patterns']:
                match = re.search(pattern, original_input, re.IGNORECASE)
                if match:
                    return intent, match, rule['responses']
        
        return None, None, None
    
    def generate_response(self, intent, match, responses, user_input):
        """Generate response based on matched intent"""
        response = random.choice(responses)
        
        # Handle special cases
        if intent == 'time_query':
            response = response.format(time=self.get_current_time())
        
        elif intent == 'date_query':
            response = response.format(date=self.get_current_date())
        
        elif intent == 'datetime_query':
            response = response.format(datetime=self.get_current_datetime())
        
        elif intent == 'day_query':
            response = response.format(day=self.get_day_of_week())
        
        elif intent == 'month_year_query':
            response = response.format(month_year=self.get_month_year())
        
        elif intent == 'age_calculation' and match:
            # Extract birth year
            birth_year_match = re.search(r'(\d{4})', user_input)
            if birth_year_match:
                birth_year = int(birth_year_match.group(1))
                age = self.calculate_age(birth_year)
                response = response.format(birth_year=birth_year, age=age)
        
        elif intent == 'math_calculation':
            result = self.solve_math(user_input)
            if result is not None:
                response = response.format(result=result)
            else:
                response = "I couldn't solve that math problem. Try a simpler format like '5 + 3' or '10 * 4'"
        
        elif intent == 'user_name' and match:
            # Extract name from match
            name = match.group(2) if match.group(2) else match.group(1)
            self.user_name = name.capitalize()
            response = response.format(name=self.user_name)
        
        elif intent == 'feelings' and match:
            # Extract emotion
            emotion_match = re.search(r'\b(good|great|bad|sad|happy|excited|tired|angry)\b', user_input, re.IGNORECASE)
            if emotion_match:
                emotion = emotion_match.group(1)
                response = response.format(emotion=emotion)
        
        return response
    
    def get_default_response(self):
        """Return default response for unmatched input"""
        default_responses = [
            "I'm not sure I understand. Can you rephrase that?",
            "That's interesting! Can you tell me more?",
            "I'm still learning. Could you ask me something else?",
            "Hmm, I'm not sure about that. What else would you like to know?",
            "I don't quite get that. Try asking me about the time, math, or just say hello!",
            "Sorry, I didn't catch that. I can help with greetings, time, math, jokes, and basic conversation!",
            "Could you rephrase that? I'm here to help with time, calculations, chat, and more!",
            "I'm not sure what you mean. Try asking 'what can you do?' to see how I can help!",
            "That's a bit unclear to me. Feel free to ask about time, math, or just chat with me!",
            "I didn't quite understand that. Ask me for the time, a joke, or just say hi!"
        ]
        return random.choice(default_responses)
    
    def chat(self, user_input):
        """Main chat function"""
        if not user_input.strip():
            return "Please say something!"
        
        # Check for quit commands
        if user_input.lower().strip() in ['quit', 'exit', 'stop']:
            return "Goodbye! Thanks for chatting with me!"
        
        # Match input against rules
        intent, match, responses = self.match_pattern(user_input)
        
        if intent:
            response = self.generate_response(intent, match, responses, user_input)
        else:
            response = self.get_default_response()
        
        return response

# Example usage and testing
def main():
    """Main function to run the chatbot"""
    print("Rule-Based Chatbot with NLTK")
    print("="*40)
    print("Hello! I'm a rule-based chatbot. Type 'quit' to exit.")
    print("I can greet you, tell time, and have basic conversations!")
    print("="*40)
    
    chatbot = RuleBasedChatbot()
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'stop']:
            print("ChatBot: Goodbye! Thanks for chatting with me!")
            break
        
        if user_input:
            response = chatbot.chat(user_input)
            print(f"ChatBot: {response}")
        else:
            print("ChatBot: Please say something!")

# Test the chatbot with example conversations
def test_chatbot():
    """Test function to demonstrate chatbot capabilities"""
    print("\n" + "="*50)
    print("TESTING CHATBOT FUNCTIONALITY")
    print("="*50)
    
    chatbot = RuleBasedChatbot()
    
    test_inputs = [
        "Hello!",
        "Thank you so much!",
        "Sorry about that",
        "Please help me",
        "Yes, that's right",
        "No, I don't think so",
        "Excuse me, can I ask something?",
        "You're welcome!",
        "I don't understand",
        "I agree with you",
        "I disagree",
        "Good morning!",
        "Have a nice day!",
        "Nice weather today",
        "I see, that makes sense",
        "What time is it?",
        "What's today's date?",
        "What's the date and time?",
        "What day is it?",
        "What month and year?",
        "How are you?",
        "My name is Sarah",
        "What's your name?",
        "I was born in 1995",
        "What is 15 + 7?",
        "Calculate 20 * 3",
        "Solve 100 / 4",
        "Tell me a joke",
        "Motivate me",
        "Fun fact",
        "What can you do?",
        "I'm feeling happy",
        "How's the weather?",
        "You're awesome",
        "Where are you?",
        "How do you learn?",
        "Help me with math",
        "Goodbye!"
    ]
    
    for test_input in test_inputs:
        response = chatbot.chat(test_input)
        print(f"User: {test_input}")
        print(f"Bot: {response}")
        print("-" * 30)

if __name__ == "__main__":
    # Run tests first
    test_chatbot()
    
    # Then start interactive chat
    main()
    