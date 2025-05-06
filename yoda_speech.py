import random
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize NLTK
lemmatizer = WordNetLemmatizer()
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def invert_sentence(sentence):
    # Rearrange sentence to mimic Yoda's object-subject-verb structure.
    tokens = word_tokenize(sentence)
    if len(tokens) < 2:
        return sentence
    
    # Simple inversion: move last few words to front
    if random.choice([True, False]):
        if tokens[-1] in '.!?':
            end_punct = tokens.pop(-1)
        else:
            end_punct = '.'
        
        # Move last phrase to front
        phrase_length = random.randint(1, min(3, len(tokens)))
        phrase = tokens[-phrase_length:]
        tokens = tokens[:-phrase_length]
        inverted = phrase + tokens
        return ' '.join(inverted).capitalize() + end_punct
    return sentence

def transform_to_yoda_speech(response, user_input=""):
    #Transform a response to sound like Yoda and filter out hallucinated prompts.
    # Filter out hallucinated user prompts (e.g., text between * *)
    response = re.sub(r'\*.*?\*', '', response).strip()
    
    # Easter eggs
    if "sith" in user_input.lower():
        return "Dark side, I sense in you! Careful, you must be."
    if "jedi" in user_input.lower():
        return "A Jedi, you wish to be? Train hard, you must."
    if "destiny" in user_input.lower():
        return "Your destiny, clouded it is. Seek the Force, you must."
    if "force" in user_input.lower():
        return "Strong, the Force is with you. Guide you, it will."
    if "hope" in user_input.lower():
        return "A new hope, you bring. Bright, the future is."
    if "dark" in user_input.lower():
        return "To the dark side, you turn? Resist, you must!"
    if "light" in user_input.lower():
        return "The light side, you seek? Balance, you must find."
    
    # Invert sentence structure
    yoda_response = invert_sentence(response)
    
    # Add Yoda flair
    if random.random() < 0.3:
        yoda_response = f"Mmm, {yoda_response.lower()}"
    if random.random() < 0.2:
        yoda_response += " Young one."
    if random.random() < 0.15:
        yoda_response = f"Patience, you must have. {yoda_response}"
    
    return yoda_response