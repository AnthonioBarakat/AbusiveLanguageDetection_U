from load_data import load_data
import re
import string
import spacy
import pandas as pd
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet

# # laod english model
# nlp = spacy.load("en_core_web_sm")

# # download nltk data
# nltk.download('punkt_tab')          # For word_tokenize
# nltk.download('averaged_perceptron_tagger_eng')  # For pos_tag
# nltk.download('wordnet')        # For lemmatizer
# nltk.download('omw-1.4')        # WordNet's multilingual support (helps lemmatization)
# nltk.download('stopwords')   

# Initialize
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

contractions = {
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "don't": "do not",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "we're": "we are",
    "they're": "they are",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "didn't": "did not",
    "doesn't": "does not",
    "couldn't": "could not",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "how's": "how is",
    "there's": "there is",
    "when's": "when is",
    "let's": "let us",
}


# pos tag -> wordnet pos
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def expand_contractions(text):
    pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    return pattern.sub(lambda x: contractions[x.group()], text)

def normalize_abusive(text):
    patterns = [
        (r'a\$+\$', 'ass'),
        (r'f[\*@#%]+k', 'fuck'),
        (r'muth[a@]+f[\*@#%]+', 'motherfuck'),
        (r'sh[\*@#%]+t', 'shit'),
        (r'b[i1]+tch', 'bitch'),
        (r'n[i!1]+gg[a@]+[r]?s?', 'nigger'),
        (r'd[i!1]+c[k]+', 'dick'),
        (r'di[c]+k', 'dick'),
        (r'd1ck', 'dick'),
    ]
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def remove_emojis(text):
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_text(text, is_basic=False):
    # basic cleaning
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    if is_basic:
        return text.strip().lower()
    
    text = expand_contractions(text) # it's -> it is
    text = normalize_abusive(text) # d1ck -> dick
    text = remove_emojis(text) # remove emojies
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # anddddd -> andd(reduce repeated charachter)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)  # remove special characters
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

    # lemmatize
    tagged_tokens = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos)) for word, pos in tagged_tokens]


    # doc = nlp(" ".join(lemmatized))
    # entities = [(ent.text, ent.label_) for ent in doc.ents]

    return " ".join(lemmatized)







