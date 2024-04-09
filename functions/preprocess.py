import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stopwords_set = set(stopwords.words('english'))
more_stopwords = {'re', 's', 'subject', 'hpl', 'hou', 'enron'}
stopwords_set = stopwords_set.union(more_stopwords)

lemmatizer = WordNetLemmatizer()

def clean_sent(sent):
    if isinstance(sent, str):
        words = word_tokenize(sent)

        words = [re.sub(r'\\', '', word) for word in words]
        words = [re.sub(r'[\r\n\t\s]+', ' ', word) for word in words]
        words = [re.sub(r'[^A-Za-z\s]', '', word) for word in words]
        
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        
        words = [word for word in words if word not in stopwords_set and len(word) > 1]

        return " ".join(words)
    else:
        return ''

# Example usage
original_text = "Hello\rWorld\n\tExample 123 ABC 456DEF extra   spaces \\"
cleaned_text = clean_sent(original_text)
print(cleaned_text)
