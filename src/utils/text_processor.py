import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text cleaning pipeline from your notebook"""
    
    def __init__(self):
        # Download NLTK data if not present
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean(self, text: str) -> str:
        """
        Clean text exactly as in your notebook
        Converts to lowercase, removes stopwords and non-alphabetic tokens
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            text = text.lower()
            tokens = word_tokenize(text)
            # Keep only alphabetic tokens that aren't stopwords
            tokens = [w for w in tokens if w.isalpha() and w not in self.stop_words]
            return " ".join(tokens)
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return ""
    
    def extract_phishing_keywords(self, text: str) -> list:
        """
        Extract phishing-related keywords that might indicate scam
        Based on patterns from your notebook analysis
        """
        keywords = [
            'urgent', 'immediately', 'emergency', 'now',
            'bank', 'account', 'password', 'verify', 'confirm',
            'security', 'update', 'suspended', 'locked',
            'payment', 'transfer', 'money', 'credit',
            'click', 'link', 'website', 'login'
        ]
        
        found = []
        text_lower = text.lower()
        
        for keyword in keywords:
            if keyword in text_lower:
                found.append(keyword)
        
        return found
    
    def calculate_urgency_score(self, text: str) -> float:
        """
        Calculate urgency score based on time-sensitive words
        Returns value between 0 and 1
        """
        urgency_words = {
            'urgent': 0.9, 'immediately': 0.95, 'now': 0.8,
            'emergency': 0.9, 'critical': 0.85, 'asap': 0.7,
            'today': 0.6, 'within hours': 0.8, 'deadline': 0.7
        }
        
        text_lower = text.lower()
        scores = []
        
        for word, weight in urgency_words.items():
            if word in text_lower:
                # Count occurrences
                count = text_lower.count(word)
                score = min(weight * (1 + 0.1 * (count - 1)), 1.0)
                scores.append(score)
        
        return max(scores) if scores else 0.0