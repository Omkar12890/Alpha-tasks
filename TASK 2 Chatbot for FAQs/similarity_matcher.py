"""
Similarity matching module using cosine similarity
Matches user queries with FAQ questions
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessor import preprocessor


class FAQMatcher:
    """Matches user questions with FAQ database using cosine similarity"""
    
    def __init__(self, faq_database):
        """
        Initialize the FAQ matcher
        
        Args:
            faq_database: List of FAQ dictionaries with 'question' and 'answer' keys
        """
        self.faq_database = faq_database
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9
        )
        
        # Preprocess and vectorize FAQ questions
        self.preprocess_faqs()
    
    def preprocess_faqs(self):
        """Preprocess and vectorize all FAQ questions"""
        # Get preprocessed FAQ questions
        self.faq_questions = [
            preprocessor.preprocess_to_string(faq['question'])
            for faq in self.faq_database
        ]
        
        # Vectorize using TF-IDF
        self.faq_vectors = self.vectorizer.fit_transform(self.faq_questions)
    
    def find_best_match(self, user_query, threshold=0.3):
        """
        Find the best matching FAQ for a user query
        
        Args:
            user_query: User's question string
            threshold: Minimum similarity score (0-1) to consider a match
            
        Returns:
            Dictionary with 'answer', 'similarity_score', 'matched_question', and 'faq_index'
            or None if no match meets threshold
        """
        # Preprocess user query
        processed_query = preprocessor.preprocess_to_string(user_query)
        
        # Vectorize user query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity with all FAQ questions
        similarities = cosine_similarity(query_vector, self.faq_vectors)[0]
        
        # Find best match
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Check if similarity meets threshold
        if best_similarity < threshold:
            return {
                'answer': "I'm sorry, I couldn't find a matching answer in our FAQ database. Please contact our support team for assistance.",
                'similarity_score': float(best_similarity),
                'matched_question': None,
                'faq_index': -1,
                'confidence': 'low'
            }
        
        # Return best match
        return {
            'answer': self.faq_database[best_match_idx]['answer'],
            'similarity_score': float(best_similarity),
            'matched_question': self.faq_database[best_match_idx]['question'],
            'faq_index': best_match_idx,
            'confidence': 'high' if best_similarity > 0.6 else 'medium'
        }
    
    def find_top_matches(self, user_query, top_n=3, threshold=0.2):
        """
        Find top N matching FAQs for a user query
        
        Args:
            user_query: User's question string
            top_n: Number of top matches to return
            threshold: Minimum similarity score
            
        Returns:
            List of match dictionaries sorted by similarity score (descending)
        """
        # Preprocess user query
        processed_query = preprocessor.preprocess_to_string(user_query)
        
        # Vectorize user query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate cosine similarity with all FAQ questions
        similarities = cosine_similarity(query_vector, self.faq_vectors)[0]
        
        # Get indices of top matches
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        matches = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                matches.append({
                    'answer': self.faq_database[idx]['answer'],
                    'similarity_score': float(similarities[idx]),
                    'matched_question': self.faq_database[idx]['question'],
                    'faq_index': idx,
                    'confidence': 'high' if similarities[idx] > 0.6 else 'medium'
                })
        
        return matches
    
    def get_faq_count(self):
        """Get the total number of FAQs in the database"""
        return len(self.faq_database)
