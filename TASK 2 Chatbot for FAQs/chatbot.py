"""
Main chatbot module
Orchestrates FAQ matching and response generation
"""

from faq_data import FAQ_DATABASE, TOPIC
from similarity_matcher import FAQMatcher
from datetime import datetime


class FAQChatbot:
    """Main chatbot class for handling FAQ queries"""
    
    def __init__(self, faq_database=None):
        """
        Initialize the chatbot
        
        Args:
            faq_database: List of FAQ dictionaries (uses default if not provided)
        """
        self.faq_database = faq_database or FAQ_DATABASE
        self.matcher = FAQMatcher(self.faq_database)
        self.conversation_history = []
        self.topic = TOPIC
    
    def get_response(self, user_query, threshold=0.3):
        """
        Get a response to a user query
        
        Args:
            user_query: The user's question
            threshold: Minimum similarity score to consider a match
            
        Returns:
            Dictionary with response details
        """
        # Record conversation
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_query': user_query,
            'type': 'user'
        })
        
        # Find best matching FAQ
        match = self.matcher.find_best_match(user_query, threshold=threshold)
        
        # Record chatbot response
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'response': match['answer'],
            'type': 'bot',
            'match_info': {
                'similarity_score': match['similarity_score'],
                'matched_question': match['matched_question'],
                'confidence': match['confidence']
            }
        })
        
        return match
    
    def get_similar_faqs(self, user_query, top_n=3):
        """
        Get multiple similar FAQ matches
        
        Args:
            user_query: The user's question
            top_n: Number of similar results to return
            
        Returns:
            List of matching FAQs
        """
        return self.matcher.find_top_matches(user_query, top_n=top_n)
    
    def get_all_faqs(self):
        """Get all FAQs in the database"""
        return self.faq_database
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
    
    def get_stats(self):
        """Get chatbot statistics"""
        total_queries = len([h for h in self.conversation_history if h['type'] == 'user'])
        
        return {
            'total_faqs': len(self.faq_database),
            'topic': self.topic,
            'total_user_queries': total_queries,
            'total_conversation_turns': len(self.conversation_history)
        }


# Create global chatbot instance
chatbot = FAQChatbot()
