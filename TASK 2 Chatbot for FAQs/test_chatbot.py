"""
Test script for FAQ Chatbot
Demonstrates the chatbot functionality
"""

from chatbot import FAQChatbot
from faq_data import FAQ_DATABASE, TOPIC


def print_separator():
    print("\n" + "="*80 + "\n")


def demo_chatbot():
    """Run a demonstration of the chatbot"""
    
    print_separator()
    print(f"🤖 FAQ CHATBOT - {TOPIC.upper()}")
    print_separator()
    
    # Initialize chatbot
    chatbot = FAQChatbot()
    
    print(f"\n📊 Chatbot initialized with {chatbot.get_stats()['total_faqs']} FAQs\n")
    
    # Test queries
    test_queries = [
        "How long is the warranty?",
        "I forgot my password, what do I do?",
        "What payment methods are available?",
        "How can I track my order?",
        "Do you have a mobile app?",
        "What are your support options?",
        "Tell me about shipping times",
        "Can I get a refund?",
        "How do I upgrade my plan?",
        "What's your privacy policy about?",
    ]
    
    print("=" * 80)
    print("DEMO: Testing chatbot with various queries")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        print("-" * 80)
        
        # Get response
        response = chatbot.get_response(query, threshold=0.2)
        
        print(f"✅ Answer: {response['answer'][:150]}...")
        print(f"📊 Similarity Score: {response['similarity_score']:.2%}")
        print(f"🎯 Confidence: {response['confidence'].upper()}")
        if response['matched_question']:
            print(f"🔗 Matched FAQ: {response['matched_question']}")
        
        # Show similar questions
        similar = chatbot.get_similar_faqs(query, top_n=2)
        if similar:
            print(f"\n   Related questions:")
            for j, match in enumerate(similar, 1):
                print(f"   {j}. {match['matched_question']} ({match['similarity_score']:.2%})")
    
    print_separator()
    print("DEMO: Interactive Mode")
    print_separator()
    print("\nNow entering interactive mode. Type 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thank you for using FAQ Chatbot!")
                break
            
            if not user_input:
                print("Please enter a question.\n")
                continue
            
            # Get response
            response = chatbot.get_response(user_input)
            
            print(f"\n🤖 Bot: {response['answer']}")
            print(f"   Confidence: {response['confidence']} ({response['similarity_score']:.2%})")
            
            # Show similar matches
            similar = chatbot.get_similar_faqs(user_input, top_n=2)
            if similar:
                print(f"   Similar questions:")
                for j, match in enumerate(similar[:2], 1):
                    print(f"   {j}. {match['matched_question']}")
            
            print()
        
        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using FAQ Chatbot!")
            break


if __name__ == "__main__":
    demo_chatbot()
