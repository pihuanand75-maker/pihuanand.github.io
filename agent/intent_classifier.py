"""
Classifies user input before routing to the SQL pipeline.
Prevents the SQL agent from trying to execute non-data queries.
"""
import re


class IntentClassifier:
    """
    Intent types:
      - DATA_QUERY: Requires SQL generation and execution
      - META_QUESTION: Question about the data schema/structure (no SQL needed)
      - GREETING: Hello, hi, thanks, etc.
      - HELP: Questions about how to use the tool
      - UNKNOWN: Default – route to DATA_QUERY with a caveat
    """

    GREETING_PATTERNS = [
        r'^(hi|hello|hey|howdy|sup|yo|greetings)\b',
        r'^(thanks?|thank you|ty)\b',
        r'^(good morning|good afternoon|good evening)\b',
    ]

    META_PATTERNS = [
        r'\b(what (tables?|columns?|fields?)|show (me )?(the )?(tables?|schema|columns?|fields?))',
        r'\b(how many (tables?|columns?|rows?))\b',
        r'\b(list (all )?(tables?|columns?))\b',
        r'\b(what (data|information) (do|does|is) (we|I|it) have)\b',
        r'\b(describe (the )?(table|data|schema|dataset))\b',
        r'\b(what (are|is) the (column|field) names?)\b',
    ]

    HELP_PATTERNS = [
        r'\b(how do I|how to|can you help|what can you do|what are your|instructions)\b',
        r'\b(help|tutorial|guide|examples?)\b',
    ]

    @classmethod
    def classify(cls, user_input: str) -> str:
        text = user_input.lower().strip()

        for pattern in cls.GREETING_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return 'GREETING'

        for pattern in cls.HELP_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return 'HELP'

        for pattern in cls.META_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return 'META_QUESTION'

        return 'DATA_QUERY'

    @staticmethod
    def greeting_response() -> str:
        return (
            "Hello! I'm your RAG SQL Agent. Upload a data file (CSV, Excel, JSON, or Parquet) "
            "using the sidebar, then ask me questions about your data in plain English. "
            "I'll write and execute the SQL automatically."
        )

    @staticmethod
    def help_response() -> str:
        return (
            "**How to use this agent:**\n\n"
            "1. **Upload your data** using the file uploader in the left sidebar\n"
            "2. **Ask questions in plain English** – no SQL knowledge needed\n"
            "3. **Review the SQL** I generate – shown in an expandable section\n"
            "4. **Explore follow-up questions** suggested at the end of each answer\n\n"
            "**Example questions:**\n"
            "- 'Show me the top 10 customers by revenue'\n"
            "- 'What is the average order value by month?'\n"
            "- 'Which products have the highest return rate?'\n"
            "- 'Find all transactions where the amount is greater than 10000'\n\n"
            "**Supported file formats:** CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet"
        )
