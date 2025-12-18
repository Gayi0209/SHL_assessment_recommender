from dotenv import load_dotenv
load_dotenv()

from backend.llm.query_parser import parse_query

query = "Hiring a Java developer with strong communication skills"
result = parse_query(query)

print(result)
