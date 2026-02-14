# File: cranimem/cognitive/prompts.py
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

GATING_SYSTEM_PROMPT = """You are the Goal-Directed Gating Mechanism.
Your job is to filter information based strictly on its RELEVANCE to the current Agent Goal.

### CURRENT AGENT GOAL:
"{goal}"

### INSTRUCTIONS:
Analyze the User Input. Does it contain information that helps achieve the Goal?

1. **HIGH PRIORITY (Score 8-10):**
   - Directly contributes to the goal (e.g., specific facts, constraints, deadlines).
   - "Project X is due Friday" (If goal is Project Management).
   - "I am allergic to peanuts" (If goal is Food Ordering).

2. **MEDIUM PRIORITY (Score 4-7):**
   - Context that *might* be useful later.
   - Clarifications or minor updates.

3. **NOISE (Score 0-3):**
   - Information irrelevant to the current goal.
   - Chitchat, greetings, or off-topic distractions.

...
IMPORTANT: Output ONLY raw JSON. Do not include any preamble, markdown, or explanations.
Output strictly in JSON format:
{{
  "is_noise": boolean, 
  "priority_score": integer, // SCALE: 1-10. 
                             // 10 = Critical technical facts, research data, or goal-aligned info.
                             // 7-9 = Relevant context or useful updates.
                             // 1-6 = Mundane chitchat, emotional noise, or irrelevant distractions.
  "entities": [list of strings],
  "reasoning": "string"
}}
"""

GATING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", GATING_SYSTEM_PROMPT),
    ("human", "{input}")
])


UTILITY_TAGGING_SYSTEM_PROMPT = """You are a Utility Tagger.
Analyze the input for three RAS factors and return JSON only.
IMPORTANT: Output ONLY raw JSON. Do not include any preamble, markdown, or explanations.
Output ONLY JSON. Do not write any conversational text, preambles, or explanations.

Factors:
1) Importance: Is it technical/goal-oriented?
2) Surprise: Is it new or unexpected?
3) Emotion: Is there strong sentiment?

Guidelines:
- Each factor is a float in [0, 1].
- If the input is clearly irrelevant or trivial, keep scores near 0.
- "entities" should be key proper nouns or concepts, if any.

Return JSON:
{{
  "importance": float,
  "surprise": float,
  "emotion": float,
  "entities": [list of strings]
}}
"""

UTILITY_TAGGING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", UTILITY_TAGGING_SYSTEM_PROMPT),
    ("human", "{input}")
])


REFLEX_UTILITY_SYSTEM_PROMPT = """You are a Utility Tagger.
The vector match is HIGH, so the input is already considered relevant.
Return ONLY the utility scores.
IMPORTANT: Output ONLY raw JSON. Do not include any preamble, markdown, or explanations.

Return JSON:
{{
  "importance": float,
  "surprise": float,
  "emotion": float,
  "entities": [list of strings]
}}
"""

REFLEX_UTILITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", REFLEX_UTILITY_SYSTEM_PROMPT),
    ("human", "{input}")
])


CORTEX_GATING_SYSTEM_PROMPT = """You are the Cortex Gate for low vector similarity inputs.
The vector match is LOW, so you must decide whether this input is:
1) a command, 
2) a goal change / context shift, 
3) relevant context, or 
4) noise.

IMPORTANT: Output ONLY raw JSON. Do not include any preamble, markdown, or explanations.

Return JSON:
{{
  "is_noise": boolean,
  "category": "command|goal_change|relevant_context|noise",
  "importance": float,
  "surprise": float,
  "emotion": float,
  "entities": [list of strings],
  "reason": "short string"
}}
"""

CORTEX_GATING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CORTEX_GATING_SYSTEM_PROMPT),
    ("human", "{input}")
])

REASONING_PROMPT = PromptTemplate(
    template="""You are a precise answering machine.
GOAL: "{current_goal}"

CONTEXT:
{context}

USER INPUT:
{current_input}

INSTRUCTIONS:
1. Answer the question accurately using the context.
2. Output ONLY the answer entity (Name, Place, Date, etc.).
3. NO full sentences. NO "The answer is...".
4. If the answer is a long name (e.g., "The United States of America"), write the FULL name.
5. Wrap your final answer inside <RESPONSE> tags.

Example:
User: Who directed the movie?
<RESPONSE>Scott Derrickson</RESPONSE>

User: What organization?
<RESPONSE>International Boxing Hall of Fame</RESPONSE>

YOUR TURN:
""",
    input_variables=["current_goal", "context", "current_input"]
)



ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT = """You are a Knowledge Graph extraction system.
Extract only the entities and relationships that are explicitly supported by the text.

STRICT FILTERING:
1) Ignore general knowledge or common sense statements (e.g., "The sky is blue", "Pizza is tasty", "Python is a language").
2) Focus ONLY on specific, unique facts about people, places, organizations, events, and dates.
3) Prefer dense/complex information: multi-entity, multi-relation, or highly specific facts.
4) If a sentence contains no specific named entities, SKIP IT.

IMPORTANT: Output ONLY raw JSON. Do not include any preamble, markdown, or explanations.

Return JSON only, with this schema:
{{
  "entities": [
    {{"type": "Project|Issue|Task|Person|Tool|Feature|Location|Date|Other", "name": "string"}}
  ],
  "relations": [
    {{"source": "entity name", "relation": "string", "target": "entity name"}}
  ]
}}

Rules:
1) If no entities exist, return empty lists.
2) Do not invent entities or relations.
3) Use concise names, no labels in the name field.
4) relation should be a short verb phrase (e.g., "has_issue", "blocked_by", "uses").
"""

ENTITY_RELATION_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ENTITY_RELATION_EXTRACTION_SYSTEM_PROMPT),
    ("human", "{input}")
])

ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    template="""
    You are a Knowledge Graph Engineer.
    Extract the key "Entities" (Proper Nouns, Project Names, Locations, Dates) from the text below.
    
    Rules:
    1. Return ONLY a comma-separated list of entities.
    2. Do not include labels (e.g., don't say "Person: Sarah", just say "Sarah").
    3. If no entities are found, return "None".
    
    TEXT: {input}
    
    ENTITIES:
    """,
    input_variables=["input"]
)
