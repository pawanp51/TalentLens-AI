import json
import os
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv

from google import genai
from google.genai import types


OUTPUT_DIR = Path("../artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "hiring_intent.json"

load_dotenv()


DEFAULT_SCHEMA = {

    "role_title": "",
    "seniority": "",

    "experience": {
        "min_years": None,
        "max_years": None,
        "ideal_years": None
    },

    "required_skills": [],
    "preferred_skills": [],

    "required_domains": [],
    "preferred_domains": [],

    "domain_keywords": [],

    "target_titles": [],
    "adjacent_titles": [],
    "transferable_titles": [],

    "title_categories": [],

    "candidate_archetypes": [],

    "hidden_gem_profiles": [],

    "adjacent_backgrounds": [],
    "transferable_backgrounds": [],

    "production_requirements": [],

    "core_business_problems": [],

    "preferred_company_types": [],

    "downweighted_industries": [],
    "downweighted_companies": [],

    "location_preferences": [],

    "notice_period_preference_days": None,

    "disqualifiers": [],

    "positive_title_patterns": [],

    "signal_priorities": {}
}


def _configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not found."
        )

    return genai.Client(api_key=api_key)


def _build_prompt(jd_text: str) -> str:

    schema_json = json.dumps(
        DEFAULT_SCHEMA,
        indent=2
    )

    return f"""
You are a Candidate Retrieval Architect, Search Relevance Engineer, and Talent Intelligence Expert.

Your task is to convert a Job Description into a RETRIEVAL ONTOLOGY.

The output will be used for:

1. Candidate retrieval from a pool of 100,000+ candidates
2. Candidate ranking
3. Hidden-gem candidate discovery
4. False-positive reduction
5. Search query generation
6. Similarity matching

The output is NOT intended for human reading.

The output is intended for:

- BM25 retrieval
- semantic retrieval
- vector search
- candidate ranking systems

Your objective is to maximize:

- candidate recall
- candidate precision
- ranking quality

while minimizing:

- keyword matching bias
- false positives
- irrelevant candidates

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Do NOT summarize the job description.

Do NOT rewrite responsibilities.

Do NOT explain the role.

Do NOT generate recruiter-friendly descriptions.

Instead:

Convert the job description into normalized retrieval entities.

Think like:

- a search engine
- a retrieval system
- a ranking system

Focus on:

- candidate discovery
- candidate retrieval
- candidate ranking

--------------------------------------------------
NORMALIZATION RULES
--------------------------------------------------

Every value must be a canonical retrieval entity.

Use:

snake_case

Examples:

GOOD

search_engineer
ranking_engineer
retrieval_engineer
vector_search
hybrid_retrieval
information_retrieval
marketplace_ml
ads_ranking
personalization
ab_testing
offline_eval
online_eval
production_ml

BAD

engineers_who_built_search_systems

people_with_recommendation_experience

strong_python_programming_skills

experience_in_vector_databases

Never generate sentences.

Never generate explanations.

Prefer:

1-3 words

Maximum:

4 words

--------------------------------------------------
REQUIRED SKILLS
--------------------------------------------------

required_skills should contain:

core capabilities required for success.

Examples:

information_retrieval
ranking_systems
vector_search
python
production_ml
evaluation_frameworks

Do not include explanations.

--------------------------------------------------
PREFERRED SKILLS
--------------------------------------------------

preferred_skills should contain:

valuable but non-critical capabilities.

Examples:

llm_finetuning
learning_to_rank
distributed_systems
inference_optimization

--------------------------------------------------
DOMAIN KEYWORDS
--------------------------------------------------

domain_keywords should contain retrieval terms.

Include:

- technologies
- algorithms
- metrics
- concepts
- frameworks
- business concepts

Examples:

embedding
vector_search
retrieval
ranking
recommendation
personalization
ltr
ndcg
mrr
map
pinecone
qdrant
milvus
faiss
bm25
llm
rag
bert
transformer

Generate comprehensive retrieval keywords.

--------------------------------------------------
TARGET TITLES
--------------------------------------------------

target_titles:

direct matches.

adjacent_titles:

strong alternative titles recruiters should source.

transferable_titles:

titles that may succeed despite not appearing to be direct matches.

Use actual hiring-market titles.

Avoid generic titles.

GOOD

search_engineer
ranking_engineer
retrieval_engineer
relevance_engineer
recommendation_engineer
applied_scientist

BAD

engineer

developer

specialist

--------------------------------------------------
TITLE CATEGORIES
--------------------------------------------------

Generate broad hiring clusters.

Examples:

search
ranking
recommendation
relevance
personalization
information_retrieval
machine_learning
applied_ai

--------------------------------------------------
HIDDEN GEM PROFILES
--------------------------------------------------

Generate overlooked candidate types likely to succeed.

Examples:

ads_ranking_engineer
marketplace_ml_engineer
personalization_engineer
recommendation_engineer
search_engineer
relevance_engineer

Return candidate types.

Not descriptions.

--------------------------------------------------
ADJACENT BACKGROUNDS
--------------------------------------------------

Generate adjacent domains that produce strong candidates.

Examples:

ecommerce_search
adtech_ranking
social_feed_ranking
content_recommendation
marketplace_matching

--------------------------------------------------
TRANSFERABLE BACKGROUNDS
--------------------------------------------------

Generate non-obvious backgrounds that transfer well.

Examples:

backend_engineering
distributed_systems
ml_platforms
data_infrastructure

--------------------------------------------------
PRODUCTION REQUIREMENTS
--------------------------------------------------

Only production signals.

Do NOT include responsibilities.

GOOD

embedding_drift
retrieval_regression
index_refresh
ab_testing
offline_eval
online_eval
production_ranking
production_recommendation
model_monitoring
serving_infrastructure

BAD

mentor_engineers

work_with_pm

build_features

--------------------------------------------------
BUSINESS PROBLEMS
--------------------------------------------------

Generate business problems the candidate is expected to solve.

Use short retrieval entities.

Examples:

candidate_matching
search_relevance
ranking_quality
talent_discovery
personalization
marketplace_matching

--------------------------------------------------
COMPANY PREFERENCES
--------------------------------------------------

Generate preferred company categories.

Examples:

product_company
marketplace
ecommerce
adtech
saas
startup

--------------------------------------------------
DOWNWEIGHTED INDUSTRIES
--------------------------------------------------

Generate industries that should be downweighted.

Examples:

it_services
it_consulting

Only include industries explicitly implied by the JD.

--------------------------------------------------
DOWNWEIGHTED COMPANIES
--------------------------------------------------

Generate company names explicitly mentioned or strongly implied as weak-fit backgrounds.

Examples:

tcs
infosys
wipro
cognizant
capgemini
accenture

Use lowercase company names.

--------------------------------------------------
DISQUALIFIERS
--------------------------------------------------

Generate normalized failure signals.

Examples:

research_only
langchain_only
consulting_only
it_services_only
no_production_ml
no_recent_coding
cv_without_ir
speech_without_ir
robotics_without_ir

Return entities.

Not explanations.

--------------------------------------------------
CANDIDATE ARCHETYPES
--------------------------------------------------

Generate archetypes recruiters should actively search.

Examples:

search_engineer
ranking_engineer
retrieval_engineer
recommendation_engineer
relevance_engineer
marketplace_ml_engineer
applied_ml_engineer

Use actual candidate archetypes.

Not personality traits.

BAD

shipper_mindset
systems_thinker
problem_solver

GOOD

ranking_engineer
search_engineer
applied_ml_engineer

--------------------------------------------------
SIGNAL PRIORITIES
--------------------------------------------------

Generate ranking weights.

Use integers only.

Scale:

1 = weak signal

10 = critical signal

Example:

{{
  "production_retrieval": 10,
  "ranking_systems": 10,
  "evaluation_frameworks": 9,
  "product_company_background": 8,
  "python": 8,
  "vector_search": 8,
  "llm_finetuning": 5,
  "open_source": 3,
  "location_match": 2
}}

--------------------------------------------------
OUTPUT RULES
--------------------------------------------------

Return ONLY valid JSON.

Follow the schema exactly.

Do not invent fields.

Do not omit fields.

Do not generate explanations.

Do not generate sentences.

Prefer retrieval entities.

Prefer normalized tokens.

Prefer search-oriented concepts.

Schema:

{schema_json}

JOB DESCRIPTION:

{jd_text}
"""


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    elif text.startswith("```"):
        text = text.replace("```", "")
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(
            "No valid JSON found in Gemini response."
        )

    text = text[start:end + 1]

    return json.loads(text)


def _merge_defaults(
    extracted: Dict[str, Any],
    default: Dict[str, Any]
) -> Dict[str, Any]:
    result = dict(default)

    for key, value in default.items():

        if key not in extracted:
            continue

        if (
            isinstance(value, dict)
            and isinstance(extracted[key], dict)
        ):
            result[key] = _merge_defaults(
                extracted[key],
                value
            )

        else:
            result[key] = extracted[key]

    return result


def _normalize_lists(data):

    list_fields = [
        "required_skills",
        "preferred_skills",
        "required_domains",
        "preferred_domains",
        "domain_keywords",
        "target_titles",
        "adjacent_titles",
        "transferable_titles",
        "title_categories",
        "candidate_archetypes",
        "hidden_gem_profiles",
        "adjacent_backgrounds",
        "transferable_backgrounds",
        "production_requirements",
        "core_business_problems",
        "preferred_company_types",
        "downweighted_industries",
        "downweighted_companies",
        "location_preferences",
        "notice_period_preference_days",
        "disqualifiers",
        "positive_title_patterns",
        "signal_priorities",
    ]

    for field in list_fields:

        value = data.get(field)

        if value is None:
            data[field] = []

        elif isinstance(value, str):
            data[field] = [value]

        elif not isinstance(value, list):
            data[field] = []

    return data


def _deduplicate(data: Dict[str, Any]) -> Dict[str, Any]:

    for key, value in data.items():

        if isinstance(value, list):

            deduped = []
            seen = set()

            for item in value:

                if isinstance(item, dict):
                    deduped.append(item)
                    continue

                item_str = str(item).strip()

                if not item_str:
                    continue

                lowered = item_str.lower()

                if lowered not in seen:
                    seen.add(lowered)
                    deduped.append(item_str)

            data[key] = deduped

    return data


def parse_jd(jd_text: str) -> Dict[str, Any]:

    client = _configure_gemini()

    config = types.GenerateContentConfig(
        temperature=0.1,
        response_mime_type="application/json"
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=_build_prompt(jd_text),
        config=config
    )

    extracted = _extract_json(response.text)

    intent = _merge_defaults(
        extracted,
        DEFAULT_SCHEMA
    )

    intent["domain_keywords"] = list(
        set(
            kw.lower().replace(" ", "_")
            for kw in intent["domain_keywords"]
        )
    )

    intent = _normalize_lists(intent)

    intent = _deduplicate(intent)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            intent,
            f,
            indent=2,
            ensure_ascii=False
        )

    return intent




if __name__ == "__main__":

    sample_jd = """
    Job Description: Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid — flexible cadence) | Open to relocation candidates from Tier-1 Indian cities
Employment Type: Full-time
Experience Required: 5–9 years (see "what we mean by this" below)

Let's be honest about this role
We're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.
If you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.
If you've spent your career bouncing between early-stage startups and you want to "just code" without having to think about product or recruiter workflows or eval frameworks, this also isn't it.
We need someone who is simultaneously comfortable with two things that sound contradictory:
1.	Deep technical depth in modern ML systems — embeddings, retrieval, ranking, LLMs, fine-tuning.
2.	Scrappy product-engineering attitude — willing to ship a working ranker in a week even if the underlying ML is "obviously suboptimal," because we need to learn from real users before we know what to actually optimize for.
These are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into "researcher" vs "shipper" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.

What you'd actually be doing
The high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.
In practical terms, your first 90 days will probably look like:
•	Weeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.
•	Weeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.
•	Weeks 9-12: Set up the evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops — so we can keep improving without flying blind.
Beyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.

What we mean by "5-9 years"
This is a range, not a requirement. Some people hit "senior engineer" judgment at 4 years; some never hit it after 15. We've used 5-9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.
That said, here are the disqualifiers we actually apply:
•	If you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment — we will not move forward. We are explicit about this. We've tried it twice and it didn't work for either side.
•	If your "AI experience" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI — we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking before it became fashionable.
•	If you are a senior engineer who hasn't written production code in the last 18 months because you've moved into "architecture" or "tech lead" roles — we will probably not move forward. This role writes code.

The skills inventory (please read carefully)
Most JDs list 20 skills and you're supposed to have all of them. We're going to do this differently.
Things you absolutely need
•	Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model — we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.
•	Production experience with vector databases or hybrid search infrastructure — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.
•	Strong Python. Yes really, we care about code quality.
•	Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.
Things we'd like you to have but won't reject you for
•	LLM fine-tuning experience (LoRA, QLoRA, PEFT)
•	Experience with learning-to-rank models (XGBoost-based or neural)
•	Prior exposure to HR-tech, recruiting tech, or marketplace products
•	Background in distributed systems or large-scale inference optimization
•	Open-source contributions in the AI/ML space
Things we explicitly do NOT want
This is the section most JDs skip but we think it's the most important:
•	Title-chasers. If your career trajectory shows you optimizing for "Senior" → "Staff" → "Principal" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years.
•	Framework enthusiasts. If your GitHub is full of LangChain tutorials and your blog posts are "How I used [hot framework] to build [demo]" — that's fine but it's not what we need. We need people who think about systems, not frameworks.
•	People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine.
•	People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here.
•	People whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think.

On location, comp, and logistics
•	Location: Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas.
•	Notice period: We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.

The vibe check
We genuinely believe culture-fit matters more at this stage than skills-fit. Skills are teachable; the rest mostly isn't.
We work async-first and write a lot. If you find writing painful, you'll find this role painful.
We disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.
We move fast and break things, with the caveat that "things" are usually our internal assumptions, not user-facing systems. If you need a stable, mature codebase to be productive, you'll find this role unstable.

How to read between the lines
The "ideal candidate" we're imagining is roughly:
•	6-8 years total experience, of which 4-5 are in applied ML/AI roles at product companies (not pure services).
•	Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.
•	Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt) — and can defend them with reference to systems they actually built.
•	Located in or willing to relocate to Noida or Pune.
•	Active on Redrob platform (or has clear signal of being in the job market) so we can actually talk to them.
We are aware this is a narrow profile. We're not expecting to find many matches in a 100K candidate pool. We're explicitly OK with that — we'd rather see 10 great matches than 1000 maybes.

Final note for the participants of the Redrob hackathon
If you're reading this in the context of the Intelligent Candidate Discovery & Ranking Challenge:
The "right answer" to this JD is not "find candidates whose skills section contains the most AI keywords." That's a trap we've explicitly built into the dataset.
The right answer involves reasoning about the gap between what the JD says and what the JD means. A Tier 5 candidate may not use the words "RAG" or "Pinecone" in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit. A candidate who has all the AI keywords listed as skills but whose title is "Marketing Manager" is not a fit, no matter how perfect their skill list looks.
Your ranking system should also weigh behavioral signals — a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately.
Good luck.

    """

    result = parse_jd(sample_jd)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )