"""
Initialize the vector database with sample legal cases.
Run this once before using the search app.

Usage:
    python scripts/init_db.py [--model local|openai] [--store chroma|pinecone]
"""

import argparse
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.embeddings import EmbeddingModel
from utils.vector_store import get_vector_store

SAMPLE_CASES = [
    {
        "id": "case_001",
        "title": "Smith v. Jones Construction Ltd",
        "citation": "[2023] UKSC 45",
        "court": "Supreme Court",
        "date": "2023-06-15",
        "year": 2023,
        "category": "Contract Law",
        "summary": "Landmark case establishing the principle that implied terms in construction contracts must be necessary for business efficacy.",
        "text": """LORD REED (with whom Lord Hodge, Lady Arden, Lord Kitchin and Lord Burrows agree):

1. This appeal concerns the circumstances in which a party to a contract may terminate the contract for breach by the other party. The principal issue is whether the contractor's failure to complete the construction works by the contractual completion date constituted a fundamental breach entitling the employer to terminate.

2. The respondent employer engaged the appellant contractor to build a commercial warehouse. The contract specified a completion date of 30 June 2021. The contractor failed to complete by that date, and the works remained incomplete six months later.

3. The employer terminated the contract and engaged alternative contractors. The contractor claimed wrongful termination, arguing that time was not of the essence.

4. We hold that in a construction contract of this nature, the completion date is a condition of the contract. Failure to complete by that date, particularly where the delay is substantial and ongoing, constitutes a repudiatory breach entitling the innocent party to terminate.""",
    },
    {
        "id": "case_002",
        "title": "Ahmed v. British Airways Plc",
        "citation": "[2022] EWCA Civ 1234",
        "court": "Court of Appeal",
        "date": "2022-11-20",
        "year": 2022,
        "category": "Employment Law",
        "summary": "Employment tribunal's decision on wrongful dismissal upheld. Employers must follow proper procedures even in cases of gross misconduct.",
        "text": """LADY JUSTICE SIMLER:

1. This is an appeal against the decision of the Employment Appeal Tribunal which upheld the Employment Tribunal's finding of unfair dismissal.

2. The claimant was employed by British Airways as a senior cabin crew member for 15 years. He was dismissed following allegations of misconduct during a layover.

3. The employer failed to conduct a proper investigation before proceeding to a disciplinary hearing. Key witnesses were not interviewed, and the claimant was not given adequate time to prepare his defence.

4. We affirm the principle that procedural fairness is not a mere formality. Even where the alleged misconduct is serious, an employer must follow a fair process. The failure to do so in this case rendered the dismissal both wrongful and unfair.""",
    },
    {
        "id": "case_003",
        "title": "Global Tech Solutions v. DataCorp International",
        "citation": "[2024] EWHC 567 (Comm)",
        "court": "High Court",
        "date": "2024-03-08",
        "year": 2024,
        "category": "Corporate Law",
        "summary": "Software licensing dispute. Use of software beyond license scope constitutes breach of contract and copyright infringement.",
        "text": """MR JUSTICE MILES:

1. This is a claim for breach of contract and copyright infringement arising from the defendant's use of the claimant's proprietary software.

2. The defendant licensed the claimant's enterprise resource planning software for use by up to 50 employees. An audit revealed that 347 employees had active user accounts.

3. The defendant argued that the license terms were ambiguous and that 'employees' should be interpreted to mean 'concurrent users'. I reject this argument. The license agreement is clear in its terms.

4. I find that the defendant breached the license agreement and infringed the claimant's copyright. Damages are assessed at the cost of the additional licenses that should have been purchased, plus a reasonable uplift for the flagrant nature of the infringement.""",
    },
    {
        "id": "case_004",
        "title": "R v. Thompson",
        "citation": "[2023] EWCA Crim 456",
        "court": "Court of Appeal",
        "date": "2023-04-10",
        "year": 2023,
        "category": "Criminal Law",
        "summary": "Appeal against murder conviction on grounds of self-defence. Court clarified the test for reasonable force in domestic settings.",
        "text": """LORD JUSTICE HOLROYDE:

1. The appellant appeals against his conviction for murder. He contends that the trial judge misdirected the jury on the law of self-defence.

2. The appellant killed his neighbour following a prolonged dispute over a boundary fence. The deceased had entered the appellant's property uninvited and was behaving aggressively.

3. The question for the jury was whether the force used was reasonable in the circumstances as the appellant believed them to be. The judge directed the jury that they must consider whether the appellant genuinely believed he was in danger.

4. We find no misdirection. The judge correctly applied the two-stage test: first, did the defendant genuinely believe force was necessary; second, was the degree of force reasonable. The conviction is safe.""",
    },
    {
        "id": "case_005",
        "title": "Patel v. Patel",
        "citation": "[2024] EWFC 89",
        "court": "Family Court",
        "date": "2024-01-15",
        "year": 2024,
        "category": "Family Law",
        "summary": "Divorce financial settlement involving business assets. Court applied the sharing principle to matrimonial property including company shares.",
        "text": """MRS JUSTICE ROBERTS:

1. This is a financial remedy application following the dissolution of a 20-year marriage. The principal asset is the husband's shareholding in a family business.

2. The husband founded the company during the marriage. It has grown substantially and is now valued at approximately £4.2 million. The wife made no direct financial contribution to the business but was the primary carer for the parties' three children.

3. Applying the principles established in White v White and Miller v Miller, I find that the sharing principle applies to the business. The wife's domestic contributions are to be valued equally with the husband's financial contributions.

4. I order that the wife receive 45% of the total matrimonial assets, which includes a transfer of shares in the company and a lump sum payment.""",
    },
    {
        "id": "case_006",
        "title": "Green Energy Ltd v. National Grid Plc",
        "citation": "[2023] EWHC 890 (TCC)",
        "court": "High Court",
        "date": "2023-07-22",
        "year": 2023,
        "category": "Energy Law",
        "summary": "Dispute over grid connection agreement for renewable energy project. Court upheld force majeure clause interpretation.",
        "text": """MR JUSTICE FRASER:

1. This claim concerns a dispute arising from a grid connection agreement between a renewable energy developer and the grid operator.

2. The claimant contracted with the defendant for a grid connection to serve a 50MW solar farm. The connection was delayed by 18 months due to supply chain disruptions caused by the global pandemic.

3. The defendant invoked the force majeure clause in the connection agreement. The claimant argued that the supply chain issues were foreseeable and that the defendant should have taken reasonable steps to mitigate the delay.

4. I find that the force majeure clause was properly invoked. The supply chain disruptions were of an unprecedented nature and scale. However, the defendant's obligation to use reasonable endeavours to overcome the force majeure event required it to explore alternative suppliers, which it failed to do for the first six months.""",
    },
    {
        "id": "case_007",
        "title": "Khan Family Trust v. Revenue Commissioners",
        "citation": "[2023] UKFTT 789 (TC)",
        "court": "First-tier Tribunal",
        "date": "2023-09-12",
        "year": 2023,
        "category": "Tax Law",
        "summary": "Tax treatment of trust distributions. Distributions to beneficiaries correctly assessed as income rather than capital.",
        "text": """JUDGE THOMAS SCOTT:

1. This appeal concerns the tax treatment of distributions made by the Khan Family Trust to its beneficiaries during the tax years 2019-20 to 2021-22.

2. The trust was established in 2005 with a portfolio of commercial rental properties. Annual rental income was approximately £350,000. The trustees distributed this income to beneficiaries as 'capital distributions'.

3. HMRC assessed the distributions as income, resulting in additional tax liabilities of approximately £180,000 across the three tax years.

4. I find that the distributions were correctly characterised as income. The trust deed is clear that rental income retains its character as income when distributed. The trustees' description of the payments as 'capital' does not change their true nature for tax purposes.""",
    },
    {
        "id": "case_008",
        "title": "Manchester City Council v. Heritage Properties Ltd",
        "citation": "[2024] EWHC 234 (Admin)",
        "court": "High Court",
        "date": "2024-02-14",
        "year": 2024,
        "category": "Property Law",
        "summary": "Listed building enforcement. Council's enforcement notice upheld requiring restoration of unauthorised alterations to Grade II listed building.",
        "text": """MR JUSTICE FORDHAM:

1. This is an appeal against an enforcement notice issued by Manchester City Council requiring the respondent to reverse unauthorised alterations to a Grade II listed building.

2. The respondent purchased the property and carried out extensive internal renovations without obtaining listed building consent. The works included removal of original Victorian fireplaces, replacement of sash windows with uPVC frames, and installation of a modern staircase.

3. The respondent argues that the works were necessary for the building's continued use and that requiring their reversal would be disproportionate.

4. I dismiss the appeal. The protection of listed buildings is a matter of significant public interest. The alterations were carried out in knowing disregard of the consent requirement. The enforcement notice is proportionate and necessary to preserve the building's special architectural and historic interest.""",
    },
    {
        "id": "case_009",
        "title": "Digital Rights Foundation v. Secretary of State",
        "citation": "[2024] EWHC 456 (Admin)",
        "court": "High Court",
        "date": "2024-04-01",
        "year": 2024,
        "category": "Constitutional Law",
        "summary": "Judicial review of government surveillance powers. Court found certain provisions of the Online Safety Act incompatible with Article 8 ECHR.",
        "text": """LADY JUSTICE ANDREWS:

1. This is a claim for judicial review challenging certain provisions of the Online Safety Act 2023 relating to government access to encrypted communications.

2. The claimant, a digital rights charity, argues that the provisions requiring technology companies to provide law enforcement with access to encrypted messages are incompatible with the right to respect for private life under Article 8 of the European Convention on Human Rights.

3. The Secretary of State argues that the provisions are necessary for national security and the prevention of serious crime, and are proportionate to those legitimate aims.

4. I find that while the aims pursued are legitimate, the provisions as drafted lack sufficient safeguards. The absence of prior judicial authorisation for access to encrypted communications, combined with the breadth of the power, renders the provisions disproportionate. I make a declaration of incompatibility under section 4 of the Human Rights Act 1998.""",
    },
    {
        "id": "case_010",
        "title": "Malik v. NHS Foundation Trust",
        "citation": "[2023] EWHC 678 (QB)",
        "court": "High Court",
        "date": "2023-08-30",
        "year": 2023,
        "category": "Medical Law",
        "summary": "Clinical negligence claim. Hospital found liable for delayed diagnosis of cancer, resulting in reduced survival prospects.",
        "text": """MR JUSTICE JAY:

1. This is a claim in clinical negligence. The claimant alleges that the defendant NHS Trust failed to diagnose his lung cancer in a timely manner, reducing his prospects of survival.

2. The claimant presented to his GP with a persistent cough in March 2020. He was referred for a chest X-ray which showed an abnormality. The defendant's radiologist reported the X-ray as normal. The cancer was not diagnosed until November 2021, by which time it had progressed from Stage II to Stage IV.

3. The defendant admits that the X-ray was negligently reported but disputes causation, arguing that even with timely diagnosis, the outcome would likely have been the same.

4. Applying the principles from Gregg v Scott, I find that the defendant's negligence caused the claimant to lose a 35% chance of achieving five-year survival. This loss of chance is actionable. Damages are assessed accordingly.""",
    },
]


def init_database(model_name: str = "local", store_type: str = "chroma"):
    """Initialize the vector database with sample cases."""
    print("🔧 Initializing vector database...")
    print(f"   Embedding model: {model_name}")
    print(f"   Vector store: {store_type}")
    print()

    # Initialize components
    print("📦 Loading embedding model...")
    embeddings = EmbeddingModel(model_name)
    print(f"   Model: {embeddings.model} ({embeddings.dimensions}d)")

    print("💾 Connecting to vector store...")
    store = get_vector_store(store_type)

    # Generate embeddings
    print(f"\n🔢 Generating embeddings for {len(SAMPLE_CASES)} sample cases...")
    texts = [case["text"] for case in SAMPLE_CASES]
    vectors = embeddings.embed_documents(texts)
    print(f"   Generated {len(vectors)} embeddings")

    # Store in vector DB
    print("\n📥 Storing in vector database...")
    ids = [case["id"] for case in SAMPLE_CASES]
    store.add_documents(documents=SAMPLE_CASES, embeddings=vectors, ids=ids)
    print(f"   Stored {len(SAMPLE_CASES)} cases")

    # Verify
    print("\n✅ Database initialized successfully!")
    print(f"   Total cases: {len(SAMPLE_CASES)}")
    print(f"   Vector dimensions: {embeddings.dimensions}")
    print(f"   Store type: {store_type}")

    # Test query
    print("\n🔍 Running test query: 'breach of contract'...")
    query_vec = embeddings.embed_query("breach of contract")
    results = store.search(query_vec, top_k=3)
    print(f"   Top {len(results)} results:")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(
            f"   {i}. {meta.get('title', 'Unknown')} (score: {r.get('score', 0):.3f})"
        )

    print("\n🎉 Done! Run `streamlit run app.py` to start searching.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize the RAG Legal Search database"
    )
    parser.add_argument(
        "--model",
        default="local",
        choices=["local", "openai", "openai-large"],
        help="Embedding model to use (default: local)",
    )
    parser.add_argument(
        "--store",
        default="chroma",
        choices=["chroma", "pinecone"],
        help="Vector store type (default: chroma)",
    )

    args = parser.parse_args()
    init_database(args.model, args.store)
