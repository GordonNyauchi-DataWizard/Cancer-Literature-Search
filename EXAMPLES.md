# Examples and Use Cases

Practical examples demonstrating different ways to use the Cancer Literature Search System.

## Table of Contents

1. [Basic Search Examples](#basic-search-examples)
2. [Advanced Search Techniques](#advanced-search-techniques)
3. [Question Answering Examples](#question-answering-examples)
4. [Comparative Analysis Examples](#comparative-analysis-examples)
5. [Python API Examples](#python-api-examples)
6. [Research Workflows](#research-workflows)
7. [Tips and Tricks](#tips-and-tricks)

---

## Basic Search Examples

### Example 1: Finding Treatment Information

**Query**: `immunotherapy side effects`

**Expected Results**:
- Papers discussing immune-related adverse events
- Safety profiles of checkpoint inhibitors
- Patient management strategies

**CLI**:
```bash
python cli.py --query "immunotherapy side effects"
```

**Python**:
```python
from semantic_search import CancerSearchApp

app = CancerSearchApp()
app.build_or_load_index()

results = app.search("immunotherapy side effects", top_k=10)
for r in results['results']:
    print(f"{r['pdf_file']} (Page {r['page']}): {r['similarity']:.3f}")
```

### Example 2: Research on Specific Genes

**Query**: `BRCA1 mutations breast cancer`

**Why This Works**:
- Semantic search understands relationships
- Finds relevant passages even without exact keyword matches
- Considers context (BRCA1 in the context of breast cancer)

**Results Might Include**:
- Prevalence studies
- Treatment implications
- Prognostic factors
- Genetic counseling guidelines

### Example 3: Finding Clinical Trial Information

**Query**: `phase 2 clinical trial results melanoma`

**CLI with filters**:
```bash
python cli.py --query "phase 2 clinical trial results melanoma" --top-k 15
```

---

## Advanced Search Techniques

### Technique 1: Using Medical Terminology

**Good Query**: `pembrolizumab efficacy non-small cell lung cancer`

**Poor Query**: `medicine for lung cancer`

**Why**:
- Scientific papers use specific terminology
- Generic terms match too broadly
- Specific drug names, cancer types improve precision

### Technique 2: Combining Concepts

**Query**: `tumor microenvironment immune checkpoint inhibitors`

This finds papers discussing how the tumor environment affects immunotherapy response.

### Technique 3: Mechanism of Action Queries

**Query**: `CAR-T cell therapy mechanism of action`

**Results Focus On**:
- How CAR-T cells are engineered
- Target recognition
- Tumor cell killing
- Persistence and expansion

### Technique 4: Comparative Queries

**Query**: `EGFR TKI osimertinib vs gefitinib`

Finds papers comparing these two EGFR inhibitors.

---

## Question Answering Examples

### Example 1: Understanding Treatments

**Question**: "What is CAR-T cell therapy and how does it work?"

**CLI**:
```bash
python cli.py --ask "What is CAR-T cell therapy and how does it work?"
```

**Expected Answer**:
```
CAR-T cell therapy is a type of immunotherapy that involves:

1. Collection: T cells are extracted from the patient's blood
2. Engineering: T cells are genetically modified to express chimeric 
   antigen receptors (CARs)
3. Expansion: Modified cells are grown in large numbers
4. Infusion: CAR-T cells are infused back into the patient
5. Action: CAR-T cells recognize and kill cancer cells expressing 
   the target antigen

(Source: Smith et al., Nature Reviews, Page 245)

The CAR typically consists of:
- An extracellular antigen-binding domain (often from an antibody)
- A transmembrane domain
- Intracellular signaling domains that activate the T cell

(Source: Johnson et al., Blood Journal, Page 89)
```

### Example 2: Side Effects and Management

**Question**: "What are the most common side effects of checkpoint inhibitors and how are they managed?"

**Python**:
```python
question = "What are the most common side effects of checkpoint inhibitors?"
answer = app.answer_question(question, top_k=10)
print(answer)
```

### Example 3: Diagnostic Questions

**Question**: "What biomarkers predict response to immunotherapy?"

**Why This is Powerful**:
- Retrieves information from multiple papers
- Synthesizes findings
- Provides citations
- More comprehensive than reading one paper

### Example 4: Technical Questions

**Question**: "How does tumor mutational burden affect immunotherapy response?"

**Expected Answer Structure**:
- Definition of TMB
- Mechanism linking TMB to response
- Supporting evidence from studies
- Clinical implications
- Caveats and exceptions

---

## Comparative Analysis Examples

### Example 1: Treatment Comparisons

**Query**: `checkpoint inhibitors vs chemotherapy first-line treatment`

**Use Case**: 
A clinician wants to understand evidence comparing these approaches.

**What the Analysis Provides**:
- Efficacy comparison (response rates, survival)
- Toxicity profiles
- Patient selection criteria
- Cost considerations (if discussed in papers)
- Quality of life data

**CLI**:
```bash
python cli.py --compare "checkpoint inhibitors vs chemotherapy"
```

### Example 2: Different CAR-T Approaches

**Query**: `CD19 CAR-T vs CD22 CAR-T`

**Analysis Focuses On**:
- Target antigen differences
- Efficacy in different malignancies
- Resistance mechanisms
- Combination approaches

### Example 3: Surgical Approaches

**Query**: `mastectomy vs lumpectomy outcomes`

**Comparison Covers**:
- Survival rates
- Recurrence risks
- Cosmetic outcomes
- Psychological impact

---

## Python API Examples

### Example 1: Building a Research Dashboard

```python
from semantic_search import CancerSearchApp
import pandas as pd

app = CancerSearchApp()
app.build_or_load_index()

# Search multiple topics
topics = [
    "immunotherapy",
    "targeted therapy",
    "radiation therapy",
    "chemotherapy"
]

results_data = []
for topic in topics:
    results = app.search(topic, top_k=5, enhance=False)
    for r in results['results']:
        results_data.append({
            'topic': topic,
            'paper': r['pdf_file'],
            'page': r['page'],
            'similarity': r['similarity'],
            'excerpt': r['text'][:100]
        })

df = pd.DataFrame(results_data)
df.to_csv('research_dashboard.csv', index=False)
print(df.groupby('topic')['similarity'].mean())
```

### Example 2: Batch Question Answering

```python
questions = [
    "What is the mechanism of action of pembrolizumab?",
    "What are common side effects of CAR-T therapy?",
    "How is tumor mutational burden measured?",
    "What is the role of PD-L1 expression in treatment selection?"
]

answers = {}
for q in questions:
    print(f"\nProcessing: {q}")
    answer = app.answer_question(q, top_k=5)
    answers[q] = answer

# Save to file
with open('qa_results.txt', 'w') as f:
    for q, a in answers.items():
        f.write(f"Q: {q}\n")
        f.write(f"A: {a}\n")
        f.write("\n" + "="*80 + "\n\n")
```

### Example 3: Custom Result Processing

```python
def extract_drug_mentions(text):
    """Extract drug names from text (simple example)."""
    drugs = ['pembrolizumab', 'nivolumab', 'atezolizumab', 
             'ipilimumab', 'durvalumab']
    found = []
    text_lower = text.lower()
    for drug in drugs:
        if drug in text_lower:
            found.append(drug)
    return found

# Search and analyze
results = app.search("checkpoint inhibitors clinical trials", top_k=20)

drug_counts = {}
for r in results['results']:
    drugs = extract_drug_mentions(r['text'])
    for drug in drugs:
        drug_counts[drug] = drug_counts.get(drug, 0) + 1

print("\nMost mentioned drugs:")
for drug, count in sorted(drug_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{drug}: {count} mentions")
```

### Example 4: Building a Citation Network

```python
import re
from collections import defaultdict

def extract_citations(text):
    """Extract citation patterns from text."""
    # Simple pattern: (Author et al., Year)
    pattern = r'\(([A-Z][a-z]+ et al\., \d{4})\)'
    return re.findall(pattern, text)

citation_network = defaultdict(list)

results = app.search("immunotherapy resistance mechanisms", top_k=50)

for r in results['results']:
    paper = r['pdf_file']
    citations = extract_citations(r['text'])
    citation_network[paper].extend(citations)

# Most cited papers
from collections import Counter
all_citations = [c for cites in citation_network.values() for c in cites]
most_cited = Counter(all_citations).most_common(10)

print("\nMost cited works:")
for citation, count in most_cited:
    print(f"{citation}: {count} times")
```

---

## Research Workflows

### Workflow 1: Literature Review

**Scenario**: You're writing a review on immunotherapy in lung cancer.

**Steps**:

1. **Broad Search**
   ```bash
   python cli.py --query "immunotherapy lung cancer" --top-k 50
   ```

2. **Narrow Down**
   ```bash
   python cli.py --query "PD-1 PD-L1 inhibitors NSCLC"
   python cli.py --query "durvalumab stage III NSCLC"
   python cli.py --query "immunotherapy biomarkers lung cancer"
   ```

3. **Get Summaries**
   ```python
   topics = [
       "checkpoint inhibitors NSCLC efficacy",
       "immunotherapy NSCLC toxicity",
       "predictive biomarkers lung cancer immunotherapy",
       "combination immunotherapy NSCLC"
   ]
   
   for topic in topics:
       results = app.search(topic, top_k=10, enhance=True)
       print(f"\n{topic}:\n{results['summary']}\n")
   ```

4. **Answer Specific Questions**
   ```bash
   python cli.py --ask "What is the role of PD-L1 expression in selecting patients for immunotherapy?"
   python cli.py --ask "What are the outcomes of durvalumab in stage III NSCLC?"
   ```

### Workflow 2: Clinical Decision Support

**Scenario**: Selecting treatment for a patient with melanoma.

**Steps**:

1. **Treatment Options**
   ```bash
   python cli.py --query "melanoma first-line treatment options"
   ```

2. **Evidence for Specific Agents**
   ```bash
   python cli.py --ask "What is the evidence for nivolumab plus ipilimumab in melanoma?"
   ```

3. **Side Effect Profiles**
   ```bash
   python cli.py --compare "nivolumab monotherapy vs nivolumab plus ipilimumab toxicity"
   ```

4. **Special Populations**
   ```bash
   python cli.py --query "melanoma immunotherapy elderly patients"
   ```

### Workflow 3: Hypothesis Generation

**Scenario**: Looking for potential combination therapies.

```python
# Find papers about resistance mechanisms
resistance_results = app.search("immunotherapy resistance mechanisms", top_k=20)

# Identify mentioned pathways
pathways = set()
for r in resistance_results['results']:
    text = r['text'].lower()
    if 'wnt' in text:
        pathways.add('WNT signaling')
    if 'pi3k' in text or 'akt' in text:
        pathways.add('PI3K/AKT pathway')
    if 'mapk' in text:
        pathways.add('MAPK pathway')

# Search for inhibitors of identified pathways
for pathway in pathways:
    print(f"\nSearching for: {pathway} inhibitors")
    results = app.search(f"{pathway} inhibitors cancer", top_k=5)
    # Analyze results...
```

---

## Tips and Tricks

### Tip 1: Iterative Refinement

Start broad, then narrow:
```
1. "cancer immunotherapy" (too broad)
2. "melanoma immunotherapy" (better)
3. "melanoma pembrolizumab resistance" (specific)
```

### Tip 2: Use Synonyms

If initial results aren't good, try synonyms:
- "CAR-T" vs "chimeric antigen receptor"
- "NSCLC" vs "non-small cell lung cancer"
- "checkpoint inhibitor" vs "immune checkpoint blockade"

### Tip 3: Combine with Manual Review

```python
# Get initial results
results = app.search("KRAS mutations targeted therapy", top_k=20)

# Review and flag interesting papers
interesting = []
for r in results['results']:
    print(f"\n{r['pdf_file']} (p.{r['page']})")
    print(f"{r['text'][:200]}...")
    
    keep = input("Keep? (y/n): ")
    if keep.lower() == 'y':
        interesting.append(r['pdf_file'])

# Read full papers
print(f"\nPapers to read in full: {interesting}")
```

### Tip 4: Export Results

```python
import json

results = app.search("your query", top_k=20)

# Export for external analysis
with open('search_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Or as CSV for spreadsheet
import csv

with open('search_results.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pdf_file', 'page', 'similarity', 'text'])
    writer.writeheader()
    writer.writerows(results['results'])
```

### Tip 5: Track Research Progress

```python
import datetime

# Log queries and findings
log_file = 'research_log.txt'

def log_search(query, findings):
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Query: {query}\n")
        f.write(f"Findings: {findings}\n")

query = "EGFR mutations lung cancer"
results = app.search(query, top_k=5)
summary = results.get('summary', 'No summary available')

log_search(query, summary)
```

---

## Use Case: Building a Knowledge Base

Full example of building a searchable knowledge base:

```python
from semantic_search import CancerSearchApp
import json
from datetime import datetime

class CancerKnowledgeBase:
    """Custom knowledge base built on semantic search."""
    
    def __init__(self):
        self.app = CancerSearchApp()
        self.app.build_or_load_index()
        self.cache = {}
    
    def query(self, question, cache=True):
        """Query with caching."""
        if cache and question in self.cache:
            return self.cache[question]
        
        answer = self.app.answer_question(question, top_k=10)
        
        if cache:
            self.cache[question] = {
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            }
        
        return answer
    
    def batch_query(self, questions):
        """Answer multiple questions."""
        return {q: self.query(q) for q in questions}
    
    def export_kb(self, filepath):
        """Export knowledge base to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def import_kb(self, filepath):
        """Import knowledge base from JSON."""
        with open(filepath, 'r') as f:
            self.cache = json.load(f)

# Usage
kb = CancerKnowledgeBase()

# Build knowledge base
questions = [
    "What is immunotherapy?",
    "How does CAR-T therapy work?",
    "What are checkpoint inhibitors?",
    "What is tumor mutational burden?",
    # ... more questions
]

kb.batch_query(questions)
kb.export_kb('cancer_kb.json')

# Later, reload and use
kb2 = CancerKnowledgeBase()
kb2.import_kb('cancer_kb.json')
```

---

## Conclusion

These examples demonstrate the versatility of the system for:
- Research and literature review
- Clinical decision support
- Hypothesis generation
- Knowledge management
- Educational purposes

The key is to:
1. Start with clear questions
2. Refine iteratively
3. Combine search with manual review
4. Build on results systematically

For more examples or custom use cases, see the documentation or reach out to the community!
