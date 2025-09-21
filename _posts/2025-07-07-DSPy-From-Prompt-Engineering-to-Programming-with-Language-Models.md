---
layout: single
classes: wide
author_profile: true
title: "DSPy: From Prompt Engineering to Programming with Language Models"
seo_title: "DSPy tutorial - declarative LLM programming with signatures, modules, and optimization from Stanford"
published: true
---

TL;DR: As someone who's spent countless hours tweaking prompts and chaining LLM calls, discovering DSPy felt like finding a missing piece of the puzzle. Instead of treating prompts as fragile strings, what if we could treat them as programs? This post walks through my journey building everything from simple QA modules to production-ready code documentation generators, showing how DSPy transforms prompt engineering into systematic programming.

> **Run the code yourself.** The companion notebook [`dspy_colab.ipynb`](https://github.com/venkateshterikuti/notebooks/blob/main/dspy_colab.ipynb) mirrors every section in this post. You can either download it and upload to [Google Colab](https://colab.research.google.com/) via *File -> Upload notebook*, or open it directly in Colab using the GitHub integration. Each notebook cell prints the key outputs I reference below.

---

## What I Built (at a glance)

- **Declarative signatures**: Input/output contracts that replace fragile prompt strings
- **Composable modules**: Reusable LLM components that chain together cleanly  
- **Automatic optimization**: Bootstrap few-shot learning that improves prompts from data
- **Production pipelines**: Research paper analyzer, code documentation generator, multi-hop QA
- **Advanced patterns**: Chain-of-thought reasoning, assertions, parallel processing
- **Real-world integration**: FastAPI endpoints and performance monitoring

![DSPy conceptual flow: signatures → modules → optimization](/assets/images/dspy/chain%20of%20thought.png)

## Why DSPy? The Problem with Traditional Prompt Engineering

Before touching DSPy, my workflow looked exactly like the "fragile prompt engineering" pattern. I'd wrap extraction requests in handcrafted prompts, cross my fingers that the LLM returned valid JSON, and pray the downstream parser didn't explode. Even when it worked, I never knew how brittle the setup was until a tiny change broke everything.

```python
# The old way: fragile and unpredictable
response = call_llm(prompt_template.format(text=sample_text))
parsed = parse_somehow(response)  # Hope for the best!
```

Reading that legacy code now feels like looking at a Rube Goldberg machine: carefully balanced string concatenation, hopeful comments, and brittle parsing. The fundamental issue? We were asking models to behave like deterministic programs without giving them the abstractions they deserved.

---

## Enter DSPy: A New Paradigm

DSPy (Declarative Self-improving Language Programs) reframes that entire experience. Instead of shoving instructions into prompts, we declare what we want with *signatures* and let DSPy handle the messy prompt compilation under the hood.

**Core DSPy Philosophy:**
- **Signatures**: Define input/output contracts instead of writing prompts
- **Modules**: Composable components that chain predictably  
- **Optimization**: Automatically improve prompts from training data
- **Systematic**: Treat LLM applications like traditional software

```python
# The DSPy way: declarative and robust
class BasicQA(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
```

---

## Core Concepts: Signatures and Modules

### **Signatures: Defining What, Not How**

Signatures are contracts you write with your future self: inputs on one side, outputs on the other, and optional descriptions that become part of the generated prompt. The move from "clever prompt" to "explicit signature" might seem small, but it rewires your brain. Instead of thinking about strings, you think about data transformations.

```python
class DocumentQA(dspy.Signature):
    """Answer question based on given context."""
    context = dspy.InputField(desc="relevant background information")
    question = dspy.InputField(desc="question to be answered")
    answer = dspy.OutputField(desc="detailed answer based on context")
```

### **Modules: Composable LLM Programs**

Once you have signatures, modules feel natural. A module wraps `dspy.Predict` and returns whatever the signature promised. What I love here is how module composition feels like building neural network layers—only now the layers reason about language.

```python
class SimpleQAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(BasicQA)

    def forward(self, question):
        prediction = self.generate_answer(question=question)
        return prediction.answer
```

**Testing the Module:**
```python
qa = SimpleQAModule()
result = qa(question="Which country hosts the ITER fusion reactor construction site?")
print(f"Answer: {result}")  # Output: France
```

---

## Building Real Applications: Research Paper Analyzer

Here's where DSPy gets interesting. Instead of one monolithic prompt, I built a research paper analyzer as a pipeline of focused signatures, each handling a specific slice of work.

### **The Pipeline Architecture**

```python
class ExtractMainClaim(dspy.Signature):
    """Extract the main claim or thesis from an academic abstract."""
    abstract = dspy.InputField(desc="academic paper abstract")
    main_claim = dspy.OutputField(desc="the primary claim or contribution in one sentence")

class IdentifyMethods(dspy.Signature):
    """Identify research methods used in the paper."""
    abstract = dspy.InputField()
    methods = dspy.OutputField(desc="list of research methods, separated by semicolons")

class AssessNovelty(dspy.Signature):
    """Assess the novelty of the research contribution."""
    abstract = dspy.InputField()
    main_claim = dspy.InputField()
    novelty_score = dspy.OutputField(desc="integer from 1 to 10")
    novelty_explanation = dspy.OutputField(desc="brief explanation of score")
```

### **Composing the Full Module**

```python
class ResearchPaperAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_claim = dspy.Predict(ExtractMainClaim)
        self.identify_methods = dspy.Predict(IdentifyMethods)
        self.assess_novelty = dspy.Predict(AssessNovelty)
        self.generate_summary = dspy.Predict(GenerateSummary)

    def forward(self, abstract):
        # Extract main claim
        claim = self.extract_claim(abstract=abstract).main_claim
        
        # Identify methods
        methods = self.identify_methods(abstract=abstract).methods
        
        # Assess novelty with context
        novelty_assessment = self.assess_novelty(
            abstract=abstract,
            main_claim=claim
        )
        
        # Generate final summary
        summary = self.generate_summary(
            abstract=abstract,
            main_claim=claim,
            methods=methods,
            novelty_score=novelty_assessment.novelty_score
        ).summary

        return dspy.Prediction(
            main_claim=claim,
            methods=methods,
            novelty_score=novelty_assessment.novelty_score,
            novelty_explanation=novelty_assessment.novelty_explanation,
            summary=summary
        )
```

**Sample Analysis Results:**
```
Main Claim: GridSketch is an orchestration framework for hybrid microgrids that reduces diesel peaker usage by 38% in island grids.

Methods: machine learning; hierarchical control; field trials; dispatch logs analysis

Novelty Score: 7/10

Summary: GridSketch is a novel orchestration framework for hybrid microgrids that effectively reduces diesel peaker usage by 38% in island grids while maintaining reserve margins during cyclone-driven outages.
```

There's something deeply satisfying about seeing `main_claim`, `methods`, and `novelty_explanation` show up as structured attributes instead of plain text. DSPy frees you to think about the flow of information, not just the phrasing of prompts.

---

## The Power of Optimization: Bootstrap Few-Shot Learning

No pipeline stays perfect without iteration. This is where DSPy's optimizers step onto the stage. I create a training set, define a validation metric, and let `BootstrapFewShotWithRandomSearch` tune the prompts for me.

```python
def create_training_examples():
    examples = []
    examples.append(dspy.Example(
        abstract="We introduce GridSketch, an orchestration layer for hybrid microgrids...",
        main_claim="GridSketch coordinates hybrid microgrids using hierarchical graph control.",
        novelty_score="8"
    ).with_inputs('abstract'))
    return examples

def validate_analysis(example, pred, trace=None):
    """Check if the analysis is reasonable"""
    claim_valid = len(pred.main_claim) > 10 and len(pred.main_claim) < 200
    try:
        score = int(pred.novelty_score)
        score_valid = 1 <= score <= 10
    except Exception:
        score_valid = False
    summary_valid = len(pred.summary) > 50 and len(pred.summary) < 500
    return claim_valid and score_valid and summary_valid

# Optimize the analyzer
optimizer = BootstrapFewShotWithRandomSearch(
    metric=validate_analysis,
    max_bootstrapped_demos=2,
    max_labeled_demos=2
)

compiled_analyzer = optimizer.compile(ResearchPaperAnalyzer(), trainset=trainset)
```

The key insight: DSPy turns "prompt tinkering" into a repeatable training loop. You feed it examples, it learns better prompt programs.

---

## Advanced Patterns: Chain of Thought Reasoning

Complex reasoning requires more than single leaps. By swapping `Predict` for `ChainOfThought`, the model exposes its intermediate reasoning directly.

![Chain of thought reasoning flow](/assets/images/dspy/chain%20of%20thought%20prompting.png)

```python
class ChainOfThoughtQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought("question -> reasoning, answer")

    def forward(self, question):
        result = self.generate_answer(question=question)
        return result

# Example with complex calculation
complex_question = """
A climate observatory stores 120 GB of raw satellite data per day. Adding a hyperspectral sensor increases the volume by 35%, and automated filtering removes 12% of the combined data. How many gigabytes are archived each day after these changes?
"""

cot_result = cot_qa(question=complex_question)
print(f"Reasoning: {cot_result.reasoning}")
print(f"Answer: {cot_result.answer}")
```

**Output:**
```
Reasoning: To find the total volume of data archived each day:
1. Calculate the increase: 35% of 120 GB = 42 GB
2. Total after sensor: 120 + 42 = 162 GB
3. Data removed: 12% of 162 GB = 19.44 GB
4. Final archived: 162 - 19.44 = 142.56 GB

Answer: 142.56 GB
```

---

## Building Multi-Hop Question Answering

Multi-hop QA used to demand custom orchestration glue. With DSPy's modular vocabulary, it becomes surprisingly natural:

![Multi-hop reasoning architecture](/assets/images/dspy/multi-hop.png)

```python
class MultiHopQA(dspy.Module):
    def __init__(self, passages_per_hop=3):
        super().__init__()
        self.passages_per_hop = passages_per_hop
        self.generate_query = dspy.ChainOfThought("context, question -> reasoning, query")
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought("context, question -> reasoning, answer")

    def forward(self, question):
        context = []
        # Perform multiple retrieval hops
        for hop in range(2):
            query_result = self.generate_query(
                context="\n".join(context) if context else "No context yet",
                question=question
            )
            passages = self.retrieve(query_result.query).passages
            context.extend(passages[:2])

        # Generate final answer with accumulated context
        final_answer = self.generate_answer(
            context="\n".join(context),
            question=question
        )

        return dspy.Prediction(
            answer=final_answer.answer,
            reasoning=final_answer.reasoning,
            supporting_passages=context
        )
```

---

## Practical Example: Code Documentation Generator

To ground things in an everyday workflow, I built a code documentation assistant that extracts function metadata, generates docstrings, and creates usage examples.

```python
class ExtractFunctionInfo(dspy.Signature):
    """Extract key information from Python function code."""
    code = dspy.InputField(desc="Python function code")
    function_name = dspy.OutputField()
    parameters = dspy.OutputField(desc="list of parameters with types if available")
    return_type = dspy.OutputField(desc="return type if specified, otherwise 'inferred'")

class GenerateDocstring(dspy.Signature):
    """Generate a comprehensive docstring for a Python function."""
    code = dspy.InputField()
    function_name = dspy.InputField()
    parameters = dspy.InputField()
    return_type = dspy.InputField()
    docstring = dspy.OutputField(desc="Google-style docstring without the triple quotes")

class CodeDocumentationGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_info = dspy.Predict(ExtractFunctionInfo)
        self.generate_docstring = dspy.Predict(GenerateDocstring)
        self.generate_example = dspy.Predict(GenerateUsageExample)

    def forward(self, code):
        info = self.extract_info(code=code)
        docstring = self.generate_docstring(
            code=code,
            function_name=info.function_name,
            parameters=info.parameters,
            return_type=info.return_type
        ).docstring
        
        example = self.generate_example(
            function_name=info.function_name,
            parameters=info.parameters,
            docstring=docstring
        ).example_code

        documentation = f'"""\n{docstring}\n\nExample:\n    >>> {example}\n"""'
        
        return dspy.Prediction(
            function_name=info.function_name,
            documentation=documentation,
            example=example
        )
```

This is the moment where DSPy stops feeling like an experiment and starts looking like a productivity machine.

---

## Evaluation and Testing

Great applications need systematic evaluation. DSPy makes this natural with built-in evaluation frameworks:

```python
def documentation_metric(example, pred, trace=None):
    """Evaluate documentation quality"""
    name_correct = pred.function_name == example.function_name
    doc_exists = len(pred.documentation) > 50
    example_exists = len(pred.example) > 10
    return name_correct and doc_exists and example_exists

evaluator = Evaluate(
    devset=create_test_set(),
    num_threads=1,
    display_progress=False
)

score = evaluator(doc_generator, metric=documentation_metric)
print(f"Evaluation Score: {score:.2%}")
```

---

## Fine-Tuning and Distillation

The most advanced DSPy pattern involves teacher-student distillation. First, a chain-of-thought teacher generates rich reasoning traces. Then, a lighter student learns to mimic the teacher's outputs through prompt optimization or actual fine-tuning.

![DSPy fine-tuning workflow](/assets/images/dspy/finetuning.png)

```python
# Teacher generates reasoning traces
teacher_trace = []
for example in training_examples:
    pred = cot_teacher(question=example.question)
    teacher_trace.append({
        "question": example.question,
        "reasoning": pred.reasoning,
        "answer": pred.answer
    })

# Student learns from teacher traces
student_optimizer = BootstrapFewShot(metric=validate_qa)
optimized_student = student_optimizer.compile(SimpleQAModule(), trainset=distillation_examples)
```

**Results:**
- **Baseline student accuracy**: 87.50% (with context)
- **Optimized student accuracy**: 100.00% (after learning from teacher)

This teacher-student pipeline shows how DSPy bridges the gap between powerful but expensive models and efficient deployment-ready systems.

---

## Real-World Integration: Building APIs

All the experimentation is moot if you can't ship. Here's how to lift DSPy modules into production FastAPI endpoints:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
doc_generator_api = CodeDocumentationGenerator()

class CodeRequest(BaseModel):
    code: str

class DocumentationResponse(BaseModel):
    function_name: str
    documentation: str
    example: str

@app.post("/generate-docs", response_model=DocumentationResponse)
async def generate_documentation(request: CodeRequest):
    try:
        result = doc_generator_api(code=request.code)
        return DocumentationResponse(
            function_name=result.function_name,
            documentation=result.documentation,
            example=result.example
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "DSPy Documentation Generator"}
```

---

## Performance Optimization and Monitoring

Production DSPy applications need the same rigor as traditional software:

**Caching and Persistence:**
```python
def save_compiled_module(module, filename):
    """Save optimized DSPy modules to disk"""
    with open(filename, 'wb') as f:
        pickle.dump(module, f)

def load_compiled_module(filename):
    """Load pre-compiled DSPy modules"""
    with open(filename, 'rb') as f:
        return pickle.load(f)
```

**Performance Monitoring:**
```python
class MonitoredModule(dspy.Module):
    def __init__(self, base_module):
        super().__init__()
        self.base_module = base_module
        self.metrics = []

    def forward(self, **kwargs):
        start_time = datetime.now()
        try:
            result = self.base_module(**kwargs)
            success = True
            error = None
        except Exception as exc:
            success = False
            error = str(exc)
            result = None

        elapsed = (datetime.now() - start_time).total_seconds()
        self.metrics.append({
            'timestamp': datetime.now(),
            'success': success,
            'elapsed_time': elapsed,
            'error': error
        })
        
        if not success:
            raise Exception(error)
        return result

    def get_metrics_summary(self):
        if not self.metrics:
            return "No metrics collected"
        success_rate = sum(m['success'] for m in self.metrics) / len(self.metrics)
        avg_time = sum(m['elapsed_time'] for m in self.metrics) / len(self.metrics)
        return f"Success Rate: {success_rate:.2%}, Avg Time: {avg_time:.2f}s"
```

---

## Key Lessons and Best Practices

### **1. Start Simple, Then Optimize**

`dspy.Predict` is my default hammer. Only when I need extra transparency do I reach for chain-of-thought. The biggest wins come from getting the basic pipeline right before adding complexity.

### **2. Invest in Good Training Data**

High-quality examples are silent heroes. A little rigor in creating training examples saves tokens and improves results:

```python
def create_high_quality_example(input_data, expected_output):
    example = dspy.Example(**input_data, **expected_output).with_inputs(*input_data.keys())
    # Validate all fields are non-null
    assert all(v is not None for v in input_data.values())
    assert all(v is not None for v in expected_output.values())
    return example
```

### **3. Use Assertions Wisely**

Assertions aren't just for failing fast; they document intent and enforce output quality:

```python
# Output validation in modules
dspy.Suggest(len(result.answer) > 10, "Answer should be at least 10 characters")
dspy.Assert(result.answer is not None, "Answer cannot be None")
```

### **4. Monitor Everything**

In production, treat monitoring as non-negotiable. DSPy makes instrumentation feel native rather than bolted on.

---

## What's Next: Production RAG and Multi-Agent Systems

![Production RAG architecture](/assets/images/dspy/RAG.png)

My DSPy journey continues with:

1. **Building production RAG systems** with DSPy's retrieval modules
2. **Exploring multi-agent systems** using DSPy modules as agents  
3. **Creating custom teleprompters** for domain-specific optimization
4. **Integrating with vector databases** for scalable retrieval

The beauty of DSPy is that it grows with you. Start simple, and as your understanding deepens, you can build increasingly sophisticated systems.

---

## Conclusion: The Future of LLM Programming

DSPy represents a paradigm shift in how we build LLM applications. Instead of treating prompts as magical incantations, we can now:

1. **Build modular, reusable components** that compose into complex systems
2. **Automatically optimize prompts** based on data rather than intuition  
3. **Test and evaluate systematically** with built-in metrics and frameworks
4. **Scale confidently** knowing our programs are robust and maintainable

The journey from prompt engineering to prompt programming has been eye-opening. DSPy isn't just a library; it's a new way of thinking about LLM applications. As models become more powerful, the ability to program them systematically becomes increasingly important.

---

## Resources and Further Reading

- **DSPy GitHub**: [https://github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- **DSPy Documentation**: [https://dspy-docs.vercel.app/](https://dspy-docs.vercel.app/)  
- **Research Paper**: [DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://arxiv.org/abs/2310.03714)

*Have you tried DSPy? What patterns have you discovered? I'd love to hear about your experiences building with declarative LLM programming!*

---

Thank you for reading! You can find the complete code and notebook for this exploration on GitHub: **[dspy-exploration](https://github.com/venkateshterikuti/notebooks)**

---
