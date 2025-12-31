Transform the provided research text into a podcast episode script that makes complex ideas feel simple and memorable.

Must follow:
- 6–10 minutes speaking time total
- Structure: Hook (1), 3–6 body segments, Recap (1), Takeaway (1)
- Each segment: title, speaker name, 80–220 words of natural speech
- Use concrete analogies, daily-life examples, and “so what for you” explanations
- Pick 1–2 memorable analogies and reuse them in the recap and takeaway so listeners remember.
- When helpful, include concise formulas or numeric anchors (fractions/percentages/rates) to clarify mechanisms; explain them in plain language.
- If using math/physics, you may use more advanced but still interpretable formulas (e.g., exponential growth `N(t)=N0·e^{rt}`, Bayesian update `P(H|D)=P(D|H)P(H)/P(D)`, error bars with σ/μ, probability bounds with Chernoff/Hoeffding, simple derivatives/integrals to show trends). Always paraphrase what the formula means for a layperson.
- For math-heavy topics, favor: Bayes’ rule, KL divergence intuition, law of large numbers/Chernoff bounds, exponential growth/decay, derivative = rate of change, integral = area/accumulation. Always cite pages/snippets for formulas.
- Every factual claim needs a citation with page numbers/snippets from the provided context only; if unsure, say so explicitly.

Return JSON ONLY (no markdown):
{"segments":[{"title":"...","speaker":"...","text":"...","citations":[{"source":"...","page":1,"snippet":"..."}]}]}
