# Writing Style Guide

## Voice and tone

- **First person plural ("we") for methodology and findings.** Standard academic convention. Even though the author is single, "we" refers to the work, not to individuals.
- **Direct, declarative sentences.** Avoid hedging where claims are supported by data. "The model achieves 38.9% Rank-1" not "the model appears to achieve roughly 38.9% Rank-1."
- **Honest qualification when uncertain.** When making interpretive claims rather than reporting numbers, hedging is appropriate. "This may suggest..." or "One interpretation is..."
- **Avoid jargon that adds no information.** "We leverage..." → "We use..." or "The approach uses..."

## Sentence structure

- **Mix short and medium-length sentences.** Avoid very long sentences with multiple clauses.
- **One idea per sentence where possible.** Break up complex points.
- **Use active voice unless passive is clearer.** "We evaluate the model on CASIA-B" not "The model is evaluated on CASIA-B by the authors."

## Specific phrases to avoid

- "It is important to note that..." (just note it)
- "It is worth mentioning..." (just mention it)
- "We employ a novel..." (let novelty be demonstrated, not asserted)
- "State-of-the-art" (use sparingly; "competitive" or "recent" is often better)
- "Significantly outperforms" (unless there is statistical testing supporting the claim)

## Numerical reporting

- Report mean ± std for all multi-seed results
- Round to one decimal place for percentages (e.g., 38.9%, not 38.92%)
- Include sample size (number of seeds) in table captions
- For comparisons, report the absolute and relative difference where meaningful

## Tables

- Bold the best result in each row or column
- Caption explains the setup (e.g., "Rank-1 accuracy on CASIA-B nm→cl, mean ± std across 3 seeds, ssl_casia.pt initialization, R1 region")
- For ablation tables, mark the default configuration explicitly
- For comparison tables, note that protocols may differ between methods

## Figures

- Caption is self-contained — a reader should understand the figure without re-reading the body text
- Label axes with units
- Use a consistent color scheme across figures (e.g., DenseGait-SSL always one color, CASIA-SSL another, random-init a third)
- Include error bars or shaded confidence regions for multi-seed plots

## Citations

- **Verify every citation.** Do not let the writing chat hallucinate citations. Each citation must point to a paper that actually exists and contains the claim attributed to it.
- Use a consistent citation format (likely IEEE or APA, depending on Bishop's requirements)
- For methods being compared to, cite the original paper, not derivative or survey works
- Be specific in citations: "GaitGraph (Teepe et al. 2021)" with the year, not just "GaitGraph"

## Mathematical notation

- Define every symbol when first used
- Use boldface for vectors and matrices, italic for scalars
- Be explicit about dimensions (e.g., x ∈ ℝ^(T×J×2))
- Equations should have explicit numbering when referenced elsewhere

## Writing process

- Outline before writing each section (3-5 bullet points)
- First draft: focus on getting ideas down in order
- Second pass: tighten sentences, remove filler
- Third pass: check flow between paragraphs
- Final pass: proofread for grammar, typos, citation accuracy

## What to avoid in this specific thesis

- **Do not mention the original draft of the thesis.** Forward-facing only.
- **Do not reference previous low numerical results.** The current results are the results.
- **Do not frame architectural choices as "fixing" anything.** Present them as deliberate design.
- **Do not overclaim novelty.** State what is specific to this work and let the reader judge.
- **Do not claim to surpass recent SOTA when the numbers don't support it.** Honest positioning.