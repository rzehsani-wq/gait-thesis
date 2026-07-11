# Thesis Outline (Revised Draft)

## Chapter 1: Introduction (~10-15 pages)

### 1.1 Motivation
- Gait recognition as biometric modality
- Why pose-based instead of silhouette-based
- The clothing-variation problem specifically

### 1.2 Research Question
- Can a factorized spatial-temporal transformer with masked-joint SSL match published pose-only methods under clothing variation?
- What is the relative value of in-domain vs. in-the-wild SSL pretraining for this task?

### 1.3 Contributions
- Three contributions as listed in context document
- Explicit framing as deliberate design choices, not evaluation of existing tools

### 1.4 Thesis structure
- Brief preview of remaining chapters

## Chapter 2: Background and Related Work (~15-20 pages)

### 2.1 Gait recognition: problem and approaches
- Definition, applications, datasets

### 2.2 Silhouette-based methods
- GaitSet, GaitPart, etc.
- Strengths and limitations under appearance variation

### 2.3 Pose-based methods
- PoseGait (hand-crafted features)
- GaitGraph (graph convolutions)
- GaitFormer (multi-task SSL)
- GaitPT, GaitPTv2 (hierarchical attention)

### 2.4 Self-supervised learning for skeleton data
- Masked autoencoder paradigm
- Domain transfer considerations

### 2.5 CASIA-B benchmark
- Protocol, conditions, view angles
- Standard evaluation conventions

### 2.6 Position of this work
- What gap this work addresses
- Design philosophy: learned representations vs. imposed structural priors

## Chapter 3: Methodology (~20-25 pages)

### 3.1 Overview of the approach
- High-level architecture diagram
- SSL pretraining + supervised fine-tuning pipeline

### 3.2 Architecture: factorized spatial-temporal transformer
- 3.2.1 Token representation (T×J joint-frame tokens)
- 3.2.2 Embedding: per-joint coordinate + joint-ID + temporal positional
- 3.2.3 Spatial attention blocks (within-frame, across joints)
- 3.2.4 Temporal attention blocks (within-joint, across frames, with relative bias)
- 3.2.5 Encoder structure (S-T-S-T)
- 3.2.6 Pooling and downstream heads
- 3.2.7 Comparison to related architectures (the five comparisons)

### 3.3 Self-supervised pretraining
- 3.3.1 Masked-joint reconstruction objective
- 3.3.2 Masking strategy and ratio
- 3.3.3 Training schedule
- 3.3.4 Domain considerations (in-domain CASIA vs. in-the-wild DenseGait)

### 3.4 Region partitioning
- 3.4.1 R1 (upper body), R2 (lower body), ALL definitions
- 3.4.2 Justification for region-based analysis
- 3.4.3 Implementation via joint_indices

### 3.5 Data preprocessing
- 3.5.1 Pose extraction (using existing extractors)
- 3.5.2 Joint normalization (mid-hip centering, shoulder-width scaling, std rescaling)
- 3.5.3 Cross-dataset alignment between CASIA-B and DenseGait
- 3.5.4 DenseGait joint ordering remap (briefly: was recovered empirically; explain methodology without dwelling on the dataset documentation gap)

### 3.6 Supervised fine-tuning
- 3.6.1 Identification head and ArcFace metric learning
- 3.6.2 Loss schedule: CE warmup followed by ArcFace
- 3.6.3 Learning rates and optimization details

### 3.7 Evaluation protocol
- 3.7.1 Subject-disjoint splits
- 3.7.2 Gallery and probe construction (gallery: nm; probe: nm/bg/cl)
- 3.7.3 Gallery size choice
- 3.7.4 View matching
- 3.7.5 kNN-based retrieval (centroid)

## Chapter 4: Experiments and Results (~20-25 pages)

### 4.1 Setup
- 4.1.1 Datasets (CASIA-B, DenseGait)
- 4.1.2 Implementation details (compute, frameworks, training duration)
- 4.1.3 Metrics (Rank-1, Rank-5, mean across views and seeds)

### 4.2 Main results
- 4.2.1 Headline table: nm→nm, nm→bg, nm→cl × DenseGait-SSL, CASIA-SSL, random init
- 4.2.2 Discussion of cross-protocol pattern (nm→nm >> nm→bg > nm→cl)
- 4.2.3 SSL contribution (with/without pretraining comparison)

### 4.3 Comparison to state-of-the-art
- 4.3.1 SOTA comparison table
- 4.3.2 Honest positioning: competitive with older pose-only methods, below most recent SOTA
- 4.3.3 Possible reasons (model size, pretraining scale, etc.)

### 4.4 In-domain vs. in-the-wild SSL pretraining
- 4.4.1 Comparison results across protocols
- 4.4.2 Discussion: scale vs. domain alignment tradeoff
- 4.4.3 Caveat: DenseGait pretrained for 20 epochs vs. CASIA's 50; flagged as confound

### 4.5 SSL benefit by condition
- 4.5.1 +5.6 / +11.8 / +13.6 across conditions
- 4.5.2 Interpretation: SSL most valuable when appearance variation increases

### 4.6 Ablation studies
- 4.6.1 Gallery size (g = 1, 2, 4, 8)
- 4.6.2 Region (R1, R2, ALL)
- 4.6.3 View matching (on/off)
- 4.6.4 Backbone freezing (on/off)
- 4.6.5 Metric loss (ArcFace vs. CE)
- 4.6.6 Per-ablation interpretation

### 4.7 Failure analysis
- 4.7.1 Where the model fails (clothing condition, specific views, specific subjects)
- 4.7.2 Confusion matrix analysis
- 4.7.3 Possible causes (pose estimator noise under clothing, limited training data, etc.)

## Chapter 5: Discussion and Conclusion (~10-15 pages)

### 5.1 Conclusions
- Lead with the headline findings
- Architectural contribution
- Cross-domain SSL finding
- Condition-dependent SSL benefit

### 5.2 Limitations and scope
- Single-dataset evaluation (CASIA-B only)
- DenseGait pretraining duration confound
- Below-SOTA absolute performance on nm→cl
- Sensitivity to upstream pose estimator quality

### 5.3 Future work
- Multi-dataset evaluation (OUMVLP, GREW)
- Alternative input representations (bone vectors, joint angles, heatmaps)
- Clothing-invariant techniques (adversarial debiasing, targeted augmentation)
- Architectural ablations (factorized vs. full T×J attention)
- Extended DenseGait pretraining

### 5.4 Closing remarks

## Appendices

- Appendix A: Implementation details, hyperparameters, code references
- Appendix B: Per-seed result breakdowns
- Appendix C: Additional confusion matrices
- Appendix D: Computational requirements