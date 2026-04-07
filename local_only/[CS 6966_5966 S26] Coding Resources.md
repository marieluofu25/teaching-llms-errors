**Coding Resources for Interpretability of LLMs**

*University of Utah | Ana Marasović*

Libraries, notebooks, tutorials, and assignments organized by course topic.

# **Core Libraries & Platforms**

These libraries are used across multiple topics throughout the course. 

## **TransformerLens**

The primary library for mechanistic interpretability of GPT-style language models. Lets you cache, inspect, and modify any internal activation.

* [TransformerLens GitHub](https://github.com/TransformerLensOrg/TransformerLens): Install with pip install transformer\_lens

* [TransformerLens Documentation](https://transformerlensorg.github.io/TransformerLens/): Full API reference and guides

* [Main Demo Notebook](https://transformerlensorg.github.io/TransformerLens/generated/demos/Main_Demo.html): Interactive walkthrough of core features

* [Getting Started in Mech Interp](https://transformerlensorg.github.io/TransformerLens/content/getting_started_mech_interp.html): Key resources

## **SAELens**

The standard library for training and analyzing Sparse Autoencoders (SAEs) on language models. Integrates with TransformerLens and Neuronpedia.

* [SAELens GitHub](https://github.com/jbloomAus/SAELens): pip install sae-lens

* [Tutorial: Loading & Analyzing Pre-Trained SAEs](https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb): Colab notebook

* [Tutorial: Training an SAE from Scratch](https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb): Colab notebook, trains on tiny-stories-1L-21M

* [Tutorial: SAELens \+ Neuronpedia](https://github.com/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb): Feature dashboards, steering, ablation, attribution

## **Neuronpedia**

Interactive web platform for exploring SAE features, circuits, and feature descriptions for publicly released models.

* [Neuronpedia](https://www.neuronpedia.org/): Browse feature dashboards, search features via inference, explore circuit tracer

* [SAEBench Interactive Plots](https://www.neuronpedia.org/sae-bench): Benchmark comparisons for SAEs

## **NNsight**

An architecture-agnostic library for probing and intervening on model internals. Works with any HuggingFace model and supports remote execution on large models (via NDIF).

* [NNsight GitHub & Docs](https://nnsight.net/): pip install nnsight; supports probing, activation patching, causal interventions

## **HuggingFace Transformers**

Required for loading most open-source LLMs. Used as the backbone for SAELens, steering-vectors, EasyEdit, and others.

* [HuggingFace Transformers](https://huggingface.co/docs/transformers/index): pip install transformers

## **ARENA 3.0 Course Materials (Callum McDougall)**

A structured curriculum covering transformers from scratch through mechanistic interpretability. Includes exercises with solutions in Colab notebook format: the most comprehensive hands-on resource for this course.

* [ARENA 3.0 GitHub](https://github.com/callummcdougall/ARENA_3.0): Full exercises with solutions

* [ARENA Website](https://www.arena.education/chapter1): Chapter 1: Transformers & Mech Interp

## **GemmaScope**

* [Gemma Scope](https://huggingface.co/google/gemma-scope), a toolkit designed to help researchers understand the inner workings of Gemma 2; [Colab tutorial](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp?usp=sharing)

* [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2): a comprehensive, open suite of interpretability tools for all Gemma 3 model sizes, from 270M to 27B parameters; [Colab tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)

# **Background: Transformers & LLM Training**

Topics: transformer architecture and pretraining/finetuning/alignment.

## **Libraries**

* [HuggingFace Transformers \+ PEFT](https://huggingface.co/docs/transformers/index): For SFT fine-tuning and loading models; pip install peft for LoRA/QLoRA

* [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/index): pip install trl; implements DPO, PPO, RLHF training loops

## **Exercises**

* [ARENA: Build a Transformer from Scratch](https://arena-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch): Implements GPT-2 architecture step by step in Python/PyTorch 

# **Mechanistic Interpretability Foundations**

Topics: polysemantic neurons, superposition, linear representation hypothesis, residual stream.

## **Exercises**

* [ARENA: \[1.3.1\] Toy Models of Superposition & Sparse Autoencoders](https://arena-chapter1-transformer-interp.streamlit.app/[1.3.1]_Toy_Models_of_Superposition_&_SAEs):  Exercises on superposition and the linear representation hypothesis

# **Sparse Autoencoders (SAEs)**

Topics: SAE training, evaluation metrics, SAE variants, feature descriptions, and automated interpretability.

## **Libraries**

* [SAELens](https://github.com/jbloomAus/SAELens): Primary SAE training & inference library; supports TopK, JumpReLU, BatchTopK variants

* [SAE-Vis](https://github.com/callummcdougall/sae_vis): pip install sae-vis; generates Neuronpedia-style feature dashboards locally

* [SAEBench](https://github.com/EleutherAI/lm-evaluation-harness): Comprehensive SAE evaluation suite; integrated with SAELens

* [Sparsify (EleutherAI)](https://github.com/EleutherAI/sparsify): Lean TopK SAE training library; designed for HuggingFace models

## **Notebook Tutorials**

* [SAELens: Train an SAE on TinyStories-1L](https://github.com/jbloomAus/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb): \~2 hrs on consumer GPU; good first try

* [SAELens: Load & Analyze Pre-Trained SAEs (GPT-2 Small)](https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb): Covers L0, CE loss score, feature density statistics

* [SAELens \+ Neuronpedia Tutorial](https://github.com/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb): Feature search, steering with features, attribution

* [Logit Lens with SAE Features](https://github.com/jbloomAus/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb) 

* [Gemma Scope](https://huggingface.co/google/gemma-scope), a toolkit designed to help researchers understand the inner workings of Gemma 2; [Colab tutorial](https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp?usp=sharing)

* [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2): a comprehensive, open suite of interpretability tools for all Gemma 3 model sizes, from 270M to 27B parameters; [Colab tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r)

## **Excercises**

* [ARENA: \[1.3.2\] Interpretability with SAEs](https://arena-chapter1-transformer-interp.streamlit.app/[1.3.2]_Interpretability_with_SAEs): Exercises on training SAEs, evaluating sparsity-reconstruction tradeoff

* [Tel Aviv University Interpretability Course Materials](https://github.com/mega002/llm-interp-tau) Week 7 and 8 Practicums

## **Feature Descriptions (Automated Interpretability)**

* [ElutherAI/delphi:](https://github.com/EleutherAI/delphi) Library for implementing the feature description pipeline 

* [Neuronpedia Feature Search (No Code Required)](https://www.neuronpedia.org/): Search by inference → enter a prompt → see top activating features with descriptions

# **Feature-Circuit Tracing**

Topics: transcoders, cross-layer transcoders (CLTs), attribution graphs, pruning, sparse feature circuits.

## **Platforms** 

* [Neuronpedia Circuit Tracer](https://www.neuronpedia.org/): Navigate to “Circuit Tracer”: interactive attribution graphs for Claude-3 and Gemma; no code required for exploration

## **Libraries**

* [Circuit-tracer](https://github.com/decoderesearch/circuit-tracer) library that implements tools for finding circuits using features from (cross-layer) MLP transcoders, as originally introduced by Anthropic that we covered in detail in class (Jan 27 to Feb 5); [Tutorial notebook](https://github.com/decoderesearch/circuit-tracer/blob/main/demos/circuit_tracing_tutorial.ipynb) \+ [paper](https://aclanthology.org/2025.blackboxnlp-1.14/)

* [crosscode](https://github.com/oclivegriffin/crosscode): A library for training crosscoders, and by extension, (cross-layer, skip) transcoders, SAEs, and other sparse coding models.

* [CircuitLab](https://github.com/circuits-research/CircuitLab): A library for training cross-layer transcoders at scale.  

## **Tutorials**

* [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2): a comprehensive, open suite of interpretability tools for all Gemma 3 model sizes, from 270M to 27B parameters; [Colab tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r) includes transcoders

## **Other Code**

* [Transcoders GitHub (Dunefsky et al., 2024\)](https://github.com/jacobdunefsky/transcoder_circuits/): Original transcoder paper’s code

* [Sparse Feature Circuits Github (Marks et al., 2025\)](https://github.com/saprmarks/feature-circuits): Code for the paper that Anthropic’s circuit tracing heavily built on; uses TransformerLens

* [ACDC: Automated Circuit Discovery](https://github.com/ArthurConmy/Automatic-Circuit-Discovery): These are not **feature** circuits, nodes are full components, but might be helpful

---

# **Probing**

Topics: Probes test whether a specific property is linearly decodable from a frozen model's representations

## **Libraries**

* [Probity](https://github.com/curt-tigges/probity): Dedicated probing library built on TransformerLens; manages datasets, activation collection, and logistic probe training

* [scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression): pip install scikit-learn; LogisticRegression and LinearSVC are standard probe classifiers

## **Exercises**

* [ARENA: OthelloGPT Linear Probes](https://arena-chapter1-transformer-interp.streamlit.app/[1.5.3]_OthelloGPT): Trains probes to identify board state representations; uses logit attribution and activation patching alongside probing

* [Tel Aviv University Interpretability Course Materials](https://github.com/mega002/llm-interp-tau) Week 2 Practicum

## **Tutorials**

* [Carpentries Tutorial: Linear Probes on BERT](https://carpentries-incubator.github.io/fair-explainable-ml/5c-probes.html): Step-by-step notebook; good template for implementing your own probe

* [The Geometry of Truth: nnsight](https://nnsight.net/notebooks/mini-papers/marks_geometry_of_truth/) 


## **Model Diffing**

Topics: crosscoder

## **Libraries**

* [crosscode](https://github.com/oclivegriffin/crosscode): A library for training crosscoders, and by extension, (cross-layer, skip) transcoders, SAEs, and other sparse coding models.

## **Other Code**

* [https://github.com/science-of-finetuning/crosscoder\_learning](https://github.com/science-of-finetuning/crosscoder_learning) training crosscoders

* Github repo of [Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](https://arxiv.org/abs/2504.02922): [https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders](https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders) 

* Github repo of [BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](https://aclanthology.org/2025.acl-long.923/): [https://github.com/lindiatjuatja/BehaviorBox](https://github.com/lindiatjuatja/BehaviorBox) 

## **Tutorials**

* [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2): a comprehensive, open suite of interpretability tools for all Gemma 3 model sizes, from 270M to 27B parameters; [Colab tutorial](https://colab.research.google.com/drive/1NhWjg7n0nhfW--CjtsOdw5A5J_-Bzn4r) includes **crosscoders**

# **Logit Analysis**

Topics: Direct Logit Attribution (DLA) and Logit Lens

## **Tutorials**  

* [TransformerLens: Exploratory Analysis Demo](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb): Demonstrates DLA and logit lens on a concrete task; shows how to generate logit difference bar charts

* [Logan Thomson: Direct Logit Attribution Demo](https://loganthomson.com/Direct-Logit-Attribution/): Includes a notebook for the blog post

* [Logit Lens (nostalgebraist, 2020\)](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens): Original logit lens post; comes with a Colab notebook

* [Logit Lens: nnsight](https://nnsight.net/notebooks/tutorials/probing/logit_lens/): Logit lens using the nnsight library 

## **Other Code**

* [https://github.com/AI4Bharat/Romanlens](https://github.com/AI4Bharat/Romanlens) codebase for [RomanLens: The Role Of Latent Romanization In Multilinguality In LLMs](https://aclanthology.org/2025.findings-acl.1354/)

## **Exercises**

* [ARENA: Logit Lens:](https://arena-chapter1-transformer-interp.streamlit.app/[1.4.1]_Indirect_Object_Identification#1-4-1-indirect-object-identification) Exercise for DLA 

# 

# **Activation Patching**

# Topics: causal interventions on model internals, denoising vs. noising patching, path patching

# **Tutorials**

* # [**TransformerLens: Activation Patching Demo**](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb)**:** Colab notebook; sweeps patches across residual stream, attention heads, and MLPs on IOI; generates logit difference heatmaps

* # [**nnsight: Activation Patching Demo**](https://nnsight.net/tutorials/tutorials/causal_mediation_analysis/activation_patching/) 

# **Exercises**

* # [ARENA: \[1.4.1\] Indirect Object Identification.Activation Patching](https://arena-chapter1-transformer-interp.streamlit.app/[1.4.1]_Indirect_Object_Identification#1-4-1-indirect-object-identification)  

* # [ARENA: OthelloGPT](https://arena-chapter1-transformer-interp.streamlit.app/%255B1.5.3%255D_OthelloGPT): 2\. Looking for modular circuits

---

# **Attribution Patching**

# **Tutorials**

* # [Attribution Patching Notebook](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Attribution_Patching_Demo.ipynb): Notebook that came with the blog post by Neel Nanda

* [nnsight: Attribution Patching Demo](https://nnsight.net/tutorials/tutorials/causal_mediation_analysis/attribution_patching/)

# **Induction Heads & Function Vectors**

# **Tutorials**

* [Transformer Lens Main Demo Notebook | Induction Heads](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Main_Demo.ipynb)

## **Notebooks & Exercises**

* [ARENA: Function Vectors & Model Steering](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter1_transformer_interp/exercises/part32_function_vectors_and_model_steering): Practical exercises on extracting and applying function vectors

* [https://github.com/SaraMolas/induction-heads-exploration/tree/main](https://github.com/SaraMolas/induction-heads-exploration/tree/main) 

# **Steering**

Topics: Steering modifies model behavior at inference time by adding learned direction vectors to activations

## **Libraries**

* [steering-vectors (PyPI)](https://github.com/steering-vectors/steering-vectors): pip install steering-vectors; trains and applies contrastive activation addition vectors on HuggingFace models

* [activation-steering (IBM / CAST)](https://github.com/IBM/activation-steering): ICLR 2025; condition-aware steering with fine-grained control; pip install activation-steering

* [SAELens Steering with Features](https://github.com/jbloomAus/SAELens/blob/main/tutorials/tutorial_2_0.ipynb): Steer via SAE feature amplification (related to Anthropic’s Golden Gate Bridge demo)

## **Notebooks & Exercises**

* [ARENA: Function Vectors & Model Steering](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter1_transformer_interp/exercises/part32_function_vectors_and_model_steering): Practical exercises on extracting and applying function vectors

* [Implementing Activation Steering (LessWrong)](https://www.lesswrong.com/posts/ndyngghzFY388Dnew/implementing-activation-steering): Blog walkthrough with minimal code; good conceptual introduction

# **Data Attribution**

Data attribution methods identify which training examples are responsible for a model’s predictions or parameters.

## **Libraries**

* [DATTRI](https://github.com/TRAIS-Lab/dattri): pip install dattri; unified library for influence functions, TracIn, TRAK, and other data attribution methods for PyTorch models

* [Kronfluence](https://github.com/pomonam/kronfluence): Scalable Kronecker-factored influence function computation for large models

