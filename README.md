# Inclusion-of-Thought

## Stage 1: Initial Preference Elicitation
The model is first prompted using a standard CoT approach to generate its most probable answer, which we denote as the top-choice candidate. This initial output reflects the model's raw preference over the full set.

<img src="./img/IoT-1.png" alt="IoT-1" width="400" height="200">


## Stage 2: Second Plausibility Assessment
We remove the initial (stage 1) selection option set and replace it with a neutral placeholder, "none of the options", yielding the modified option set. The model is then queried again on this modified MCQ, resulting in the second (stage 2) selection.

<img src="./img/IoT-2.png" alt="IoT-2" width="400" height="200">

## Stage 3: Confined Final Inference
The framework then constructs a reduced MCQ consisting solely of the two most plausible model-selected candidates. This reframed question sharply reduces the model's cognitive load (i.e. model's preference instability in the presence of distractor options) and focuses the reasoning process on its own shortlisted alternatives. 

<img src="./img/IoT-3.png" alt="IoT-3" width="400" height="200">




### IoT Performance Across Benchmarks compared to other baselines

| Model | Method | OBQA | CSQA | SIQA | ARC | MMLU | GSM8K-MC | AQUA | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Olmo-2-7B** | **CoT** | <u>80.80</u> | 74.53 | 70.42 | <u>84.21</u> | <u>65.62</u> | 89.08 | 62.99 | <u>75.38</u> |
| | **SC** | 75.80 | <u>76.42</u> | **71.36** | 79.22 | 64.91 | <u>89.39</u> | **66.54** | 74.81 |
| | **EoT** | 74.60 | 73.79 | 69.91 | 79.69 | 62.62 | 87.06 | 55.82 | 71.93 |
| | **IoT** | **84.20** | **76.58** | <u>70.88</u> | **87.54** | **66.78** | **91.66** | <u>63.78</u> | **77.35** |
