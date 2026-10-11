| Layer | Department accuracy | Urgency ±1 | Safety recall | False red alerts | Time / complaint |
|---|---|---|---|---|---|
| Naive Bayes + rules (offline pipeline) | 92% | 96% | 92% | 3 | 0.001 s |
| Keyword rules only | 68% | 91% | 92% | 1 | 0.000 s |

Department accuracy by language:

| Layer | English | Gujarati | Gujlish | Hindi | Hinglish |
|---|---|---|---|---|---|
| Naive Bayes + rules (offline pipeline) | 86% | 96% | 95% | 100% | 89% |
| Keyword rules only | 64% | 65% | 63% | 83% | 72% |
