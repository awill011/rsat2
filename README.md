# RSAT Scoring Pipeline

**Automated reading strategy scoring for verbal protocol research · LLT Lab, UC Merced**

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Research](https://img.shields.io/badge/Type-Research%20Tool-green)

---

## What this does

The Reading Strategies Assessment Tool (RSAT) measures reading comprehension by analyzing think-aloud verbal protocols. This pipeline automates that scoring — taking raw participant text responses and computing three key metrics:

- **SW/PW (paraphrasing)** — overlap between content words in the source text and the participant's response
- **Total word count (effort)** — overall verbosity as a proxy for engagement and processing depth
- **SNO/PW (elaboration)** — degree to which the participant extends beyond the source text

---

## Why it matters

Manual RSAT scoring is time-intensive and prone to human error. This tool replaces a largely manual process with a reproducible Python pipeline, enabling the LLT Lab to process larger participant datasets faster and more consistently.

---



**Input:** CSV with participant verbal protocol responses per item  
**Output:** CSV with SW/PW, effort, and SNO/PW scores per participant per item

---

## Tech stack

`Python` `pandas` `NLTK / spaCy` `CSV I/O`

---
