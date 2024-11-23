# Project Name

## Description
This is a project to detect diabetic retinopathy using deep learning. This project is done for CSCE 566: Data Mining course at UL Lafayette.


## Data set
The data set is consists of train, validate, and test: three sets. The data folder contains 6,390 NPZ files named in the format "xxxxx_left.npz" or "xxxxx_right.npz", where "xxxxx" (e.g., 691) is a unique numeric ID. 

Harvard-FairVLMed
├── Training
├── Validation
└── Test


Each NPZ file contains the following fields:

```bash
slo_fundus: slo fundus image
race: Asian (0), Black (1), White (2)
male: describing gender: Female (0), Male (1)
hispanic: ethnicity: non-Hispanic (0), Hispanic (1), Unknown (-1)
maritalstatus: maritalstatus: Marriage or Partnered (0), Single (1), Divorced (2), Widoled (3), Legally Separated (4), Unknown (-1)
language: language: English (0), Spanish (1), Other (2), Unknown (-1)
dr_class:
dr_subtype:


glaucoma: Non-Glaucoma (0) or Glaucoma (1)
note: the original de-identified clinical note
note_extra: the original de-identified clinical note with demographic attributes placed at the beginning






