# eICU Time Dataset

## Introduction

The eICU Collaborative Research Database represents a significant advancement in critical care research, developed through a collaborative effort between Philips Healthcare and the MIT Laboratory for Computational Physiology. To prepare it for a class project, I have refined this dataset into what we refer to as the eICU Time dataset, which focuses on key parameters such as patient ID, hospital ID, drug administration time, and the names of the drugs administered.

## User Manual

### Data Schema

**Input: Drugs**

File: `eicu_drug_comp565.csv`

| Name              | Data type    |  Description                                                                                        |
|-------------------|--------------|----------------------------------------------------------------------------------------------------|
| patientunitstayid | int          |  Primary key for patient id                                                                         |
| hospitalid        | int          |  Hospital id where the patient is admitted                                                          |
| drugstartoffset   | int          |  Minutes from ICU admission that drug was started. Negative values indicate pre-ICU admission drugs |
| drugname          | varchar(255) |  Name of the selected drug                                                                          |

**Labels: Ventilator**

File: `eicu_ventilator_comp565.csv`

| Name              | Data type    |  Description                                                   |
|-------------------|--------------|----------------------------------------------------------------|
| patientunitstayid | int          | Primary key for patient id                                     |
| ventilatortreatmentoffset   | int  |  Minutes from ICU admission of ventilator treatment |

**Labels: Sepsis**

File: `eicu_sepsis_comp565.csv`

| Name              | Data type    | Description                                    |
|-------------------|--------------|------------------------------------------------|
| patientunitstayid | int          | Primary key for patient id                     |
| sepsisdiagnosisoffset   | int          | Minutes from ICU admission of sepsis diagnosis |

**Labels: Death**

File: `eicu_mortality_comp565.csv`


| Name              | Data type    | Description                                |
|-------------------|--------------|--------------------------------------------|
| patientunitstayid | int          | Primary key for patient id                 |
| deathoffset   | int          | Minutes from ICU admission of patient death |


### Unique Drugs

The eICU Collaborative Research Database is pivotal for understanding intensive care unit patient data. However, there are several significant challenges when training models using the eICU dataset.

- Approximately one-third of drug names remain unrecorded.
- Some drugs share the same identity (e.g. "aspirin" and "acetylsalicylic acid"), and patient data can be merged across all identities of said medications.
- The dosage information in the drug names can be disregarded (e.g. "aspirin 10 mg").

I imputed the missing drugs and harmonised the drugs by referring the following drug reference list.  

```
['acetamin', 'biotene', 'compazine', 'ferrous', 'imdur', 'lidocaine', 'milk of magnesia', 'nystatin', 'prochlorperazine', 'tamsulosin',
'advair diskus', 'bisacodyl', 'coreg', 'flagyl', 'influenza vac', 'lipitor', 'mineral', 'omeprazole', 'promethazine', 'thiamine',
'albumin', 'bumetanide', 'cozaar', 'flomax', 'infuvite', 'lisinopril', 'mineral oil', 'ondansetron', 'propofol', 'ticagrelor',
'albuterol', 'bumex', 'decadron', 'flumazenil', 'insulin', 'lispro', 'mono-sod', 'optiray', 'pulmicort respule', 'tiotropium',
'allopurinol', 'buminate', 'definity', 'fluticasone-salmeterol', 'insulin detemir', 'loratadine', 'morphine', 'oxycodone', 'quetiapine', 'toradol',
'alprazolam', 'calcium carbonate', 'deltasone', 'folic acid', 'iohexol', 'lorazepam', 'motrin', 'pantoprazole', 'refresh p.m. op oint', 'tramadol',
'alteplase', 'calcium chloride', 'dexamethasone', 'furosemide', 'iopamidol', 'losartan', 'mupirocin', 'parenteral nutrition', 'reglan', 'trandate',
'alum hydroxide', 'calcium gluconate', 'dexmedetomidine', 'gabapentin', 'ipratropium', 'maalox', 'nafcillin', 'percocet', 'restoril', 'transde rm-scop',
'ambien', 'cardizem', 'dextrose', 'glargine', 'isosorbide', 'magnesium chloride', 'naloxone', 'phenergan', 'ringers solution', 'trazodone',
'aminocaproic acid', 'carvedilol', 'diazepam', 'glucagen', 'kayciel', 'magnesium hydroxide', 'narcan', 'phenylephrine', 'rocuronium', 'ultram',
'amiodarone', 'catapres', 'digoxin', 'glucagon', 'kayexalate', 'magnesium oxide', 'neostigmine', 'phytonadione', 'roxicodone', 'valium',
'amlodipine', 'cefazolin', 'diltiazem', 'glucose', 'keppra', 'magnesium sulf', 'neostigmine methylsulfate', 'piperacillin', 'sennosides', 'vancomycin',
'anticoagulant', 'cefepime', 'diphenhydramine', 'glycopyrrolate', 'ketorolac', 'magox', 'neurontin', 'plasmalyte', 'seroquel', 'vasopressin',
'apresoline', 'ceftriaxone', 'diprivan', 'guaifenesin', 'klonopin', 'medrol', 'nexterone', 'plavix', 'sertraline', 'ventolin',
'ascorbic acid', 'cephulac', 'docusate', 'haldol', 'labetalol', 'meperidine', 'nicardipine', 'pneumococcal', 'simethicone', 'vitamin',
'aspart', 'cetirizine', 'dopamine', 'haloperidol', 'lactated ringer', 'meropenem', 'nicoderm', 'pnu-immune-23', 'simvastatin', 'warfarin',
'aspirin', 'chlorhexidine', 'ecotrin', 'heparin', 'lactulose', 'merrem', 'nicotine', 'polyethylene glycol', 'sodium bicarbonate', 'xanax',
'atenolol', 'ciprofloxacin', 'enoxaparin', 'humulin', 'lanoxin', 'metformin', 'nitro-bid', 'potassium chloride', 'sodium chloride', 'zestril',
'atorvastatin', 'cisatracurium', 'ephedrine', 'hydralazine', 'lantus', 'methylprednisolone', 'nitroglycerin', 'potassium phosphate', 'sodium phosphate', 'zocor',
'atropine', 'citalopram', 'epinephrine', 'hydrochlorothiazide', 'levaquin', 'metoclopramide', 'nitroprusside', 'pravastatin', 'polystyrene sulfonate', 'zolpidem',
'atrovent', 'clindamycin', 'etomidate', 'hydrocodone', 'levemir', 'metoprolol', 'norco', 'precedex', 'spironolactone', 'zosyn',
'azithromycin', 'clonazepam', 'famotidine', 'hydrocortisone', 'levetiracetam', 'metronidazole', 'norepinephrine', 'prednisone', 'sublimaze',
'bacitracin', 'clonidine', 'fat emulsion', 'hydromorphone', 'levofloxacin', 'midazolam', 'normodyne', 'prilocaine', 'succinylcholine',
'bayer chewable', 'clopidogrel', 'fentanyl', 'ibuprofen', 'levothyroxine', 'midodrine', 'norvasc', 'prinivil', 'tacrolimus']
```

# Reference
```
@article{pollard_eicu_2018,
	author = {Pollard, Tom and Johnson, Alistair E. W. and Raffa, Jesse D. and Celi, Leo Anthony and Mark, Roger G. and Badawi, Omar},
	journal = {Scientific Data},
	title = {{The eICU Collaborative Research Database, a freely available multi-center database for critical care research}},
	year = {2018}
}
```


