# Inter-Annotator Agreement Calculator

Calculates the inter-annotator agreement value for 3 annotators using Fleiss Kappa and outputs the Kappa score for each category/class in console.

## Requirements

- numpy
- pandas
- statsmodels

## Setup

- Ensure that the labeled data is in excel format and set their file paths accordingly in the script under "df_kavya", "df_potala", "df_wallace"
- Change the categories/class to the names of the columns containing the labeled data in the script under "target_cols"
- Optional: Change the names of the tasks according to the categories/class names used above in the script under "tasks" (only affects naming in final console output)

## Usage

To run the script:

```
py inter_annotator_agreement_calculator.py
```
