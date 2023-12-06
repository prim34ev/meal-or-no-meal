### Meal or no Meal - Machine Model Training

---

#### Purpose
Train and test a machine model to use patient insulin level data to assess whether or not they've eaten a meal

---

#### Technology Used
- Python 3.9
- scikit-learn==0.21.2
- pandas==0.25.1
- Python pickle

---

#### Project Description

##### Meal data is extracted as follows:
- From the InsulinData.csv file search the column Y for a non NAN non zero value. This time indicates the start of meal consumption time tm. Meal data comprises a 2hr 30 min stretch of CGM data that starts from tm-30min and extends to tm+2hrs.
- No meal data comprises 2 hrs of raw data that does not have meal intake.

---

##### Extraction: Meal data
- Start of a meal obtained from InsulinData.csv. Searches column Y for a non NAN non zero value. This time indicates the start of a meal. There are three conditions:
  - If there is no meal from time tm to time tm+2hrs, uses this stretch as meal data
  - If there is a meal at some time tp in between tp>tm and tp< tm+2hrs, ignores the meal data at time tm and considers the meal at time tp instead.
  - If there is a meal at time tm+2hrs then considers the stretch from tm+1hr 30min to tm+4hrs as meal data.

##### Extraction: No Meal data
- Start of no meal is at time tm+2hrs where tm is the start of some meal. Finds all 2 hr stretches in a day that have no meal and do not fall within 2 hrs of the start of a meal.

---

#### Handling missing data:
- Applies linear interpolation to fill in stretches of missing data within meal times.

---

#### Test Data:
- The test data is a matrix of size N×24 where N is the total number of tests and 24 is the size of the CGM time series. N will have some distribution of meal and no meal data.

---

#### Output format:
- Generates "Result.csv" file as output, containing an Nx1 vector of binary digits where an entry is 1 if its corresponding row in the InsulinData was determined to be meal data, and 0 if determined to be no meal data.