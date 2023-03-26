This folder contains all code involved in analyzing Collabera and Glassdoor open-ended texts
Order of scripts is chronological: later scripts depend on output produced by earlier scripts
Steps are not necessarily consecutive in nature: e.g., deductive linguistic feature modeling is a separate approach from topic modeling glassdoor reviews

Step 1: Preprocessing Survey Data
	Input: Collabera/Data/Collabera_Survey_Responses_all.csv, Collabera/Data/Collabera_HR_Perf.csv
	Output: analyses_data/preprocessed_survey_hr.csv
	Essential preprocessing steps: naming qualtric responses, averaging identification, and converting letters to numbers
	No preprocessing is done on open-ended texts

Step 2: Topic Modeling Survey Responses
2_preprocess_topic_modeling.ipynb
	Input: analyses_data/preprocessed_survey_hr.csv
	Output: analyses_survey_hr_topic_modeling.csv
	Standardizes survey features - averaging for burnout and identification, standardization for race
	Cleans survey text for topic modeling
2R_survey_responses_topic_modeling.Rmd
	Input: analyses_data/survey_hr_topic_modeling.csv

Step 3: Deductive Linguistic Feature Modeling
3_deductve_survey_measures.ipynb
	Input: analyses_data/preprocessed_survey_hr.csv
	Output: analyses_data/survey_hr_deductive_vars.csv
	Standardizes survey features - averaging for burnout and identification, standardization for race
	Produces linguistic features used for modeling identification using linear models, lasso, elasticnet, etc.
	Linguistic features include LIWC, pronoun counts, distilbert measures.
3R_deductive_vars_modeling.Rmd
	Input: analyses_data/survey_hr_deductive_vars.csv
	Runs models using deductive linguistic measures as features

Step 4: Topic Modeling Glassdoor Reviews and Survey Responses
4_glassdoor_btm.ipynb
	Input: reviews_new_processed.csv
	Output: BTM/glassdoor_data/pros.txt, BTM/glassdoor_data/cons.txt, glassdoor_reviews_custom.csv, glassdoor_companies.csv, glassdoor_reviews_only.csv
	Preprocesses raw review text and output texts that are appropriate to C-based BTM code and separate, smaller csvs for downstream topic modeling and analyses
4_glassdoor_lda.ipynb
	Input: glassdoor_reviews_only.csv, survey_hr_topic_modeling.csv, reviews_new_processed.csv
	Output: coco_glassdoor_topic_modeling.csv, glassdoor_topic_prop.csv, glassdoor_pros_top_words_lda_k10.csv, glassdoor_cons_top_words_lda_k10.csv
	Builds LDA model using glassdoor reviews and apply to Coco reviews
4R_analyze_coco_glassdoor_btm.Rmd
	Input: BTM/output_coco_pros/model/k20.pz_d, BTM/output_coco_cons/model/k20.pz_d, analyses_data/survey_hr_topic_modeling.csv
4R_analyze_coco_glassdoor_lda.Rmd
	Input: analyses_data/coco_glassdoor_topic_modeling.csv, glassdoor_topic_prop.csv, glassdoor_pros_top_words_lda_k10.csv, glassdoor_cons_top_words_lda_k10.csv, glassdoor_companies.csv
	Build linear model using topic proportions and apply linear model to Glassdoor reviews to estimate firm-level average identification