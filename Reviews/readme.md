Step 1: Preprocessing Survey Data
Step 2: Topic Modeling Survey Responses
2_preprocess_topic_modeling.ipynb
	Input: analyses_data/preprocessed_survey_hr.csv
	Output: analyses_survey_hr_topic_modeling.csv
	Standardizes survey features - averaging for burnout and identification, standardization for race
	Cleans survey text for topic modeling
2R_survey_responses_topic_modeling.Rmd
	Input: analyses_data/survey_hr_topic_modeling.csv
Step 3: Deductive Linguistic Feature Modeling
3_deductive_survey_measures.ipynb
	Input: analyses_data/preprocessed_survey_hr.csv
	Output: analyses_data/survey_hr_deductive_vars.csv
	Standardizes survey features - averaging for burnout and identification, standardization for race
	Produces linguistic features used for modeling identification using linear models, lasso, elasticnet, etc.
	Linguistic features include LIWC, pronoun counts, distilbert measures.
3_apply_distilbert_survey.ipynb
	Input:
	Output:
	Applies Distilbert model trained on Glassdoor reviews on review-like survey responses
	Distilbert model is trained in Gooogle Colab for its GPUs
3R_deductive_vars_modeling.Rmd
	Input: analyses_data/survey_hr_deductive_vars.csv
	Runs models using deductive linguistic measures as features
Step 4: Topic Modeling Glassdoor Reviews and Survey Responses
4_glassdoor_btm.ipynb
	Input: /ifs/projects/amirgo-identification/glassdoor_data/reviews_new_processed.csv
	Output: /ifs/projects/amirgo-identification/glassdoor_data/glassdoor_companies.csv
			/ifs/projects/amirgo-identification/glassdoor_data/glassdoor_reviews_only.csv
			/ifs/projects/amirgo-identification/glassdoor_data/glassdoor_reviews_custom.csv
			/ifs/projects/amirgo-identification/glassdoor_data/pros.txt
			/ifs/projects/amirgo-identification/glassdoor_data/cons.txt
	Prepares data for building BTM models as well as generating other smaller datasets that are useful
	for downstream processing
4_glassdoor_lda.ipynb
	Builds LDA topic models using data already cleaned by 4_glassdoor_btm.ipynb
	Uses the resulting models to run predictions on the Coco survey-based glassdoor data
	Input: /ifs/projects/amirgo-identification/glassdoor_data/glassdoor_reviews_only.csv
	Output: /ifs/projects/amirgo-identification/glassdoor_lda_model/pros_dictionary.gensim
	/ifs/projects/amirgo-identification/glassdoor_lda_model/cons_dictionary.gensim

/ifs/projects/amirgo-identification/BTM/script
runBTM_pros.sh runBTM_cons.sh
These scripts run BTM models on Glassdoor reviews, as the amount of Glassdoor review data available made it hard to run BTM models using R. The focal training step in this folder is written in C.
	Input: pros.txt, cons.txt
	Output: output_pros/*, output_cons/*
inferBTM_coco.sh
These scripts infer BTM topic model proportions on Coco survey-based Glassdoor reviews
	Input: output_pros/*, output_cons/*
	Output: output_coco_pros/*, output_coco_cons/*
Miscellaneous: Producing Reputation Quality Checks
	Input: /Users/Lara/Documents/Stanford/Research/Glassdoor/harris_reputation_poll.csv
		/Users/Lara/Documents/Stanford/Research/Glassdoor/forbes_reputation_poll.csv
	Output: /Users/Lara/Documents/Stanford/Research/Glassdoor/harris_glassdoor_match.csv
	/Users/Lara/Documents/Stanford/Research/Glassdoor/forbes_glassdoor_match.csv
	Matches manually copied Harris Reputation Poll 2021 and Forbes Most Admired Companies Poll 2021 with
	Glassdoor using fuzzy matching on names and manual disambiguation
