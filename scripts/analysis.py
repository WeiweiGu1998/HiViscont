import pandas
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import tstd, tmean, ttest_rel,shapiro, wilcoxon



NUMBERED_PRE_SURVEY_PATH = "/home/local/ASUAD/weiweigu/human_subject_data/numbered_pre_survey.csv"
TEXT_PRE_SURVEY_PATH = "/home/local/ASUAD/weiweigu/human_subject_data/text_presurvey.csv"

NUMBERED_POST_SURVEY_PATH = "/home/local/ASUAD/weiweigu/human_subject_data/numbered_post_survey.csv"
TEXT_POST_SURVEY_PATH = "/home/local/ASUAD/weiweigu/human_subject_data/text_post_survey.csv"

NUMBERED_COMPARATIVE_SURVEY_PATH = "/home/local/ASUAD/weiweigu/human_subject_data/numbered_comparison.csv"
TEXT_COMPARATIVE_SURVEY_PATH = "/home/local/ASUAD/weiweigu/human_subject_data/text_comparison.csv"

OBJECTIVE_RESULTS_PATH = "/home/local/ASUAD/weiweigu/human_subject_data/objective.csv"


PILOT_IDS = [
"8599384",
"7118826",
"8893457"
]


SUBJECT_ID_LIST = [
    "1670074",
    "1827562",
    "2139391",
    "2776259",
    "4053843",
    "7377450",
    "7392259",
    "8322982",
    "8587281",
    "9193125",
    "9289671",
    "2623099",
    "3949836",
    "6626945",
    "9262202",
    "7777344",
    "7469444",
    "2095456"
]
TRUST_REVERSE_NUMBERS = [
    1,2,3,4,5
]
FALCON_GEN = [
    0.8890,
    0.8934,
    0.8791,
    0.868,
    0.8826
]

OURS_GEN = [
    0.9127,
    0.9043,
    0.8996,
    0.8965,
    0.9072
]

FALCON_FAM = [
    0.8514,
    0.8625,
    0.8417,
    0.8376,
    0.8233
]

OURS_FAM = [
    0.9330,
    0.9109,
    0.9001,
    0.8871,
    0.9047
]

FALCON_ORD = [
    0.8319,
    0.8695,
    0.8507,
    0.8687,
    0.7925
]

OURS_ORD = [
    0.9225,
    0.9052,
    0.9247,
    0.9196,
    0.9253
]


FALCON_AFF = [
    0.5803,
    0.6794,
    0.5460,
    0.4349,
    0.6271
]

OURS_AFF = [
    0.7442,
    0.9497,
    0.9785,
    0.9004,
    0.9204
]

FALCON_COLOR = [
    0.9053,
    0.8945,
    0.7688,
    0.8941,
    0.9009
]
OURS_COLOR = [
    0.9842,
    0.9873,
    0.9914,
    1,
    0.9990
]
FALCON_NON_LEAF = [
    0.6683,
    0.5777,
    0.6961,
    0.6495,
    0.7162
]

OURS_NON_LEAF = [
    0.8422,
    0.7944,
    0.8704,
    0.9499,
    0.8349
]

IMPRESSION_REVERSE_NUMBERS = [
]
NATURAL_REVERSE_NUMBERS = [
]

CUSTOM_REVERSE_NUMBERS = [
    
]
USABILITY_REVERSE_NUMBERS = [
 2, 4, 6, 8, 10
]
COMPARATIVE_REVERSE_NUMBERS = [
    2, 4, 5, 8
]
MAPPING_TO_VALUE_FOR_CUSTOM_QUESTIONS = {
    "Strongly Disagree": 1,
    "Disagree": 2,
    "Somewhat Disagree": 3,
    "Neither Agree or Disagree": 4,
    "Somewhat Agree": 5,
    "Agree": 6,
    "Strongly Agree": 7
}
def main():
    numbered_post_survey = pandas.read_csv(NUMBERED_POST_SURVEY_PATH)
    text_post_survey = pandas.read_csv(TEXT_POST_SURVEY_PATH)
    objective_results = pandas.read_csv(OBJECTIVE_RESULTS_PATH)
    numbered_comparative_survey = pandas.read_csv(NUMBERED_COMPARATIVE_SURVEY_PATH)
    post_survey_column_names = list(numbered_post_survey.columns)
    comparative_survey_column_names = [x for x in list(numbered_comparative_survey.columns) if x[:2] == "Q1"]
    # scale of 7
    trust_questions = [x for x in post_survey_column_names if x[:2] == "Q5"]
    # scale of 5
    impression_questions = [x for x in post_survey_column_names if x[:2] == "Q6"]
    # scale of 7
    usability_questions = [x for x in post_survey_column_names if x[:2] == "Q8"]
    # scale of 5
    natural_questions = [x for x in post_survey_column_names if x[:2] == "Q9"]
    # scale of 7
    our_own_post_survey_questions = [x for x in post_survey_column_names if x[:3] == "Q16"]
    # bound to change for test
    ids = SUBJECT_ID_LIST
    #ids = PILOT_IDS
    #ids = numbered_post_survey["Q4"].unique()[3:]
    post_survey_data = {}
    comparative_survey_data = {}
    #breakpoint()
    for ide in ids:
        rows_of_the_id_from_postsurvey = numbered_post_survey[numbered_post_survey["Q4"] == ide]
        rows_of_the_id_from_text_post_survey = text_post_survey[text_post_survey["Q4"] == ide]
        rows_of_the_id_from_comparative_survey = numbered_comparative_survey[numbered_comparative_survey["Participant Id"] == ide]

        if (int(ide[-1])%2 == 0):
            #Even id: SYS-1 -> ours, SYS-2 -> FALCON
            our_line_post_survey = rows_of_the_id_from_postsurvey[rows_of_the_id_from_postsurvey["Q35"] == "1"]
            our_line_text_post_survey = rows_of_the_id_from_text_post_survey[rows_of_the_id_from_postsurvey["Q35"] == "1"]
            FALCON_line_post_survey = rows_of_the_id_from_postsurvey[rows_of_the_id_from_postsurvey["Q35"] == "4"]
            FALCON_line_text_post_survey = rows_of_the_id_from_text_post_survey[rows_of_the_id_from_postsurvey["Q35"] == "4"]
            mapping_for_comparative = {
                "1": "ours",
                "2": "FALCON",
                "3": "same"
            }
        else:
            our_line_post_survey = rows_of_the_id_from_postsurvey[rows_of_the_id_from_postsurvey["Q35"] == "4"]
            our_line_text_post_survey = rows_of_the_id_from_text_post_survey[rows_of_the_id_from_postsurvey["Q35"] == "4"]
            FALCON_line_post_survey = rows_of_the_id_from_postsurvey[rows_of_the_id_from_postsurvey["Q35"] == "1"]
            FALCON_line_text_post_survey = rows_of_the_id_from_text_post_survey[rows_of_the_id_from_postsurvey["Q35"] == "1"]
            mapping_for_comparative = {
                "2": "ours",
                "1": "FALCON", 
                "3": "same"
            }


        # code to compute scores for this user for the post survey
        # Trust score
        our_trust_score = 0
        FALCON_trust_score = 0

        for trust_question_id in trust_questions:
            question_sub_id = int(trust_question_id.split("_")[1])
            if question_sub_id in TRUST_REVERSE_NUMBERS:
                # reverse the scores
                try:
                    our_scores = 8 - int(our_line_post_survey[trust_question_id])
                except:
                    breakpoint()
                falcon_scores = 8 - int(FALCON_line_post_survey[trust_question_id])
            else:
                our_scores = int(our_line_post_survey[trust_question_id])
                falcon_scores = int(FALCON_line_post_survey[trust_question_id])
            our_trust_score += our_scores
            FALCON_trust_score += falcon_scores

        our_impression_score = 0
        FALCON_impression_score = 0
        # Impression score is for scale of 5
        # reverse will be 6-x
        for impression_question_id in impression_questions:
            question_sub_id = int(impression_question_id.split("_")[1])
            if question_sub_id in IMPRESSION_REVERSE_NUMBERS:
                # reverse the scores
                our_scores = 6 - int(our_line_post_survey[impression_question_id])
                falcon_scores = 6 - int(FALCON_line_post_survey[impression_question_id])
            else:
                our_scores = int(our_line_post_survey[impression_question_id])
                falcon_scores = int(FALCON_line_post_survey[impression_question_id])
            our_impression_score += our_scores
            FALCON_impression_score += falcon_scores

        our_usability_score = 0
        FALCON_usability_score = 0
        # Usability score is for scale of 7
        # reverse will be 8-x
        for usability_question_id in usability_questions:
            question_sub_id = int(usability_question_id.split("_")[1])
            if question_sub_id in USABILITY_REVERSE_NUMBERS:
                # reverse the scores
                our_scores = 8 - int(our_line_post_survey[usability_question_id])
                falcon_scores = 8 - int(FALCON_line_post_survey[usability_question_id])
            else:
                our_scores = int(our_line_post_survey[usability_question_id])
                falcon_scores = int(FALCON_line_post_survey[usability_question_id])
            our_usability_score += our_scores
            FALCON_usability_score += falcon_scores

        our_natural_score = 0
        FALCON_natural_score = 0
        # Natural score is for scale of 5
        # reverse will be 6-x
        for natural_question_id in natural_questions:
            question_sub_id = int(natural_question_id.split("_")[1])
            if question_sub_id in NATURAL_REVERSE_NUMBERS:
                # reverse the scores
                our_scores = 6 - int(our_line_post_survey[natural_question_id])
                falcon_scores = 6 - int(FALCON_line_post_survey[natural_question_id])
            else:
                our_scores = int(our_line_post_survey[natural_question_id])
                falcon_scores = int(FALCON_line_post_survey[natural_question_id])
            our_natural_score += our_scores
            FALCON_natural_score += falcon_scores

        our_custom_score = 0
        FALCON_custom_score = 0
        #  score is for scale of 7
        # reverse will be 8-x

        for custom_question_id in our_own_post_survey_questions:
            question_sub_id = int(custom_question_id.split("_")[1])
            if question_sub_id in CUSTOM_REVERSE_NUMBERS:
                # reverse the scores
                our_scores = 8 - int(MAPPING_TO_VALUE_FOR_CUSTOM_QUESTIONS[str(list(our_line_text_post_survey[custom_question_id])[0])])
                falcon_scores = 8 - int(MAPPING_TO_VALUE_FOR_CUSTOM_QUESTIONS[str(list(FALCON_line_text_post_survey[custom_question_id])[0])])
            else:
                our_scores = int(
                    MAPPING_TO_VALUE_FOR_CUSTOM_QUESTIONS[str(list(our_line_text_post_survey[custom_question_id])[0])])
                falcon_scores = int(
                    MAPPING_TO_VALUE_FOR_CUSTOM_QUESTIONS[str(list(FALCON_line_text_post_survey[custom_question_id])[0])])
            our_custom_score += our_scores
            FALCON_custom_score += falcon_scores
        this_user_data = {
            "FALCON_scores": {
                "trust_score": FALCON_trust_score,
                "impression_score": FALCON_impression_score,
                "usability_score": FALCON_usability_score,
                "natural_score": FALCON_natural_score,
                "our_own_score": FALCON_custom_score,
            },
            "our_model_scores": {
                "trust_score": our_trust_score,
                "impression_score": our_impression_score,
                "usability_score": our_usability_score,
                "natural_score": our_natural_score,
                "our_own_score": our_custom_score,
            },
        }
        post_survey_data[ide] = this_user_data

        # process the comparative survey data
        comparative_scores = {
            "ours": {
                "total":0,
                "subscore": []
                },

            "FALCON": {
                "total":0,
                "subscore": []
                }
        }
        for q in comparative_survey_column_names:
            try:
                this_answer = int(rows_of_the_id_from_comparative_survey[q])
            except:
                breakpoint()
            if (this_answer != 3):
                try:
                    question_sub_id = int(q.split("_")[1])
                except:
                    breakpoint()
                if question_sub_id not in COMPARATIVE_REVERSE_NUMBERS:
                    comparative_scores[mapping_for_comparative[str(this_answer)]]["total"] += 1
                    comparative_scores[mapping_for_comparative[str(this_answer)]]["subscore"].append(1)
                    comparative_scores[mapping_for_comparative[str(3 - this_answer)]]["subscore"].append(0)
                    
                else:
                    comparative_scores[mapping_for_comparative[str(3-this_answer)]]["total"] += 1
                    comparative_scores[mapping_for_comparative[str(this_answer)]]["subscore"].append(0)
                    comparative_scores[mapping_for_comparative[str(3 - this_answer)]]["subscore"].append(1)
            else:
                comparative_scores["ours"]["subscore"].append(0)
                comparative_scores["FALCON"]["subscore"].append(0)

        comparative_survey_data[ide] = comparative_scores

    our_final_trust_scores = []
    our_final_usability_scores = []
    our_final_impression_scores = []
    our_final_natural_scores = []
    our_final_custom_scores = []
    our_final_comparative_scores = []
    our_final_comparative_subscores = [[] for _ in range(8)]

    FALCON_final_trust_scores = []
    FALCON_final_usability_scores = []
    FALCON_final_impression_scores = []
    FALCON_final_natural_scores = []
    FALCON_final_custom_scores = []
    FALCON_final_comparative_scores = []
    FALCON_final_comparative_subscores = [[] for _ in range(8)]

    for uid in ids:
        our_final_trust_scores.append(post_survey_data[uid]['our_model_scores']['trust_score'])
        FALCON_final_trust_scores.append(post_survey_data[uid]['FALCON_scores']['trust_score'])
        our_final_usability_scores.append(post_survey_data[uid]['our_model_scores']['usability_score'])
        FALCON_final_usability_scores.append(post_survey_data[uid]['FALCON_scores']['usability_score'])
        our_final_impression_scores.append(post_survey_data[uid]['our_model_scores']['impression_score'])
        FALCON_final_impression_scores.append(post_survey_data[uid]['FALCON_scores']['impression_score'])

        our_final_natural_scores.append(post_survey_data[uid]['our_model_scores']['natural_score'])
        FALCON_final_natural_scores.append(post_survey_data[uid]['FALCON_scores']['natural_score'])

        our_final_custom_scores.append(post_survey_data[uid]['our_model_scores']['our_own_score'])
        FALCON_final_custom_scores.append(post_survey_data[uid]['FALCON_scores']['our_own_score'])

        our_final_comparative_scores.append(comparative_survey_data[uid]['ours']['total'])
        FALCON_final_comparative_scores.append(comparative_survey_data[uid]['FALCON']['total'])

        for i in range(8):
            our_final_comparative_subscores[i].append(comparative_survey_data[uid]['ours']['subscore'][i])
            FALCON_final_comparative_subscores[i].append(comparative_survey_data[uid]['FALCON']['subscore'][i])

    for stat_type, our_stat, FALCON_stat in zip(
            ["Trust", "Usability", "Intelligence", "Natural", "Custom", "Comparative Total"],
            [our_final_trust_scores, our_final_usability_scores, our_final_impression_scores, our_final_natural_scores, our_final_custom_scores, our_final_comparative_scores],
            [FALCON_final_trust_scores, FALCON_final_usability_scores, FALCON_final_impression_scores, FALCON_final_natural_scores, FALCON_final_custom_scores, FALCON_final_comparative_scores]):
            print("===================================================================================================================")
            difference = [our_stat[i] - FALCON_stat[i] for i in range(len(our_stat))]
            print(f"Statistics on {stat_type}:")
            our_mean = tmean(our_stat)
            our_std = tstd(our_stat)
            FALCON_mean = tmean(FALCON_stat)
            FALCON_std = tstd(FALCON_stat)
            ttest_out = ttest_rel(our_stat, FALCON_stat, alternative='greater')
            shapiro_out = shapiro(difference)
            wilcoxon_out = wilcoxon(difference)
            print("Regular results:")
            print(f"Our Stat: {our_mean} +/- {our_std}")
            print(f"FALCON Stat: {FALCON_mean} +/- {FALCON_std}")
            print("Shapiro-Wilk Normality Test:")
            print(f"p_value: {shapiro_out.pvalue}")
            print(f"statistic: {shapiro_out.statistic}")
            print("Paired t-test results:")
            print(f"statistic: {ttest_out.statistic}")
            print(f"p_value: {ttest_out.pvalue}")
            print(f"df: {ttest_out.df}")
            print("Wilcoxon results:")
            print(f"statistic: {wilcoxon_out.statistic}")
            print(f"p_value: {wilcoxon_out.pvalue}")

    for i in range(8):
        our_stat = our_final_comparative_subscores[i]
        FALCON_stat = FALCON_final_comparative_subscores[i]
        print(
            "===================================================================================================================")
        difference = [our_stat[j] - FALCON_stat[j] for j in range(len(our_stat))]
        print(f"Statistics on {stat_type}:")
        our_mean = tmean(our_stat)
        our_std = tstd(our_stat)
        FALCON_mean = tmean(FALCON_stat)
        FALCON_std = tstd(FALCON_stat)
        ttest_out = ttest_rel(our_stat, FALCON_stat, alternative='greater')
        shapiro_out = shapiro(difference)
        wilcoxon_out = wilcoxon(difference)
        print("Regular results:")
        print(f"Our Stat: {our_mean} +/- {our_std}")
        print(f"FALCON Stat: {FALCON_mean} +/- {FALCON_std}")
        print("Shapiro-Wilk Normality Test:")
        print(f"p_value: {shapiro_out.pvalue}")
        print(f"statistic: {shapiro_out.statistic}")
        print("Paired t-test results:")
        print(f"statistic: {ttest_out.statistic}")
        print(f"p_value: {ttest_out.pvalue}")
        print(f"df: {ttest_out.df}")
        print("Wilcoxon results:")
        print(f"statistic: {wilcoxon_out.statistic}")
        print(f"p_value: {wilcoxon_out.pvalue}")

    # compute objective results
    our_node_level = list(objective_results['our node level'])
    our_total = list(objective_results['our total'])
    FALCON_node_level = list(objective_results['FALCON node level'])
    FALCON_total = list(objective_results['total'])
    for stat_type, our_stat, FALCON_stat in zip(["total", "node accuracy"], [our_total, our_node_level], [FALCON_total, FALCON_node_level]):
        print(
            "===================================================================================================================")
        difference = [our_stat[i] - FALCON_stat[i] for i in range(len(our_stat))]
        print(f"Statistics on {stat_type}:")
        our_mean = tmean(our_stat)
        our_std = tstd(our_stat)
        FALCON_mean = tmean(FALCON_stat)
        FALCON_std = tstd(FALCON_stat)
        ttest_out = ttest_rel(our_stat, FALCON_stat, alternative='greater')
        shapiro_out = shapiro(difference)
        wilcoxon_out = wilcoxon(difference)
        print("Regular results:")
        print(f"Our Stat: {our_mean} +/- {our_std}")
        print(f"FALCON Stat: {FALCON_mean} +/- {FALCON_std}")
        print("Shapiro-Wilk Normality Test:")
        print(f"p_value: {shapiro_out.pvalue}")
        print(f"statistic: {shapiro_out.statistic}")
        print("Paired t-test results:")
        print(f"statistic: {ttest_out.statistic}")
        print(f"p_value: {ttest_out.pvalue}")
        print(f"df: {ttest_out.df}")
        print("Wilcoxon results:")
        print(f"statistic: {wilcoxon_out.statistic}")
        print(f"p_value: {wilcoxon_out.pvalue}")
    breakpoint()
    for name, our_stat, falcon_stat in zip(["GEN", "FAM", "ORD", "COLOR", "AFF", "NON_LEAF"],[OURS_GEN, OURS_FAM, OURS_ORD, OURS_COLOR, OURS_AFF, OURS_NON_LEAF], [FALCON_GEN, FALCON_FAM, FALCON_ORD, FALCON_COLOR, FALCON_AFF, FALCON_NON_LEAF]):
  
        print("==========================================================")
        diff = [our_stat[i] - falcon_stat[i] for i in range(5)]
        wilcoxon_out = wilcoxon(diff, alternative="greater")
        p=wilcoxon_out.pvalue
        print(f"stats for {name}")
        print(diff)
        print(f"Z:{wilcoxon_out.statistic}")
        print(f"p-value: {p}")
    breakpoint()
    return

if __name__ == "__main__":
    main()
