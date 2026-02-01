import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import requests
import zipfile
import io

# The direct download URL for the dataset
url = "https://analyse.kmi.open.ac.uk/open-dataset/download"

print("Downloading dataset...")
response = requests.get(url)

# Unzip the content into a folder named 'oulad_data'
with zipfile.ZipFile(io.BytesIO(response.content)) as z:
    z.extractall("FAIDM/oulad_data")
    print("Files extracted: ", z.namelist())


dir = 'FAIDM/oulad_data/'

studentInfo = pd.read_csv(dir + 'studentInfo.csv')
courses = pd.read_csv(dir + 'courses.csv')
assessments = pd.read_csv(dir + 'assessments.csv')
studentRegistration = pd.read_csv(dir + 'studentRegistration.csv')
studentAssessment = pd.read_csv(dir + 'studentAssessment.csv')
studentVle = pd.read_csv(dir + 'studentVle.csv')
vle = pd.read_csv(dir + 'vle.csv')

studentCourseRegistration = pd.merge(studentRegistration, courses, on=['code_module', 'code_presentation'], how='left')
students = pd.merge(studentInfo, studentCourseRegistration, on=['id_student', 'code_module', 'code_presentation'], how='left')

def clean_student_demographics(df):
  # IMD_BAND
  imd_mapping = {
      '0-10%': 5, '10-20%': 15, '20-30%': 25, '30-40%': 35, '40-50%': 45,
      '50-60%': 55, '60-70%': 65, '70-80%': 75, '80-90%': 85, '90-100%': 95
  }
  df['imd_band_clean'] = df['imd_band'].map(imd_mapping)
  df['imd_band_clean'] = df['imd_band_clean'].fillna(-1) # filling -1 as imd_band for unknown

  # AGE BAND
  age_midpoint_map = { '0-35': 18, '35-55': 45, '55<=': 60 }
  df['age_band_clean'] = df['age_band'].map(age_midpoint_map)

  # EDUCATION
  edu_standard_map = {
      'Post Graduate Qualification': 'Post_Grad',
      'HE Qualification': 'Higher_Ed',
      'A Level or Equivalent': 'A_Level',
      'Lower Than A Level': 'Below_A_Level',
      'No Formal quals': 'No_Formal'
  }
  df['edu_clean'] = df['highest_education'].map(edu_standard_map)

  # REGION MAP - Prevents Overfitting 
  region_map = {
      'Scotland': 'North', 'North Western Region': 'North', 'Yorkshire Region': 'North', 'North Region': 'North',
      'South Region': 'South', 'South West Region': 'South', 'South East Region': 'South',
      'East Anglian Region': 'Midlands_East', 'West Midlands Region': 'Midlands_East', 'East Midlands Region': 'Midlands_East',
      'London Region': 'London',
      'Wales': 'Devolved_Ireland', 'Ireland': 'Devolved_Ireland'
  }
  df['region_clean'] = df['region'].map(region_map)
  return df

students = clean_student_demographics(students)

# ONE HOT ENCODING FOR STUDENTS 
cols_to_encode = ['gender', 'disability']
df_students = pd.get_dummies(students, columns=cols_to_encode, drop_first=True, dtype=int)
cols_to_encode = ['edu_clean', 'region_clean']
df_students = pd.get_dummies(df_students, columns=cols_to_encode, dtype=int)

cols_to_drop = ['highest_education', 'region', 'imd_band', 'age_band']
df_students = df_students.drop(columns=[c for c in cols_to_drop if c in df_students.columns])
df_students.head()

# IMPORTANT ------ # Removing all the exam scores from the data 

assessments = assessments[assessments['assessment_type'] != 'Exam']

# INPORTANT ------ # Removing all student assessement records where data_submission < 0 and is_banked == 0 

studentAssessment = studentAssessment[~((studentAssessment['date_submitted'] < 0) & (studentAssessment['is_banked'] == 0))]

def map_assessment_weights(group):
    type_weights = group.groupby('assessment_type')['weight'].agg(['sum', 'size'])
    return ', '.join([
        f"{t} ({int(row['size'])} , {row['sum']}% )" 
        for t, row in type_weights.sort_index().iterrows()
    ])

df_modules =  assessments.groupby(['code_module', 'code_presentation']).apply(
    lambda x: pd.Series({
        'total_weight': x['weight'].sum(),
        'assessment_count': x['weight'].count(),
        'optional_count': (x['weight'] == 0).sum(),
        'core_count': (x['weight'] != 0).sum(),
        'exam_count': (x['assessment_type'] == 'Exam').sum(),
        'exam_weight': x[x['assessment_type'] == 'Exam']['weight'].sum(),
        # 'weight_by_type': map_assessment_weights(x)
    })
).reset_index()

# STUDENT PERFORMANCE
df_student_performance = pd.merge(studentAssessment, assessments, on=['id_assessment'])
df_student_performance['core_score'] = df_student_performance['score'] * df_student_performance['weight'] * 0.01
df_student_performance['optional_score'] = df_student_performance['score'].where(df_student_performance['weight'] == 0, 0)
df_student_performance['exam_score'] = np.where(
    df_student_performance['assessment_type'] == 'Exam',
    df_student_performance['score'] * df_student_performance['weight'] * 0.01,
    0
)
df_student_performance['submission_gap'] = df_student_performance['date_submitted'] - df_student_performance['date']

df_student_performance = df_student_performance.groupby(['id_student', 'code_module', 'code_presentation']).agg(
    attempted_core_score = ('core_score', 'sum'),
    attempted_core_weight = ('weight', 'sum'),
    attempted_core = ('weight', lambda x: (x > 0).sum()),
    attempted_optional_score = ('optional_score', 'sum'),
    attempted_optional = ('optional_score', lambda x: (x != 0).sum()),
    avg_submission_gap = ('submission_gap', 'mean'),
    # std_submission_gap = ('submission_gap', 'std'),
    has_banked_score = ('is_banked', 'max'),
    is_final_exam_score_avlbl = ('assessment_type', lambda x: (x == 'Exam').sum()),
    exam_score = ('exam_score', 'sum'),
).reset_index()

df_student_performance['is_early_submitter'] = (df_student_performance['avg_submission_gap'] < -5).astype(int)


df_student_performance = pd.merge(df_student_performance, df_modules, on=['code_module', 'code_presentation'])
# df_student_performance[df_student_performance['attempted_optional'] > 0].head()
df_student_performance['performance_efficiency'] = (
    df_student_performance['attempted_core_score'] / (df_student_performance['attempted_core_weight'] * 0.01 + 1e-5)
)

df_student_performance['core_completion_ratio'] = df_student_performance['attempted_core'] / (df_student_performance['core_count']* 0.01 + 1e-5)
df_student_performance['optional_completion_ratio'] = df_student_performance['attempted_optional'] / (df_student_performance['optional_count'] + 1e-5)

df_student_performance['total_attempt_ratio'] = (
    (df_student_performance['attempted_core'] + df_student_performance['attempted_optional']) / 
    df_student_performance['assessment_count']
)

df_student_performance['attempted_optional_score'] = df_student_performance['attempted_optional_score'] / (df_student_performance['optional_count'] + 1e-5)


# df_student_performance['avg_submission_gap'] = df_student_performance['avg_submission_gap'].clip(-50, 50)

# Final column selection for the ML ready table
df_student_performance = df_student_performance[[
    'id_student', 'code_module', 'code_presentation',
    'performance_efficiency', 'total_attempt_ratio',
    'attempted_core_score', 'core_completion_ratio',
    'attempted_optional_score', 'optional_completion_ratio', 
    'avg_submission_gap', 'is_early_submitter',
    'exam_score', 'has_banked_score', 'is_final_exam_score_avlbl'
]]


activity_map = {
    'forumng': 'Social', 'ouwiki': 'Social', 'oucollaborate': 'Social',
    'quiz': 'Active_Study', 'externalquiz': 'Active_Study', 'questionnaire': 'Active_Study',
    'oucontent': 'Content', 'resource': 'Content', 'url': 'Content', 'page': 'Content', 'folder': 'Content',
    'homepage': 'Nav', 'subpage': 'Nav', 'ouelluminate': 'Nav'
}
vle_with_cat = vle.copy()
vle_with_cat['activity_category'] = vle['activity_type'].map(activity_map)

cols_to_drop = ['activity_type', 'week_from', 'week_to']
vle_with_cat = vle_with_cat.drop(columns=[c for c in cols_to_drop if c in vle_with_cat.columns])
vle_with_cat.head()

df_student_vle = studentVle.groupby(["id_student", "code_module", "code_presentation", "id_site", "date"]).agg(
    sum_click=("sum_click", "sum")
).reset_index()

df_student_vle["early_access"] = (df_student_vle["date"] < 0).astype(int)
df_student_vle["week"] = (df_student_vle["date"] // 7).astype(int)
df_student_vle.head()


df_engagement = pd.merge(df_student_vle, vle_with_cat, on=['id_site', 'code_module', 'code_presentation'], how='inner')

df_engagement_agg = df_engagement.groupby(["id_student", "code_module", "code_presentation"]).agg(
    total_clicks=("sum_click", "sum"),
    active_days=("date", "nunique"),
    active_weeks=("week", "nunique"),
    has_early_access=("early_access", "max")
).reset_index()


df_engagement_agg_by_category = df_engagement.groupby(['id_student', 'code_module', 'code_presentation', 'activity_category']).agg(
    click = ('sum_click', 'sum'),
    active_days=('date', 'nunique'),
).unstack(fill_value=0)

df_engagement_agg_by_category.columns = [f"{metric}{ "_" if cat else "" }{cat}" for metric, cat in df_engagement_agg_by_category.columns]
df_engagement_agg_by_category = df_engagement_agg_by_category.reset_index()


df_student_engagement = pd.merge(df_engagement_agg, df_engagement_agg_by_category, on=['id_student', 'code_module', 'code_presentation'], how='left')


df_performance_engagement = pd.merge( df_student_performance, df_student_engagement, on=['id_student', 'code_module', 'code_presentation'], how='outer')

df_student_performance_engagement = pd.merge(df_students, df_performance_engagement, on=['id_student', 'code_module', 'code_presentation'], how='inner')

df_student_performance_engagement.shape


# TAKING STUDENTS WITH EITHER PERFORMANCE DATA OR VLE DATA AVAILABLE 
df_final = df_student_performance_engagement[
    (~df_student_performance_engagement['exam_score'].isnull()) & 
    (~df_student_performance_engagement['total_clicks'].isnull())
  ]

cols_to_encode = ['code_module']
df_final = pd.get_dummies(df_final, columns=cols_to_encode, dtype=int)
df_final = df_final[[
       'code_module_AAA', 'code_module_BBB','code_module_CCC', 'code_module_DDD', 'code_module_EEE', 'code_module_FFF', 'code_module_GGG', 
       'num_of_prev_attempts',
       'studied_credits', 
       'imd_band_clean',
       'age_band_clean',
       'edu_clean_A_Level', 'edu_clean_Below_A_Level', 'edu_clean_Higher_Ed', 'edu_clean_No_Formal', 'edu_clean_Post_Grad', 
       'region_clean_Devolved_Ireland','region_clean_London', 'region_clean_Midlands_East', 'region_clean_North', 'region_clean_South', 
       'gender_M', 
       'disability_Y',
       'total_attempt_ratio','performance_efficiency',  
       'attempted_core_score', 'core_completion_ratio', 
       'attempted_optional_score', 'optional_completion_ratio', 
       'has_banked_score',
       'avg_submission_gap', 'is_early_submitter',
       'total_clicks', 'active_days', 'active_weeks', 'has_early_access',
       'click_Active_Study', 'click_Content', 'click_Nav', 'click_Social',
       'active_days_Active_Study', 'active_days_Content', 'active_days_Nav', 'active_days_Social', 
       'final_result']]
df_final.isna().sum()
df_final = df_final.fillna(0)


df_final.to_csv('FAIDM/df_final_raw.csv', index=False)

from sklearn.preprocessing import StandardScaler

cols_to_scale = ['num_of_prev_attempts', 'studied_credits', 'imd_band_clean', 
    'age_band_clean', 'total_attempt_ratio', 'performance_efficiency', 
    'attempted_core_score', 'core_completion_ratio', 'attempted_optional_score',
    'optional_completion_ratio', 'avg_submission_gap', 'total_clicks', 
    'active_days', 'active_weeks', 'click_Active_Study', 'click_Content', 
    'click_Nav', 'click_Social', 'active_days_Active_Study', 
    'active_days_Content', 'active_days_Nav', 'active_days_Social']

scaler = StandardScaler()

df_final_scaled = df_final.copy()
df_final_scaled[cols_to_scale] = scaler.fit_transform(df_final[cols_to_scale])

df_final_scaled.describe()


df_final_scaled.to_csv('FAIDM/final_all_scaled_without_exam_score.csv', index=False)