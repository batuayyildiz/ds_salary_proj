# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 15:40:51 2020

@author: LENOVO
"""

import pandas as pd

df = pd.read_csv("glassdoor_jobs.csv")

# Remove first column (Unnamed : 0)
df.drop("Unnamed: 0", axis = 1, inplace = True)

# Salary parsing
df["hourly"] = df["Salary Estimate"].apply(lambda x: 1 if "per hour" in  x.lower() else 0)
df["employer_provided"] = df["Salary Estimate"].apply(lambda x: 1 if "employer provided salary:" in x.lower() else 0)

df = df[df["Salary Estimate"] != "-1"]

salary = df["Salary Estimate"].apply(lambda x: x.split("(")[0])
minus_kd = salary.apply(lambda x: x.replace("K", "").replace("$", ""))
min_hour = minus_kd.apply(lambda x: x.lower().replace("per hour", "").replace("employer provided salary:", ""))

df["min_salary"] = min_hour.apply(lambda x: int(x.split("-")[0]))
df["max_salary"] = min_hour.apply(lambda x: int(x.split("-")[1]))
df["avg_salary"] = (df["min_salary"] + df["max_salary"]) / 2

# Company name correction - text only
df["company_txt"] = df.apply(lambda x: x["Company Name"] if x["Rating"] < 0 else x["Company Name"][:-3], axis = 1)

# State field - Location column
df["job_state"] = df["Location"].apply(lambda x: x.split(",")[1])
print(df["job_state"].value_counts())

# Check job_state is whether or not in the same location with Headquarters
df["is_same_location"] = df.apply(lambda x: 1 if x["Location"] == x["Headquarters"] else 0, axis = 1)


# Finding age of company
df["company_age"] = df["Founded"].apply(lambda x: x if x < 1 else 2020 - x)

# Parsing of job description into different categories
# Python
df["python_yn"] = df["Job Description"].apply(lambda x: 1 if "python" in x.lower() else 0)
# R Studio
df["r_studio_yn"] = df["Job Description"].apply(lambda x: 1 if "r studio" in x.lower() or "r-studio" in x.lower() else 0)
# Spark
df["spark_yn"] = df["Job Description"].apply(lambda x: 1 if "spark" in x.lower() else 0)
# AWS
df["aws_yn"] = df["Job Description"].apply(lambda x: 1 if "aws" in x.lower() else 0)
# Excel
df["excel_yn"] = df["Job Description"].apply(lambda x: 1 if "excel" in x.lower() else 0)

# Check there are how many yes and how many no
print(df["python_yn"].value_counts())
print(df["r_studio_yn"].value_counts())
print(df["spark_yn"].value_counts())
print(df["aws_yn"].value_counts())
print(df["excel_yn"].value_counts())


df.to_csv("salary_data_cleaned.csv", index = False)