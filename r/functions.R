library(httr)

# Function to fetch REDCap report
fetch_report <- function(url, token, report_id) {
  formData <- list(
    "token" = token,
    content = "report",
    format = "csv",
    report_id = report_id,
    csvDelimiter = "",
    rawOrLabel = "label",
    rawOrLabelHeaders = "raw",
    exportCheckboxLabel = "false",
    returnFormat = "json"
  )
  
  # Make the POST request
  response <- POST(url, body = formData, encode = "form")
  
  # Extract content
  df <- suppressMessages(content(response, as = "parsed", type = "text/csv"))
  
  return(df)
}

count_individuals <- function(data, condition) {
  nrow(subset(data, eval(parse(text = condition))))
}

library(dplyr)
library(lubridate)
library(labelled)

clean_and_transform_table1_data <- function(df) {
  df_cleaned <- df |>
    # Remove people we did not successfully contact
    filter(!is.na(contact_refused)) |>
    filter(!record_id %in% c('217', '287')) |>  # Remove wrong phone numbers
    
    # Standardize dx naming conventions
    mutate(dx_type = recode(dx_type,
                            "Parkinson's disease" = "Parkinsons disease",
                            "Essential Tremor" = "Essential Tremor",
                            "Atypical Parkinsonism/PSP" = "Parkinsonism",
                            "Parkinsonism" = "Parkinsonism",
                            "Post traumatic Parkinson's" = "Parkinsonism",
                            "PSP" = "Parkinsonism"),
           enrollment_date = as.Date(enrollment_date, format = "%Y-%m-%d"),
           cf_dt = as.Date(cf_dt, format = "%Y-%m-%d %H:%M:%S"),
           enroll_date = ifelse(is.na(cf_dt), enrollment_date, cf_dt) |> as.Date(),
           dob = as.Date(dob),
           age = interval(start = dob, end = enroll_date) / years(1),
           diagnosis_date = as.Date(diagnosis_date),
           disease_duration = interval(start = diagnosis_date, end = enroll_date) / years(1),
           pd_yesno = recode(pd_yesno, 'Yes' = 'People with Movement Disorders', 'No' = 'Caregivers'),
           dx_type = ifelse(pd_yesno == 'Caregivers', "Caregivers", dx_type),
           mds_updrs_iv_total = ifelse(dx_type == "Essential Tremor", NA, mds_updrs_iv_total),
           gender = recode(gender, 'Man' = 'Men', 'Woman' = 'Women'),
           education = recode(education, 
                              "Completed junior college (associate's degree, technical training, etc...)" = 'Completed junior college'),
           education = factor(education, levels = c('Completed high school', 'Completed junior college', 'Completed college', 'Completed graduate degree'))) |>
  
    # Set variable labels
    set_variable_labels(
      age = "Age, years",
      gender = "Gender",
      race = "Race",
      ethnicity = "Ethnicity",
      education = "Education",
      disease_duration = "Disease Duration, years",
      mds_updrs_i_total = "MDS UPDRS Part 1",
      mds_updrs_ii_total = "MDS UPDRS Part 2",
      mds_updrs_iv_total = "MDS UPDRS Part 4",
      has_facebook = "Has Facebook",
      fb_data = "Facebook data submitted"
    )

  return(df_cleaned)
}