
##########################################################################
#LOAD LIBRARIES


library(dplyr)
library(tidyr)   
library(glmnet)   
library(sandwich) 
library(lmtest) 
library(tidyverse)
library(knitr)  
library(rstudioapi)

# Load E. coli data
my_data_ecoli <- read_csv("ecoli_cleaned2.csv") %>% as.data.frame() #has the ecoli abundance data, same applies for the ARG data

# Rename variables for clarity
my_data_ecoli2 <- my_data_ecoli %>%
  rename(
    living_below          = proportion_of_people_living_below_50percent_of_median_income,
    open_defecation       = people_practicing_open_defecation_of_population,
    at_least_basic_sani   = people_using_at_least_basic_sanitation_services_of_population,
    safely_managed_drin   = people_using_safely_managed_drinking_water_services_of_population,
    at_least_basic_drin   = people_using_at_least_basic_drinking_water_services_of_population,
    mortality_rate        = under_five_mortality_rate,
    access_to_electricity = access_to_electricity_of_population,
    capita_cu             = health_expenditure_per_capita_current_US.,
    daily_dose_antibio    = combined_ddd,
    medical_doctors       = medical_doctors_per10000,
    age_in_days           = age_days,
    expenditure_of_gd     = current_health_expenditure_of_gdp,
    safely_managed_sani   = people_using_safely_managed_sanitation_services_of_population,
    gini_index            = gini_index)

# Filter to age group <3 months (adjust as needed)
my_data_ecoli2 <- my_data_ecoli2 %>% filter(age_group == "<3") %>% as.data.frame()

# Create cluster variable and drop missing data
my_data_ecoli <- my_data_ecoli2 %>%
  mutate(cluster = as.integer(factor(country))) %>%
  drop_na(e_coli, daily_dose_antibio, medical_doctors, age_in_days,
          expenditure_of_gd, safely_managed_sani, gini_index,
          living_below, open_defecation, at_least_basic_sani,
          safely_managed_drin, at_least_basic_drin, mortality_rate,
          access_to_electricity, capita_cu, income_group, cluster) %>%
  mutate(
    income_group = factor(income_group),
    country = factor(country))

# Log-transform outcome
my_data_ecoli <- my_data_ecoli %>%
  mutate(log_ecoli = log10(e_coli + 1))

# Standardize variables (z-score)
vars_of_interest <- c(
  "daily_dose_antibio", "medical_doctors", "age_in_days", 
  "expenditure_of_gd", "safely_managed_sani", "gini_index",
  "capita_cu", "mortality_rate", "open_defecation", 
  "at_least_basic_sani", "safely_managed_drin", 
  "at_least_basic_drin", "access_to_electricity", 
  "living_below", "c_section")

my_data_ecoli <- my_data_ecoli %>%
  mutate(across(
    all_of(vars_of_interest),
    ~ (. - mean(.)) / sd(.),
    .names = "std_{.col}"))

# Define outcome
y_var_ecoli <- "log_ecoli"

# Define D_vars (variables of interest - want unbiased estimates)
D_vars_ecoli <- c(
  "std_daily_dose_antibio",
  "std_medical_doctors",
  "std_gini_index",
  "c_section")

# Define X_vars (confounders to partial out using LASSO)
X_vars_ecoli <- c(
  "std_age_in_days",
  "income_group",
  "cluster")

cluster_var <- "cluster"

# Build formula for X
X_formula_ecoli <- as.formula(paste0("~ ", paste(X_vars_ecoli, collapse = " + "), " - 1"))

# Cross-fitting (K=10)
set.seed(1234)
n_ecoli <- nrow(my_data_ecoli)
K <- 10
fold_id_ecoli <- sample(rep(1:K, length.out = n_ecoli))

# Placeholders for residuals
tildeY_ecoli <- rep(NA, n_ecoli)
tildeD_ecoli <- matrix(NA, nrow = n_ecoli, ncol = length(D_vars_ecoli))
colnames(tildeD_ecoli) <- D_vars_ecoli

cat("Running cross-fitting for E. coli...\n")

for (k in 1:K) {
  train_idx <- which(fold_id_ecoli != k)
  hold_idx <- which(fold_id_ecoli == k)

  # Design matrix
  X_train <- model.matrix(X_formula_ecoli, data = my_data_ecoli[train_idx, ])
  Y_train <- as.numeric(my_data_ecoli[train_idx, y_var_ecoli, drop = TRUE])

  # Partial out Y ~ X
  fit_y <- cv.glmnet(x = X_train, y = Y_train, alpha = 1)
  X_hold <- model.matrix(X_formula_ecoli, data = my_data_ecoli[hold_idx, ])
  Y_hat_hold <- as.vector(predict(fit_y, newx = X_hold, s = "lambda.min"))
  Y_obs_hold <- as.numeric(my_data_ecoli[hold_idx, y_var_ecoli, drop = TRUE])
  tildeY_ecoli[hold_idx] <- Y_obs_hold - Y_hat_hold

  # Partial out each D_j ~ X
  for (j in seq_along(D_vars_ecoli)) {
    Dj_train <- as.numeric(my_data_ecoli[train_idx, D_vars_ecoli[j], drop = TRUE])
    fit_dj <- cv.glmnet(x = X_train, y = Dj_train, alpha = 1)
    Dj_hat_hold <- as.vector(predict(fit_dj, newx = X_hold, s = "lambda.min"))
    D_obs_hold <- as.numeric(my_data_ecoli[hold_idx, D_vars_ecoli[j], drop = TRUE])
    tildeD_ecoli[hold_idx, j] <- D_obs_hold - Dj_hat_hold}}

# Final regression with cluster-robust SE
residual_data_ecoli <- data.frame(
  tildeY = tildeY_ecoli,
  tildeD_ecoli,
  cluster = my_data_ecoli[[cluster_var]])

final_formula_ecoli <- as.formula(paste("tildeY ~", paste(D_vars_ecoli, collapse = " + ")))
final_lm_ecoli <- lm(final_formula_ecoli, data = residual_data_ecoli)

# Cluster-robust variance
vcov_cluster_ecoli <- vcovCL(final_lm_ecoli, cluster = ~ cluster, type = "HC1")

# Extract results
ctest_ecoli <- coeftest(final_lm_ecoli, vcov = vcov_cluster_ecoli)
cint_ecoli <- coefci(final_lm_ecoli, vcov = vcov_cluster_ecoli, level = 0.95)

res_table_ecoli <- cbind(
  Estimate   = ctest_ecoli[, 1],
  Std.Error  = ctest_ecoli[, 2],
  t_value    = ctest_ecoli[, 3],
  p_value    = ctest_ecoli[, 4],
  CI_Lower   = cint_ecoli[, 1],
  CI_Upper   = cint_ecoli[, 2])

# Format and clean
res_table_ecoli_formatted <- as.data.frame(res_table_ecoli) %>%
  filter(row.names(.) != "(Intercept)") %>%
  mutate(
    variable = recode(row.names(.),
                     "std_daily_dose_antibio" = "DDD",
                     "std_medical_doctors" = "Medical doctors",
                     "std_gini_index" = "Gini index",
                     "c_section" = "C-section"),
    outcome = "E. coli",
    coefficient = round(Estimate, 6),
    lower_ci = round(CI_Lower, 6),
    upper_ci = round(CI_Upper, 6),
    p_value = round(p_value, 4)) %>%
  select(variable, outcome, coefficient, lower_ci, upper_ci, p_value)

cat("\n=== E. coli Results ===\n")
print(res_table_ecoli_formatted)

######################################################################
#                LOAD AND PREPARE ARG DATA                        #


cat("\n=== Processing ARG Model ===\n")

# Load ARG data
my_data_arg <- read.csv("updated_amr_metadata.csv", header = TRUE, sep = ",", 
                        stringsAsFactors = FALSE)

# Aggregate ARG abundance per biosample
amr_aggregated <- my_data_arg %>%
  group_by(biosample, aro_term) %>%
  summarise(total_rpkm = sum(rpkm), .groups = 'drop') %>%
  group_by(biosample) %>%
  summarise(total_rpkm = sum(total_rpkm, na.rm = TRUE), .groups = 'drop')

# Get metadata
amr_metadata <- my_data_arg %>%
  select(biosample, country, age_days = age_months, 
         daily_dose_antibio = combined_ddd,
         medical_doctors = who_medical_doctors_per10000_national,
         gini_index = gini_index_national,
         c_section, income_group = income_group, age_group) %>%
  distinct(biosample, .keep_all = TRUE)

# Join
model_data_arg <- left_join(amr_aggregated, amr_metadata, by = "biosample")

# Filter to age <3 (adjust as needed)
model_data_arg <- model_data_arg %>%
  filter(age_group == "<3") %>%
  drop_na(daily_dose_antibio, medical_doctors, gini_index, c_section, age_days)

# Create cluster
model_data_arg <- model_data_arg %>%
  mutate(
    cluster = as.integer(factor(country)),
    income_group = factor(income_group),
    country = factor(country))

# Log-transform outcome
model_data_arg <- model_data_arg %>%
  mutate(log_rpkm = log10(total_rpkm + 1))

# Standardize
vars_to_std_arg <- c("daily_dose_antibio", "medical_doctors", "gini_index", 
                     "c_section", "age_days")

model_data_arg <- model_data_arg %>%
  mutate(across(
    all_of(vars_to_std_arg),
    ~ (. - mean(., na.rm = TRUE)) / sd(., na.rm = TRUE),
    .names = "std_{.col}"))

# Define outcome
y_var_arg <- "log_rpkm"

# Define D_vars
D_vars_arg <- c(
  "std_daily_dose_antibio",
  "std_medical_doctors",
  "std_gini_index",
  "std_c_section")

# Define X_vars
X_vars_arg <- c(
  "std_age_days",
  "income_group")

# Build formula
X_formula_arg <- as.formula(paste0("~ ", paste(X_vars_arg, collapse = " + "), " - 1"))

# Cross-fitting
set.seed(1234)
n_arg <- nrow(model_data_arg)
fold_id_arg <- sample(rep(1:K, length.out = n_arg))

# Placeholders
tildeY_arg <- rep(NA, n_arg)
tildeD_arg <- matrix(NA, nrow = n_arg, ncol = length(D_vars_arg))
colnames(tildeD_arg) <- D_vars_arg

cat("Running cross-fitting for ARG...\n")

for (k in 1:K) {
  train_idx <- which(fold_id_arg != k)
  hold_idx <- which(fold_id_arg == k)

  X_train <- model.matrix(X_formula_arg, data = model_data_arg[train_idx, ])
  Y_train <- as.numeric(model_data_arg[train_idx, y_var_arg, drop = TRUE])

  # Partial out Y ~ X
  fit_y <- cv.glmnet(x = X_train, y = Y_train, alpha = 1)
  X_hold <- model.matrix(X_formula_arg, data = model_data_arg[hold_idx, ])
  Y_hat_hold <- as.vector(predict(fit_y, newx = X_hold, s = "lambda.min"))
  Y_obs_hold <- as.numeric(model_data_arg[hold_idx, y_var_arg, drop = TRUE])
  tildeY_arg[hold_idx] <- Y_obs_hold - Y_hat_hold

  # Partial out each D_j ~ X
  for (j in seq_along(D_vars_arg)) {
    Dj_train <- as.numeric(model_data_arg[train_idx, D_vars_arg[j], drop = TRUE])
    fit_dj <- cv.glmnet(x = X_train, y = Dj_train, alpha = 1)
    Dj_hat_hold <- as.vector(predict(fit_dj, newx = X_hold, s = "lambda.min"))
    D_obs_hold <- as.numeric(model_data_arg[hold_idx, D_vars_arg[j], drop = TRUE])
    tildeD_arg[hold_idx, j] <- D_obs_hold - Dj_hat_hold}}

# Final regression
residual_data_arg <- data.frame(
  tildeY = tildeY_arg,
  tildeD_arg,
  cluster = model_data_arg$cluster)

final_formula_arg <- as.formula(paste("tildeY ~", paste(D_vars_arg, collapse = " + ")))
final_lm_arg <- lm(final_formula_arg, data = residual_data_arg)

# Cluster-robust SE
vcov_cluster_arg <- vcovCL(final_lm_arg, cluster = ~ cluster, type = "HC1")

# Extract results
ctest_arg <- coeftest(final_lm_arg, vcov = vcov_cluster_arg)
cint_arg <- coefci(final_lm_arg, vcov = vcov_cluster_arg, level = 0.95)

res_table_arg <- cbind(
  Estimate   = ctest_arg[, 1],
  Std.Error  = ctest_arg[, 2],
  t_value    = ctest_arg[, 3],
  p_value    = ctest_arg[, 4],
  CI_Lower   = cint_arg[, 1],
  CI_Upper   = cint_arg[, 2])

# Format
res_table_arg_formatted <- as.data.frame(res_table_arg) %>%
  filter(row.names(.) != "(Intercept)") %>%
  mutate(
    variable = recode(row.names(.),
                     "std_daily_dose_antibio" = "DDD",
                     "std_medical_doctors" = "Medical doctors",
                     "std_gini_index" = "Gini index",
                     "std_c_section" = "C-section"),
    outcome = "ARG",
    coefficient = round(Estimate, 6),
    lower_ci = round(CI_Lower, 6),
    upper_ci = round(CI_Upper, 6),
    p_value = round(p_value, 4)) %>%
  select(variable, outcome, coefficient, lower_ci, upper_ci, p_value)

cat("\n=== ARG Results ===\n")
print(res_table_arg_formatted)

# Combine E. coli and ARG results
combined_results <- bind_rows(res_table_ecoli_formatted, res_table_arg_formatted)

# Save to CSV
write.csv(combined_results, "plotting_model.csv", row.names = FALSE)
cat("\n✓ Combined results saved to: plotting_model.csv\n")

# Display in viewer
html_table <- kable(combined_results, format = "html")
tmp <- tempfile(fileext = ".html")
writeLines(html_table, tmp)
viewer(tmp)

######################################################
#Visualisation

library(ggplot2)
library(ggpubr)
library(scales)

# Add significance flag
combined_results <- combined_results %>%
  mutate(significance = ifelse(p_value < 0.05, "Significant", "Not significant"))

# Set factor levels
combined_results$variable <- factor(
  combined_results$variable,
  levels = rev(c("DDD", "Medical doctors", "Gini index", "C-section", "Age")))

combined_results$outcome <- factor(
  combined_results$outcome,
  levels = c("ARG", "E. coli"),
  labels = c("ARG Abundance", expression(paste(italic("E. coli"), " Abundance"))))

# Create forest plot
forest_plot <- ggplot(combined_results, aes(x = coefficient, y = variable)) +
  geom_errorbarh(
    aes(xmin = lower_ci, xmax = upper_ci, color = significance),
    height = 0.1, 
    size = 0.5) +
  geom_point(aes(color = significance), size = 2.0) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
  facet_wrap(~ outcome, ncol = 2) +
  scale_x_continuous(
    name = "Difference in mean log abundance (95% CI)",
    breaks = pretty_breaks(n = 5)) +
  labs(y = NULL, color = "Significance") +
  scale_color_manual(
    values = c("Significant" = "brown", "Not significant" = "black")) +
  theme_pubclean(base_size = 12) +
  theme(
    panel.grid.major = element_line(color = "gray90", linetype = "dotted"),
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "plain", size = 12),
    axis.text = element_text(size = 11),
    axis.title = element_text(size = 12, face = "plain"),
    legend.position = "bottom",
    panel.spacing = unit(0.5, "lines"))

print(forest_plot)

