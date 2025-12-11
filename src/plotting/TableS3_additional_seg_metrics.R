#### Frontmatter ####

# import libraries
library(ggplot2)
library(rmarkdown)
library(tidyverse)
library(readr)
library(here)
library(ggthemes)
library(scales)
library(ggpubr)
library(rstatix)
library(patchwork)
library(gt)
library(gtsummary)

# set theme
my_theme <-
  theme_bw(base_size = 10, base_family = "Arial") +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(
      size = 10,
      hjust = 0.5,
      vjust = 0.5
    ),
    axis.text = element_text(size = 10, color = "black"),
    axis.ticks = element_line(color = "black", linewidth = 0.25),
    plot.title = element_text(
      size = 10,
      color = "black",
      face = "italic",
      hjust = 0,
      vjust = 0
    )
  )


#### Helpers ####
normalize_model <- function(df, factor_levels = NULL) {
  if (!"model" %in% names(df)) {
    return(df)
  }

  df <- df %>%
    mutate(model = recode(as.character(model), "2" = "2D", "3" = "3D"))

  if (!is.null(factor_levels)) {
    df <- df %>%
      mutate(model = factor(model, levels = factor_levels))
  }

  df
}


#### Load Data ####
conf_mat <- read_csv(
  here("results", "TableS3_additional_seg_metrics.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()


#### Generate Tables ####

fas_sum_tbl <- conf_mat %>%
  filter(class == "fas") %>%
  select(id, model, class, IoU, Sensitivity, Specificity) %>%
  group_by(id, model, class) %>%
  mutate(unique_id = paste(id, model, class, sep = "_")) %>%
  tbl_summary(
    by = model,
    statistic = list(all_continuous() ~ "{mean}"),
    digits = list(all_continuous() ~ 4),
    missing = "no",
    include = c(IoU, Sensitivity, Specificity)
  ) %>%
  add_difference(
    test = list(
      all_continuous() ~ "wilcox.test"
    ),
  ) %>%
  modify_table_body(
    ~ .x %>% select(-p.value)
  ) %>%
  add_p(
    test = list(
      all_continuous() ~ "paired.wilcox.test"
    ),
    group = id
  ) %>%
  add_q(method = "bonferroni") %>%
  modify_column_hide(columns = p.value) %>%
  modify_header(label ~ "**Metric**") %>%
  modify_header(all_stat_cols() ~ "**{level}**, N = {n}") %>%
  modify_header(q.value ~ "***p*-value**") %>%
  modify_spanning_header(all_stat_cols() ~ "**Network**")

epi_sum_tbl <- conf_mat %>%
  filter(class == "epi") %>%
  select(id, model, class, IoU, Sensitivity, Specificity) %>%
  group_by(id, model, class) %>%
  mutate(unique_id = paste(id, model, class, sep = "_")) %>%
  tbl_summary(
    by = model,
    statistic = list(all_continuous() ~ "{mean}"),
    digits = list(all_continuous() ~ 4),
    missing = "no",
    include = c(IoU, Sensitivity, Specificity)
  ) %>%
  add_difference(
    test = list(
      all_continuous() ~ "wilcox.test"
    ),
  ) %>%
  modify_table_body(
    ~ .x %>% select(-p.value)
  ) %>%
  add_p(
    test = list(
      all_continuous() ~ "paired.wilcox.test"
    ),
    group = id
  ) %>%
  add_q(method = "bonferroni") %>%
  modify_column_hide(columns = p.value) %>%
  modify_header(label ~ "**Metric**") %>%
  modify_header(all_stat_cols() ~ "**{level}**, N = {n}") %>%
  modify_header(q.value ~ "***p*-value**") %>%
  modify_spanning_header(all_stat_cols() ~ "**Network**")

merged_tbl <- tbl_stack(
  list(fas_sum_tbl, epi_sum_tbl),
  group_header = c("Fascicles", "Epineurium")
) %>%
  italicize_labels() %>%
  as_gt() %>%
  opt_footnote_marks(marks = "letters") %>%
  gtsave(here("tables", "TableS3_additional_seg_metrics.docx"))
