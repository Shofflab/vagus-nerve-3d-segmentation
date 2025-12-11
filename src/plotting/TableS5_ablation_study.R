#### Frontmatter ####

# import libraries
library(ggplot2)
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

# set label names
global_labller <- labeller(
  cls = c("fas" = "Fascicles", "epi" = "Epineurium"),
  model = c("2D" = "2D", "3D" = "3D"),
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
cldice_df <- read_csv(
  here("results", "TableS5_ablation_cldice.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()

topo_error_df <- read_csv(
  here("results", "TableS5_ablation_topo_error.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()


#### Generate Table ####

ablation_tbl <- bind_rows(cldice_df, topo_error_df) %>%
  select(model, id, metric, value) %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  tbl_summary(
    by = model,
    statistic = all_continuous() ~ "{mean}",
    include = c(cl_dice, error_rate),
    digits = all_continuous() ~ 2,
    label = list(cl_dice = "clDice", error_rate = "Error rate (%)")
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
  modify_spanning_header(all_stat_cols() ~ "**Network**") %>%
  as_gt() %>%
  fmt_number(
    columns = all_stat_cols(),
    decimals = 2
  ) %>%
  opt_footnote_marks(marks = "letters") 

#### Export ####

ablation_tbl %>%
  gtsave(here("tables", "TableS5_ablation_study.docx"))