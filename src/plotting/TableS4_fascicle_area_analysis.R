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
library(janitor)
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
pair_seg_metric_df <- read_csv(
  here("results", "TableS4_fascicle_area_metrics.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()


#### Stats ####

pair_seg_metric_fold_df <- pair_seg_metric_df %>%
  group_by(model, fold, metric, area_category) %>%
  summarise(
    value = mean(value),
    .groups = "drop"
  )

pair_iou_stat_df <- pair_seg_metric_df %>%
  filter(metric == "pair_iou") %>%
  group_by(area_category) %>%
  rstatix::wilcox_test(value ~ model, detailed = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance() %>%
  add_xy_position(x = "model", fun = "mean_ci") %>%
  mutate(
    y.position = max(
      pair_seg_metric_fold_df %>%
        filter(metric == "pair_iou") %>%
        pull(value)
    )
  )

pair_hd_stat_df <- pair_seg_metric_df %>%
  filter(metric == "hd_um") %>%
  group_by(area_category) %>%
  rstatix::wilcox_test(value ~ model, detailed = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance() %>%
  add_xy_position(x = "model", fun = "mean_ci")

area_diff_stat_df <- pair_seg_metric_df %>%
  filter(metric == "area_diff_percent") %>%
  group_by(area_category) %>%
  rstatix::wilcox_test(value ~ model, detailed = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance() %>%
  add_xy_position(x = "model", fun = "mean_ci") %>%
  mutate(
    y.position = max(
      pair_seg_metric_fold_df %>%
        filter(metric == "area_diff_percent") %>%
        pull(value)
    )
  )


#### Plots ####

pair_iou_plot <- pair_seg_metric_df %>%
  filter(metric == "pair_iou") %>%
  ggplot(aes(x = model, y = value, color = model)) +
  stat_summary(
    fun.data = mean_ci,
    geom = "errorbar",
    width = 0.2,
    linewidth = 0.75
  ) +
  stat_summary(
    fun = mean,
    geom = "errorbar",
    aes(ymax = after_stat(y), ymin = after_stat(y)),
    linewidth = 1,
    width = 0.3
  ) +
  stat_pvalue_manual(
    data = pair_iou_stat_df,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = -0.5,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  labs(
    x = "Fascicle area (μm²)",
    y = "Pairwise IoU"
  ) +
  facet_grid(~area_category) +
  my_theme +
  theme(legend.position = "none") +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(
    breaks = breaks_pretty(5),
    expand = expansion(mult = c(0.05, 0.1))
  )

pair_hd_plot <- pair_seg_metric_df %>%
  filter(metric == "hd_um") %>%
  ggplot(aes(x = model, y = value, color = model)) +
  stat_summary(
    fun.data = mean_ci,
    geom = "errorbar",
    width = 0.2,
    linewidth = 0.75
  ) +
  stat_summary(
    fun = mean,
    geom = "errorbar",
    aes(ymax = after_stat(y), ymin = after_stat(y)),
    linewidth = 1,
    width = 0.3
  ) +
  stat_pvalue_manual(
    data = pair_hd_stat_df,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = -0.5,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  labs(
    x = "Fascicle area (μm²)",
    y = "Pairwise Hausdorff distance (μm)"
  ) +
  facet_grid(~area_category) +
  my_theme +
  theme(legend.position = "none") +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(
    breaks = breaks_pretty(5),
    expand = expansion(mult = c(0.05, 0.1))
  )

area_diff_plot <- pair_seg_metric_df %>%
  filter(metric == "area_diff_percent") %>%
  ggplot(aes(x = model, y = value, color = model)) +
  geom_point(
    data = pair_seg_metric_fold_df %>%
      filter(metric == "area_diff_percent"),
    aes(y = value, group = fold),
    size = 2,
    color = "darkgrey",
    alpha = 0.75,
  ) +
  geom_line(
    data = pair_seg_metric_fold_df %>%
      filter(metric == "area_diff_percent"),
    aes(group = fold),
    color = "darkgrey",
    linewidth = 0.5,
    linetype = "dashed",
    alpha = 0.75,
  ) +
  stat_summary(
    fun.data = mean_ci,
    geom = "errorbar",
    width = 0.2,
    linewidth = 0.75
  ) +
  stat_summary(
    fun = mean,
    geom = "errorbar",
    aes(ymax = after_stat(y), ymin = after_stat(y)),
    linewidth = 1,
    width = 0.3
  ) +
  stat_pvalue_manual(
    data = area_diff_stat_df,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = -0.5,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  labs(
    x = "Fascicle area (μm²)",
    y = "Area difference (%)"
  ) +
  facet_grid(~area_category, scales = "free") +
  my_theme +
  theme(legend.position = "none") +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(
    breaks = breaks_pretty(5),
    expand = expansion(mult = c(0.05, 0.1))
  )


#### Generate Table ####

pair_seg_metric_summary <- pair_seg_metric_df %>%
  group_by(id, subject, fold, model, area_category, metric) %>%
  summarise(value = mean(value), .groups = "drop")

pair_seg_tbl <- pair_seg_metric_summary %>%
  pivot_wider(names_from = metric, values_from = value) %>%
  select(model, area_category, pair_iou, hd_um, area_diff_percent) %>%
  tbl_strata2(
    strata = area_category,
    .tbl_fun = ~ .x %>%
      tbl_summary(
        by = model,
        statistic = list(all_continuous() ~ "{mean}"),
        label = list(
          pair_iou = "IoU",
          hd_um = "HD (μm)",
          area_diff_percent = "Area difference (%)"
        ),
        digits = list(
          all_continuous() ~ 2
        )
      ) %>%
      add_difference(
        test = list(
          all_continuous() ~ "wilcox.test"
        ),
        estimate_fun = list(
          all_continuous() ~ label_style_number(digits = 2)
        )
      ) %>%
      add_q(
        method = "bonferroni",
        pvalue_fun = label_style_pvalue(digits = 2)
      ) %>%
      modify_header(),
    .combine_with = "tbl_stack",
    .header = "{strata}, N = {n}"
  ) %>%
  modify_column_hide(columns = p.value) %>%
  modify_header(
    label ~ "**Metric**",
    q.value ~ "***p*-value**",
    all_stat_cols() ~ "**{level}**"
  ) %>%
  modify_spanning_header(all_stat_cols() ~ "**Network**") %>%
  as_gt() %>%
  fmt_number(
    columns = all_stat_cols(),
    decimals = 1
  ) %>%
  opt_footnote_marks(marks = "letters") %>%
  tab_options(
    latex.use_longtable = TRUE
  )


#### Export ####

pair_seg_tbl %>%
  gtsave(here("tables", "TableS4_fascicle_area_metrics.docx"))
