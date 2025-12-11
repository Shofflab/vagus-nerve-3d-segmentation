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
    panel.background = element_blank(),
    plot.background = element_blank(),
    plot.title = element_text(
      size = 10,
      color = "black",
      face = "italic",
      hjust = 0,
      vjust = 0
    )
  )

# set output directories
fig_output_dir <- here("figures")
if (!dir.exists(fig_output_dir)) {
  dir.create(fig_output_dir, recursive = TRUE)
}

# set label names
global_labller <- labeller(
  cls = c("fas" = "Fascicles", "epi" = "Epineurium"),
  model = c("2D" = "2D", "3D" = "3D"),
  error_type = c(
    "over_seg" = "Over-segmentation",
    "under_seg" = "Under-segmentation",
    "fp_rate" = "False positives"
  ),
)


#### Helpers ####
normalize_model <- function(df) {
  if (!"model" %in% names(df)) {
    return(df)
  }

  df %>%
    mutate(model = recode(as.character(model), "2" = "2D", "3" = "3D"))
}


#### Load Data ####
fas_f1_df <- read_csv(
  here("results", "Fig4_fascicle_detection_f1.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()

missed_fas_df <- read_csv(
  here("results", "Fig4_fascicle_detection_missed.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()

object_error_df <- read_csv(
  here("results", "Fig4_fascicle_detection_object_error.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()

added_fas_df <- read_csv(
  here("results", "Fig4_fascicle_detection_fp.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()


#### Stats ####

# Fascicle F1 score
fas_f1_stat <- fas_f1_df %>%
  group_by(th) %>%
  wilcox_test(f1 ~ model, detailed = TRUE) %>%
  adjust_pvalue() %>%
  add_significance("p.adj") %>%
  add_y_position(fun = "mean_ci") %>%
  mutate(xmin = th, xmax = th)

# Fascicle missed rate - complete the dataset by filling in zeros for missing combinations
missed_fas_df_complete <- missed_fas_df %>%
  ungroup() %>%
  complete(
    id,
    model,
    area_category,
    fold,
    fill = list(n_missed = 0, missed_rate = 0)
  )

missed_fas_stat <- missed_fas_df_complete %>%
  group_by(area_category) %>%
  wilcox_test(missed_rate ~ model, detailed = TRUE, paired = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  add_xy_position(fun = "mean_ci", scales = "fixed") %>%
  mutate(y.position = max(y.position))

# Object-level error (over and under segmentation)
object_error_stat <- object_error_df %>%
  group_by(error_type) %>%
  wilcox_test(rate ~ model, detailed = TRUE, paired = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  add_xy_position(fun = "mean_ci", scales = "fixed") %>%
  mutate(y.position = max(y.position))

# False positives
added_fas_stat <- added_fas_df %>%
  wilcox_test(fp_rate ~ model, detailed = TRUE, paired = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  add_xy_position(fun = "mean_ci", scales = "fixed") %>%
  mutate(y.position = max(y.position))


#### Plot ####

#### Fascicle F1 Plot ####

fas_f1_plot <- fas_f1_df %>%
  ggplot(aes(x = th, y = f1, color = model)) +
  stat_summary(
    geom = "errorbar",
    fun.data = mean_cl_boot,
    width = 0.015,
    linewidth = 0.75
  ) +
  stat_summary(
    geom = "line",
    fun = mean,
    linewidth = 1
  ) +
  stat_summary(
    geom = "point",
    fun = mean,
    size = 2
  ) +
  stat_pvalue_manual(
    data = fas_f1_stat,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 1,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = TRUE,
    family = "Arial",
  ) +
  labs(
    x = "IoU threshold",
    y = "F1 score",
    title = "Fascicle detection in cross sections",
    color = "Network",
    fill = "Network"
  ) +
  my_theme +
  theme(
    legend.position = "top",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 10),
  ) +
  theme(strip.placement = "outside") +
  theme(panel.spacing = unit(0, "lines")) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_fill_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(breaks = breaks_pretty(5))


#### False Negatives by Area Plot ####

missed_fas_df_by_fold <- missed_fas_df_complete %>%
  group_by(area_category, model, fold) %>%
  summarise(missed_rate = mean(missed_rate), .groups = "drop") %>%
  pivot_wider(names_from = model, values_from = missed_rate) %>%
  pivot_longer(
    cols = c("3D", "2D"),
    names_to = "model",
    values_to = "missed_rate"
  ) %>%
  mutate(model = factor(model, levels = c("3D", "2D")))

missed_fas_plot <- missed_fas_df_complete %>%
  ggplot(aes(x = model, y = missed_rate, color = model)) +
  geom_point(
    data = missed_fas_df_by_fold,
    aes(group = interaction(area_category, model)),
    size = 2,
    color = "darkgrey",
    alpha = 0.5,
    position = position_dodge(width = 0.5)
  ) +
  geom_line(
    data = missed_fas_df_by_fold,
    aes(group = interaction(area_category, fold, lex.order = TRUE)),
    color = "darkgrey",
    linewidth = 0.75,
    linetype = "dashed",
    alpha = 0.5
  ) +
  stat_summary(
    geom = "errorbar",
    fun.data = mean_cl_boot,
    width = 0.25,
    linewidth = 0.75,
    position = position_dodge(width = 0.5)
  ) +
  stat_summary(
    geom = "errorbar",
    fun = mean,
    aes(ymax = after_stat(y), ymin = after_stat(y)),
    linewidth = 1,
    width = 0.35,
    position = position_dodge(width = 0.5)
  ) +
  stat_pvalue_manual(
    data = missed_fas_stat,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 1,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  facet_wrap(
    ~area_category,
    ncol = 4,
    labeller = global_labller,
    strip.position = "top"
  ) +
  labs(
    x = "Network",
    y = "Miss rate (%)",
    title = "Missed fascicles by area"
  ) +
  my_theme +
  theme(
    legend.position = "bottom",
    panel.spacing = unit(0, "lines"),
  ) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(breaks = breaks_pretty(5))


#### Object-level error plot ####

object_error_df_by_fold <- object_error_df %>%
  group_by(error_type, model, fold, id) %>%
  summarise(rate = mean(rate, na.rm = TRUE), .groups = "drop") %>%
  group_by(error_type, model, fold) %>%
  summarise(rate = mean(rate, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = model, values_from = rate) %>%
  pivot_longer(cols = c("3D", "2D"), names_to = "model", values_to = "rate") %>%
  mutate(model = factor(model, levels = c("3D", "2D")))

# Use fold data to update the location of significance values
object_error_stat <- object_error_stat %>%
  mutate(y.position = max(object_error_df_by_fold$rate))

object_error_plot <- object_error_df %>%
  ggplot(aes(x = model, y = rate, color = model)) +
  geom_point(
    data = object_error_df_by_fold,
    aes(group = interaction(error_type, model)),
    size = 2,
    color = "darkgrey",
    alpha = 0.5,
  ) +
  geom_line(
    data = object_error_df_by_fold,
    aes(group = interaction(error_type, fold, lex.order = TRUE)),
    color = "darkgrey",
    linewidth = 0.75,
    linetype = "dashed",
    alpha = 0.5,
  ) +
  stat_summary(
    geom = "errorbar",
    fun.data = mean_ci,
    width = 0.25,
    linewidth = 0.75,
  ) +
  stat_summary(
    geom = "errorbar",
    fun = mean,
    aes(ymax = after_stat(y), ymin = after_stat(y)),
    linewidth = 1,
    width = 0.35,
  ) +
  stat_pvalue_manual(
    data = object_error_stat,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 1,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  facet_wrap(~error_type, strip.position = "top", labeller = global_labller) +
  labs(
    x = "Network",
    y = "Error rate (%)",
    title = "Fascicle-level segmentation error",
  ) +
  my_theme +
  theme(
    legend.position = "none",
    panel.spacing = unit(0, "lines"),
  ) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(breaks = breaks_pretty(5))


#### Plot Assembly ####

# use patchwork to assemble the plots
dummy <- ggplot() +
  geom_blank() +
  my_theme
top_part <- dummy

area_cat_def <- dummy +
  labs(title = "Fascicle area categories") +
  theme(
    plot.title = element_text(
      size = 10,
      color = "black",
      face = "italic",
      hjust = 0,
      vjust = 0
    )
  )

middle_part <- fas_f1_plot +
  plot_spacer() +
  object_error_plot +
  plot_layout(guides = "keep", widths = c(1, 0, 1))

bottom_part <- area_cat_def +
  plot_spacer() +
  missed_fas_plot +
  plot_layout(guides = "keep", widths = c(0.3, 0.1, 1)) &
  theme(legend.position = "none")

assembly <- top_part /
  middle_part /
  bottom_part +
  plot_layout(heights = c(1, 0.5, 0.5))

assembly <- assembly +
  plot_annotation(tag_levels = c("a")) &
  theme(
    plot.tag = element_text(face = "bold", size = 14, hjust = 0.5, vjust = 1),
    plot.tag.position = c(0, 1)
  )


#### Export ####

ggsave(
  file.path(fig_output_dir, "Fig4_fascicle_detection.svg"),
  assembly,
  width = 7,
  height = 7,
  units = "in",
  dpi = 300,
)
