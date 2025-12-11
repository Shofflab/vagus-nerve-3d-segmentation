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
library(ggpmisc)

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

# set output directories
fig_output_dir <- here("figures")
if (!dir.exists(fig_output_dir)) {
  dir.create(fig_output_dir, recursive = TRUE)
}

# set label names
global_labller <- labeller(
  cls = c("fas" = "Fascicles", "epi" = "Epineurium"),
  model = c("2D" = "2D", "3D" = "3D"),
  metric = c(
    "fas_anatomy_error_rate" = "Fascicle topology",
    "structure_error_rate" = "Structural integrity"
  ),
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
jitter_df <- read_csv(
  here("results", "Fig6_quality_jitter.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()

split_merge_df <- read_csv(
  here("results", "Fig6_quality_split_merge.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()


#### Stats ####

jitter_stat_df <- jitter_df %>%
  group_by(cls) %>%
  wilcox_test(bfscore ~ model, paired = TRUE, detailed = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  add_xy_position(fun = "mean_ci")

split_merge_stat_df <- split_merge_df %>%
  wilcox_test(diff ~ model, paired = TRUE, detailed = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  add_xy_position(scales = "free_y", fun = "mean_ci")


#### Plotting ####

#### Jittering (All subjects) ####

jitter_by_fold <- jitter_df %>%
  group_by(model, fold, cls) %>%
  summarise(bfscore = mean(bfscore), .groups = "drop")

jitter_stat_df <- jitter_stat_df %>%
  mutate(y.position = y.position + 0.05)

jitter_all_plot <- jitter_df %>%
  ggplot(aes(x = model, y = bfscore, color = model)) +
  geom_point(
    data = jitter_by_fold,
    aes(y = bfscore),
    size = 2,
    alpha = 0.5,
    color = "darkgrey"
  ) +
  geom_line(
    data = jitter_by_fold %>% filter(model != "GT"),
    aes(y = bfscore, group = fold),
    color = "darkgrey",
    linewidth = 0.5,
    linetype = "dashed",
    alpha = 0.75,
  ) +
  stat_summary(
    fun.data = mean_cl_boot,
    geom = "errorbar",
    width = 0.2,
    linewidth = 0.75,
  ) +
  stat_summary(
    fun = mean,
    geom = "errorbar",
    aes(ymax = after_stat(y), ymin = after_stat(y)),
    linewidth = 1,
    width = 0.3
  ) +
  stat_pvalue_manual(
    jitter_stat_df,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 0.5,
    hide.ns = FALSE,
    size = 4,
    family = "Arial",
  ) +
  facet_wrap(~cls, ncol = 2, labeller = global_labller) +
  labs(
    x = "Network",
    y = "BF score",
    title = "Inter-slice jittering"
  ) +
  my_theme +
  theme(legend.position = "none") +
  theme(
    strip.placement = "inside",
    panel.spacing = unit(0, "lines")
  ) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(
    expand = expansion(mult = c(0.05, 0.1))
  )


#### Split Merge Plot ####

split_merge_by_fold <- split_merge_df %>%
  group_by(model, fold) %>%
  summarise(diff = mean(diff), .groups = "drop")

split_merge_stat_df <- split_merge_stat_df %>%
  mutate(y.position = max(split_merge_by_fold$diff) + 5)

split_merge_all_plot <- split_merge_df %>%
  ggplot(aes(x = model, y = diff, color = model)) +
  geom_point(
    data = split_merge_by_fold,
    aes(y = diff),
    size = 2,
    alpha = 0.5,
    color = "darkgrey"
  ) +
  geom_line(
    data = split_merge_by_fold %>% filter(model != "GT"),
    aes(y = diff, group = fold),
    color = "darkgrey",
    linewidth = 0.5,
    linetype = "dashed",
    alpha = 0.75,
  ) +
  stat_summary(
    fun.data = mean_cl_boot,
    geom = "errorbar",
    width = 0.2,
    linewidth = 0.75,
  ) +
  stat_summary(
    fun = mean,
    geom = "errorbar",
    aes(ymax = after_stat(y), ymin = after_stat(y)),
    linewidth = 1,
    width = 0.3
  ) +
  stat_pvalue_manual(
    split_merge_stat_df,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 0.5,
    size = 4,
    family = "Arial",
    label.size = 4,
    remove.bracket = FALSE,
    bracket.size = 0,
  ) +
  labs(
    x = "Network",
    y = "Deviation of event\nfrequency (%)",
    title = "Fascicle split/merge events"
  ) +
  my_theme +
  theme(legend.position = "none") +
  theme(
    strip.text = element_blank(),
    strip.background = element_blank()
  ) +
  theme(panel.spacing = unit(0, "lines")) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.1)))


#### Plot Assembly ####

dummy <- ggplot() +
  geom_blank() +
  my_theme

overall_stat_row <- jitter_all_plot +
  plot_spacer() +
  split_merge_all_plot +
  plot_layout(widths = c(1.8, 0.1, 1))

demo_row <- dummy +
  dummy +
  dummy +
  plot_layout(
    widths = c(2, 1.5, 1),
    axis_titles = "collect",
    guides = "collect"
  ) &
  theme(legend.position = "bottom")

assembly <- overall_stat_row /
  demo_row +
  plot_layout(heights = c(1, 1)) +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 14,
      hjust = 1,
      vjust = 1
    ),
    plot.tag.position = c(0, 1),
  )


#### Export ####

ggsave(
  file.path(fig_output_dir, "Fig6_quality_for_analysis.svg"),
  assembly,
  width = 8,
  height = 5,
  units = "in",
  dpi = 150,
)
