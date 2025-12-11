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
)


#### Helpers ####
normalize_model <- function(df) {
  if (!"model" %in% names(df)) {
    return(df)
  }

  df %>%
    mutate(
      model = recode(as.character(model), "2" = "2D", "3" = "3D"),
      model = factor(model, levels = c("2D", "3D"))
    )
}


#### Load Data ####
cldice_df <- read_csv(
  here("results", "Fig5_anatomy_accuracy_cldice.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()

topo_error_df <- read_csv(
  here("results", "Fig5_anatomy_accuracy_topo_error.csv"),
  show_col_types = FALSE
) %>%
  normalize_model()


#### Stats ####

# clDice
cldice_by_fold <- cldice_df %>%
  group_by(metric, cls, model, fold) %>%
  summarise(value = mean(value)) %>%
  ungroup()

cldice_position_by_fold <- cldice_by_fold %>%
  group_by(metric) %>%
  summarise(y_max = max(value))

cldice_stat_df <- cldice_df %>%
  group_by(metric, cls) %>%
  wilcox_test(value ~ model, paired = TRUE, detailed = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  add_xy_position(scales = "free_y", fun = "mean_ci")

# Topological Error
topo_error_stat_df <- topo_error_df %>%
  group_by(metric) %>%
  wilcox_test(value ~ model, detailed = TRUE, paired = TRUE) %>%
  adjust_pvalue(method = "bonferroni") %>%
  add_significance("p.adj") %>%
  add_xy_position(scales = "free_y", fun = "mean_ci")


#### clDice Plot ####

cldice_plot <- cldice_df %>%
  filter(metric == "cl_dice") %>%
  ggplot(aes(x = model, y = value, color = model)) +
  geom_point(
    data = cldice_by_fold %>% filter(metric == "cl_dice"),
    aes(y = value),
    size = 2,
    color = "darkgrey",
    alpha = 0.75,
  ) +
  geom_line(
    data = cldice_by_fold %>% filter(metric == "cl_dice"),
    aes(group = fold),
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
  labs(x = "Network", y = "clDice", title = "Fascicle connectivity") +
  stat_pvalue_manual(
    data = cldice_stat_df %>% filter(metric == "cl_dice"),
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 1,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  my_theme +
  theme(legend.position = "none") +
  theme(strip.placement = "outside") +
  theme(panel.spacing = unit(0, "lines")) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(breaks = breaks_pretty(5))


#### Topological Error Plot ####

topo_error_by_fold <- topo_error_df %>%
  group_by(model, fold, metric) %>%
  summarise(value = mean(value)) %>%
  ungroup()

# For each metric, find the y.position by the max value of the fold average
y_pos <- topo_error_by_fold %>%
  group_by(metric) %>%
  summarise(y.position = max(value))

topo_error_stat_df <- topo_error_stat_df %>%
  mutate(y.position = y.position + 1.8)


topo_error_plot <- topo_error_by_fold %>%
  ggplot(aes(x = model, y = value, color = model)) +
  geom_point(
    data = topo_error_by_fold,
    aes(y = value),
    size = 2,
    alpha = 0.75,
    color = "darkgrey"
  ) +
  geom_line(
    data = topo_error_by_fold,
    aes(group = fold),
    color = "darkgrey",
    linewidth = 0.5,
    linetype = "dashed",
    alpha = 0.75
  ) +
  stat_summary(
    fun.data = mean_cl_boot,
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
  labs(
    x = "Network",
    y = "Error rate (%)",
    title = "Anatomical errors"
  ) +
  stat_pvalue_manual(
    topo_error_stat_df,
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 0.5,
    hide.ns = FALSE,
    size = 4,
    family = "Arial",
    bracket.size = 0,
    remove.bracket = FALSE,
  ) +
  my_theme +
  theme(
    legend.position = "none",
  ) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(
    breaks = breaks_pretty(5),
    expand = expansion(mult = c(0.05, 0.1))
  )


#### Plot Assembly ####

layout <- "
AAABC
DDDDD
DDDDD
"

dummy <- ggplot() +
  my_theme +
  theme(legend.position = "none")

assembly <- dummy +
  cldice_plot +
  topo_error_plot +
  dummy +
  plot_layout(design = layout) +
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
  file.path(fig_output_dir, "Fig5_anatomy_accuracy.svg"),
  assembly,
  width = 8,
  height = 7,
  units = "in",
  dpi = 150,
)
