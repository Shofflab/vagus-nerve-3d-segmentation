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
library(broom)

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
    "dice" = "DSC",
    "asd_um" = "ASSD (μm)",
    "surface_dice" = "Surface DSC"
  ),
  .default = label_wrap_gen(20)
)


#### Load Data ####
combined_df <- read_csv(
  here("results", "Fig3_seg_performance.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    model = recode(
      as.character(model),
      "2" = "2D",
      "3" = "3D",
      .default = as.character(model)
    ),
    model = factor(model, levels = c("2D", "3D"))
  )


#### Stats ####

data_by_fold <- combined_df %>%
  group_by(metric, cls, model, fold) %>%
  summarise(value = mean(value), .groups = "drop")

position_by_fold <- data_by_fold %>%
  group_by(metric) %>%
  summarise(y_max = max(value), .groups = "drop")

combined_stat_df <- combined_df %>%
  group_by(metric, cls) %>%
  wilcox_test(value ~ model, paired = TRUE, detailed = TRUE) %>%
  adjust_pvalue() %>%
  add_significance("p.adj") %>%
  add_xy_position(scales = "free_y", fun = "mean_ci")

# use y_max as y.position
combined_stat_df <- combined_stat_df %>%
  left_join(., position_by_fold, by = c("metric")) %>%
  mutate(y.position = y_max)


#### Dice Plot ####

dice_plot <- combined_df %>%
  filter(metric == "dice") %>%
  ggplot(aes(x = model, y = value, color = model)) +
  geom_point(
    data = data_by_fold %>% filter(metric == "dice"),
    aes(y = value),
    size = 2,
    color = "darkgrey",
    alpha = 0.75,
  ) +
  geom_line(
    data = data_by_fold %>% filter(metric == "dice"),
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
  facet_wrap(~cls, labeller = global_labller) +
  labs(x = "Network", y = "DSC") +
  stat_pvalue_manual(
    data = combined_stat_df %>% filter(metric == "dice"),
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
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
  theme(strip.placement = "outside") +
  theme(panel.spacing = unit(0, "lines")) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  scale_y_continuous(breaks = breaks_pretty(5))


#### ASD Plot ####

asd_plot <- combined_df %>%
  filter(metric == "asd_um") %>%
  ggplot(aes(x = model, y = value, color = model)) +
  geom_point(
    data = data_by_fold %>% filter(metric == "asd_um"),
    aes(y = value),
    size = 2,
    color = "darkgrey",
    alpha = 0.75,
  ) +
  geom_line(
    data = data_by_fold %>% filter(metric == "asd_um"),
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
  facet_wrap(~cls, labeller = global_labller) +
  labs(x = "Network", y = "ASSD (μm)") +
  my_theme +
  theme(legend.position = "none") +
  theme(strip.placement = "outside", strip.text = element_blank()) +
  theme(panel.spacing = unit(0, "lines")) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  stat_pvalue_manual(
    data = combined_stat_df %>% filter(metric == "asd_um"),
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 1,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  scale_y_continuous(breaks = breaks_pretty(5))


#### Surface Dice Plot ####

surface_dice_plot <- combined_df %>%
  filter(metric == "surface_dice") %>%
  ggplot(aes(x = model, y = value, color = model)) +
  geom_point(
    data = data_by_fold %>% filter(metric == "surface_dice"),
    aes(y = value),
    size = 2,
    color = "darkgrey",
    alpha = 0.75,
  ) +
  geom_line(
    data = data_by_fold %>% filter(metric == "surface_dice"),
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
  facet_wrap(~cls, labeller = global_labller, ) +
  labs(x = "Network", y = "Surface DSC") +
  my_theme +
  theme(legend.position = "none") +
  theme(
    strip.placement = "outside",
    strip.text = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
  ) +
  theme(panel.spacing = unit(0, "lines")) +
  scale_color_tableau(type = "regular", palette = "Color Blind") +
  stat_pvalue_manual(
    data = combined_stat_df %>% filter(metric == "surface_dice"),
    tip.length = 0,
    label = "p.adj.signif",
    vjust = 1,
    hide.ns = FALSE,
    size = 4,
    bracket.size = 0,
    remove.bracket = FALSE,
    family = "Arial",
  ) +
  scale_y_continuous(breaks = breaks_pretty(5))


#### Plot Assembly ####

# use patchwork to assemble the plots
right_part <- dice_plot / surface_dice_plot / asd_plot
right_part <- right_part +
  plot_layout(guides = "collect", axis_titles = "collect") &
  theme(plot.margin = margin(0, 0, 0, 0))

dummy <- ggplot() +
  geom_blank() +
  theme_bw()
left_part <- dummy

assembly <- left_part +
  plot_spacer() +
  right_part +
  plot_layout(widths = c(1.5, 0.05, 1))

assembly <- assembly +
  plot_annotation(tag_levels = "a") &
  theme(
    plot.tag = element_text(face = "bold", size = 14, hjust = 1, vjust = 1),
    plot.tag.position = c(0, 1)
  )

#### Export ####

ggsave(
  file.path(fig_output_dir, "Fig3_seg_performance.svg"),
  assembly,
  width = 8,
  height = 5,
  units = "in",
  dpi = 300,
)
