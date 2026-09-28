# ============================================================
# BEYOND THE DOTS
# Core website figures
#
# Figure 1: Latest SEP dot plot
# Figure 2: Median SEP projection vs. realized year-end EFFR
#           + one shaded SEP-meeting window per calendar year
#           + recession shading
# Figure 3: Disagreement heatmaps
# ============================================================

library(tidyverse)
library(lubridate)
library(scales)

# ============================================================
# SETTINGS
# ============================================================

data_file <- "projections_until_sept2026.csv"

dir.create("figures", showWarnings = FALSE)

col_sep      <- "#0072B2"
col_realized <- "#D55E00"
col_band     <- "#56B4E9"

current_year   <- year(Sys.Date())
completed_year <- current_year - 1


# ============================================================
# READ AND CLEAN SEP DATA
# ============================================================

dat <- read_csv(
  data_file,
  show_col_types = FALSE
) %>%
  mutate(
    SEP_Date = mdy(SEP_Date),
    Year = str_trim(as.character(Year)),
    Projection = as.numeric(Projection)
  )

stopifnot(
  all(c("Year", "Projection", "SEP_Date") %in% names(dat)),
  !any(is.na(dat$SEP_Date)),
  !any(is.na(dat$Projection))
)


# ============================================================
# COMMON HORIZON DEFINITIONS
#
# For an SEP released in year t:
#   1-year ahead = t + 1
#   2-year ahead = t + 2
#   Longer-run benchmark = t + 5
# ============================================================

dat_horizon <- dat %>%
  mutate(
    sep_year = year(SEP_Date),
    
    projection_year =
      suppressWarnings(as.integer(Year)),
    
    horizon = case_when(
      projection_year == sep_year + 1 ~ "1-year ahead",
      projection_year == sep_year + 2 ~ "2-year ahead",
      str_to_lower(Year) == "longer run" ~ "Longer run",
      TRUE ~ NA_character_
    ),
    
    target_year = case_when(
      horizon == "1-year ahead" ~ sep_year + 1L,
      horizon == "2-year ahead" ~ sep_year + 2L,
      horizon == "Longer run"   ~ sep_year + 5L,
      TRUE ~ NA_integer_
    )
  )


# ============================================================
# FIGURE 1
# LATEST SEP DOT PLOT
# ============================================================

latest_sep_date <- max(
  dat$SEP_Date,
  na.rm = TRUE
)

latest_sep <- dat %>%
  filter(
    SEP_Date == latest_sep_date
  ) %>%
  mutate(
    projection_year =
      suppressWarnings(as.integer(Year)),
    
    Year_clean = case_when(
      str_to_lower(Year) == "longer run" ~ "Longer run",
      TRUE ~ Year
    )
  )


# Numerical forecast years first, then Longer run
year_levels <- latest_sep %>%
  filter(
    !is.na(projection_year)
  ) %>%
  distinct(
    Year_clean,
    projection_year
  ) %>%
  arrange(
    projection_year
  ) %>%
  pull(
    Year_clean
  )


latest_sep <- latest_sep %>%
  mutate(
    Year_clean = factor(
      Year_clean,
      levels = c(
        year_levels,
        "Longer run"
      )
    )
  )


# Exact horizontal stacking of identical projections
latest_dot_dat <- latest_sep %>%
  arrange(
    Year_clean,
    Projection
  ) %>%
  group_by(
    Year_clean,
    Projection
  ) %>%
  mutate(
    dot_number = row_number(),
    n_dots = n(),
    
    x_base = as.numeric(Year_clean),
    
    x_plot =
      x_base +
      0.035 * (
        dot_number -
          (n_dots + 1) / 2
      )
  ) %>%
  ungroup()


fig_latest <- ggplot(
  latest_dot_dat,
  aes(
    x = x_plot,
    y = Projection
  )
) +
  
  geom_point(
    shape = 21,
    size = 3.0,
    stroke = 0.35,
    fill = col_sep,
    color = col_sep
  ) +
  
  scale_x_continuous(
    breaks = seq_along(
      levels(latest_dot_dat$Year_clean)
    ),
    
    labels = levels(
      latest_dot_dat$Year_clean
    ),
    
    expand = expansion(
      mult = c(0.08, 0.08)
    )
  ) +
  
  scale_y_continuous(
    breaks = pretty_breaks(n = 8),
    
    expand = expansion(
      mult = c(0.03, 0.08)
    )
  ) +
  
  labs(
    title = paste0(
      "Latest FOMC policy-rate projections — ",
      format(
        latest_sep_date,
        "%B %Y"
      )
    ),
    
    subtitle =
      "Each dot represents one FOMC participant",
    
    x = NULL,
    
    y =
      "Federal funds rate (percent)"
  ) +
  
  theme_classic(
    base_size = 13
  ) +
  
  theme(
    plot.title = element_text(
      face = "bold",
      size = 15
    ),
    
    plot.subtitle = element_text(
      size = 11
    ),
    
    axis.text.x = element_text(
      size = 11
    ),
    
    axis.title.y = element_text(
      margin = margin(r = 10)
    )
  )


print(fig_latest)


ggsave(
  "figures/latest_sep_dotplot.png",
  fig_latest,
  width = 8,
  height = 5,
  dpi = 300
)


# ============================================================
# FIGURE 2
# MEDIAN SEP PROJECTION VS. REALIZED YEAR-END
# EFFECTIVE FEDERAL FUNDS RATE
# ============================================================


# ------------------------------------------------------------
# SEP distribution summaries
# ------------------------------------------------------------

sep_summary <- dat_horizon %>%
  filter(
    !is.na(horizon)
  ) %>%
  
  group_by(
    SEP_Date,
    sep_year,
    horizon,
    target_year
  ) %>%
  
  summarise(
    n_participants = n(),
    
    p25 = quantile(
      Projection,
      probs = 0.25,
      na.rm = TRUE,
      type = 7
    ),
    
    median_projection = median(
      Projection,
      na.rm = TRUE
    ),
    
    p75 = quantile(
      Projection,
      probs = 0.75,
      na.rm = TRUE,
      type = 7
    ),
    
    .groups = "drop"
  )


# ------------------------------------------------------------
# Download daily effective federal funds rate from FRED
#
# DFF = daily Effective Federal Funds Rate
# FRED date field = observation_date
# ------------------------------------------------------------

dff <- read_csv(
  "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF",
  na = c(".", ""),
  show_col_types = FALSE
) %>%
  
  mutate(
    observation_date =
      ymd(observation_date),
    
    DFF =
      as.numeric(DFF),
    
    year =
      year(observation_date)
  ) %>%
  
  filter(
    !is.na(observation_date),
    !is.na(DFF)
  )


# ------------------------------------------------------------
# Final daily DFF observation of each COMPLETED calendar year
#
# Current year is deliberately excluded.
# ------------------------------------------------------------

dff_year_end <- dff %>%
  
  filter(
    year <= completed_year
  ) %>%
  
  group_by(
    year
  ) %>%
  
  slice_max(
    order_by = observation_date,
    n = 1,
    with_ties = FALSE
  ) %>%
  
  ungroup() %>%
  
  transmute(
    target_year = year,
    realized_date = observation_date,
    realized_rate = DFF
  )


# ------------------------------------------------------------
# Match realized rate to each SEP horizon
# ------------------------------------------------------------

forecast_realized <- sep_summary %>%
  
  left_join(
    dff_year_end,
    by = "target_year"
  ) %>%
  
  mutate(
    horizon = factor(
      horizon,
      
      levels = c(
        "1-year ahead",
        "2-year ahead",
        "Longer run"
      ),
      
      labels = c(
        "1-year ahead",
        "2-year ahead",
        "Longer run: 5-year benchmark"
      )
    )
  )


# Validation:
# current/future years must not have realized year-end values
stopifnot(
  !any(
    forecast_realized$target_year >= current_year &
      !is.na(forecast_realized$realized_rate)
  )
)


# ------------------------------------------------------------
# ONE meeting-window box per calendar year
#
# Box starts at first SEP release of year
# and ends at last SEP release of year.
# ------------------------------------------------------------

meeting_year_boxes <- dat %>%
  
  distinct(
    SEP_Date
  ) %>%
  
  mutate(
    sep_year = year(SEP_Date)
  ) %>%
  
  group_by(
    sep_year
  ) %>%
  
  summarise(
    start = min(SEP_Date),
    end   = max(SEP_Date),
    .groups = "drop"
  )


# ------------------------------------------------------------
# NBER recession
# ------------------------------------------------------------

recessions <- tibble(
  start = as.Date("2020-02-01"),
  end   = as.Date("2020-04-30")
)


# ------------------------------------------------------------
# X-axis:
# place year labels at July 1
# ------------------------------------------------------------

first_year <- min(
  year(forecast_realized$SEP_Date),
  na.rm = TRUE
)

last_year <- max(
  year(forecast_realized$SEP_Date),
  na.rm = TRUE
)

year_breaks <- as.Date(
  paste0(
    first_year:last_year,
    "-07-01"
  )
)


# ------------------------------------------------------------
# Figure 2
# ------------------------------------------------------------

fig_forecast <- ggplot(
  forecast_realized,
  aes(
    x = SEP_Date
  )
) +
  
  # Annual SEP meeting window
  geom_rect(
    data = meeting_year_boxes,
    
    aes(
      xmin = start,
      xmax = end,
      ymin = -Inf,
      ymax = Inf
    ),
    
    inherit.aes = FALSE,
    
    fill = "#DCEAF4",
    alpha = 0.32
  ) +
  
  # 2020 recession
  geom_rect(
    data = recessions,
    
    aes(
      xmin = start,
      xmax = end,
      ymin = -Inf,
      ymax = Inf
    ),
    
    inherit.aes = FALSE,
    
    fill = "grey55",
    alpha = 0.30
  ) +
  
  # 25th-75th percentile SEP band
  geom_ribbon(
    aes(
      ymin = p25,
      ymax = p75
    ),
    
    fill = col_band,
    alpha = 0.25
  ) +
  
  # Median SEP projection
  geom_line(
    aes(
      y = median_projection,
      color = "Median SEP projection"
    ),
    
    linewidth = 1.05
  ) +
  
  geom_point(
    aes(
      y = median_projection,
      color = "Median SEP projection"
    ),
    
    size = 1.4,
    alpha = 0.80
  ) +
  
  # Realized year-end EFFR
  geom_line(
    aes(
      y = realized_rate,
      color = "Realized year-end rate"
    ),
    
    linewidth = 1.05,
    linetype = "dashed",
    na.rm = TRUE
  ) +
  
  geom_point(
    aes(
      y = realized_rate,
      color = "Realized year-end rate"
    ),
    
    size = 1.4,
    alpha = 0.80,
    na.rm = TRUE
  ) +
  
  facet_wrap(
    ~ horizon,
    ncol = 1
  ) +
  
  scale_color_manual(
    values = c(
      "Median SEP projection" =
        col_sep,
      
      "Realized year-end rate" =
        col_realized
    )
  ) +
  
  scale_x_date(
    breaks = year_breaks,
    
    labels = format(
      year_breaks,
      "%Y"
    ),
    
    expand = expansion(
      mult = c(0.01, 0.02)
    )
  ) +
  
  scale_y_continuous(
    breaks = pretty_breaks(
      n = 6
    ),
    
    expand = expansion(
      mult = c(0.03, 0.07)
    )
  ) +
  
  labs(
    x = NULL,
    
    y =
      "Federal funds rate (percent)",
    
    color = NULL,
    
    caption =
      "Light blue shading spans the first through last SEP release in each calendar year; gray shading marks the 2020 recession."
  ) +
  
  guides(
    color = guide_legend(
      nrow = 1,
      byrow = TRUE
    )
  ) +
  
  theme_classic(
    base_size = 13
  ) +
  
  theme(
    legend.position =
      "bottom",
    
    legend.justification =
      "center",
    
    legend.text =
      element_text(
        size = 11
      ),
    
    strip.text =
      element_text(
        face = "bold",
        size = 12
      ),
    
    strip.background =
      element_blank(),
    
    panel.spacing =
      grid::unit(
        1.1,
        "lines"
      ),
    
    axis.title.y =
      element_text(
        margin =
          margin(r = 10)
      ),
    
    axis.text.x =
      element_text(
        angle = 45,
        hjust = 1
      ),
    
    plot.caption =
      element_text(
        size = 9,
        hjust = 0,
        color = "grey35",
        margin = margin(t = 8)
      ),
    
    plot.margin =
      margin(
        t = 10,
        r = 15,
        b = 5,
        l = 10
      )
  )


print(fig_forecast)


ggsave(
  "figures/median_vs_realized.png",
  fig_forecast,
  width = 9,
  height = 9,
  dpi = 300
)


# ============================================================
# FIGURE 3
# DISAGREEMENT HEATMAPS
#
# Cross-sectional IQR:
#   1-year ahead
#   2-year ahead
#   Longer run
# ============================================================

iqr_heat <- dat_horizon %>%
  
  filter(
    !is.na(horizon)
  ) %>%
  
  mutate(
    sep_month =
      month(
        SEP_Date,
        label = TRUE,
        abbr = TRUE
      )
  ) %>%
  
  group_by(
    SEP_Date,
    sep_year,
    sep_month,
    horizon
  ) %>%
  
  summarise(
    n_participants = n(),
    
    iqr =
      quantile(
        Projection,
        probs = 0.75,
        na.rm = TRUE,
        type = 7
      ) -
      quantile(
        Projection,
        probs = 0.25,
        na.rm = TRUE,
        type = 7
      ),
    
    .groups = "drop"
  ) %>%
  
  mutate(
    # ggplot places the first factor level at bottom.
    # Hence top-to-bottom display becomes:
    # Jan, Mar, Apr, Jun, Sep, Dec
    sep_month = factor(
      as.character(sep_month),
      
      levels = c(
        "Dec",
        "Sep",
        "Jun",
        "Apr",
        "Mar",
        "Jan"
      )
    ),
    
    horizon = factor(
      horizon,
      
      levels = c(
        "1-year ahead",
        "2-year ahead",
        "Longer run"
      )
    )
  )


# Common scale across all horizons
iqr_limits <- range(
  iqr_heat$iqr,
  na.rm = TRUE
)


fig_disagreement <- ggplot(
  iqr_heat,
  
  aes(
    x = factor(sep_year),
    y = sep_month,
    fill = iqr
  )
) +
  
  geom_tile(
    color = "white",
    linewidth = 0.7
  ) +
  
  geom_text(
    aes(
      label = sprintf(
        "%.2f",
        iqr
      )
    ),
    
    size = 3
  ) +
  
  scale_fill_viridis_c(
    option = "C",
    direction = -1,
    limits = iqr_limits,
    name = "IQR"
  ) +
  
  facet_wrap(
    ~ horizon,
    ncol = 1
  ) +
  
  labs(
    x = NULL,
    y = NULL,
    fill = "IQR"
  ) +
  
  theme_minimal(
    base_size = 13
  ) +
  
  theme(
    panel.grid =
      element_blank(),
    
    strip.text =
      element_text(
        face = "bold",
        size = 12
      ),
    
    strip.background =
      element_blank(),
    
    axis.text.x =
      element_text(
        angle = 45,
        hjust = 1
      ),
    
    axis.text.y =
      element_text(
        size = 10.5
      ),
    
    legend.position =
      "right",
    
    panel.spacing =
      grid::unit(
        1.2,
        "lines"
      ),
    
    plot.margin =
      margin(
        t = 10,
        r = 10,
        b = 5,
        l = 5
      )
  )


print(fig_disagreement)


ggsave(
  "figures/disagreement_heatmaps.png",
  fig_disagreement,
  width = 10,
  height = 9,
  dpi = 300
)


# ============================================================
# PDF VERSIONS
# ============================================================

ggsave(
  "figures/latest_sep_dotplot.pdf",
  fig_latest,
  width = 8,
  height = 5
)

ggsave(
  "figures/median_vs_realized.pdf",
  fig_forecast,
  width = 9,
  height = 9
)

ggsave(
  "figures/disagreement_heatmaps.pdf",
  fig_disagreement,
  width = 10,
  height = 9
)


# ============================================================
# FINAL CHECKS
# ============================================================

cat(
  "\nLatest SEP:",
  format(
    latest_sep_date,
    "%Y-%m-%d"
  ),
  "\n"
)

cat(
  "Latest completed year used for realized rates:",
  completed_year,
  "\n"
)

cat(
  "Number of SEP releases:",
  n_distinct(dat$SEP_Date),
  "\n"
)

cat(
  "Number of annual SEP meeting windows:",
  nrow(meeting_year_boxes),
  "\n"
)

cat(
  "All three figures generated successfully.\n"
)