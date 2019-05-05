library(tidyverse)
library(stargazer)


# Top 20 nodes with different c -------------------------------------------
stargazer(
  read.csv("results/top_20_nodes_with_different_c.csv") %>%
    setNames(c("Node id", "Rank (c=0.3)", "Rank (c=0.1)", "Rank (c=0.5)")),
  title = "First 20 Nodes Returned, with Different c",
  label = "tab:tab_top_20_nodes_different_c",
  type = "latex", summary = FALSE, rownames = FALSE, header = FALSE,
  digit.separator = "",
  out = "tex/top_20_nodes_with_different_c.tex"
)
