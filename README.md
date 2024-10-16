# Alcohol Traffic Collisions and Fatalities in California

This repository contains an analysis of traffic accident fatalities in California using data from the Statewide Integrated Traffic Records System (SWITRS). The data covers incidents from 2020 and 2021 and is stored in a SQLite database.

## Project Overview

The analysis investigates the relationship between alcohol involvement and traffic collisions, particularly focusing on fatalities. The analysis is accompanied by a Shiny app, which provides an interactive map displaying traffic fatalities across California counties.

### Data Source

The data is sourced from SWITRS, which can be accessed via [TIMS Berkeley](https://tims.berkeley.edu/help/SWITRS.php). The SQLite database includes information on all traffic incidents within California from 2020 and 2021.

### Dependencies

To reproduce the analysis, you need the following R packages:



```
library(DBI)
library(RSQLite)
library(tidyverse)
library(ggplot2)
library(shiny)
library(readr)
library(plotly)
library(leaflet)
library(sf)
```

### Key Features

1. **Data Analysis**:
   - The \`collisions\` and \`parties\` tables are linked by \`case_id\`.
   - An analysis of vehicle makes involved in weekend collisions.
   - Exploration of the relationship between alcohol involvement and traffic fatalities.

2. **Shiny App**:
   - Interactive map showing traffic collision data across California counties.
   - Filter by day of the week, collision severity, and alcohol involvement.
   - Available online at: [Shiny App](https://samielsabri.shinyapps.io/alcohol_traffic_fatalities/).

3. **Machine Learning**:
   - A logistic regression model to predict the likelihood of a fatality in collisions involving alcohol.

### Usage

Clone the repository and run the R scripts to reproduce the analysis. To launch the Shiny app locally, use:

```
shiny::runApp()
```

For more information, view the data dictionary at [TIMS Berkeley](https://tims.berkeley.edu/help/SWITRS.php).
"""

