library(shiny)
library(tidyverse)
library(leaflet)
library(dplyr)
library(sf)
library(bslib)

# Add spatial data
alcohol_fatality_data <- read_csv("alcohol_fatality_data.csv")
counties <- st_read("ca_counties/CA_Counties.shp")
alcohol_fatality_sf <- st_as_sf(alcohol_fatality_data, coords = c("longitude", "latitude"), crs = 4326)

# Reproject the counties shapefile to EPSG:4326 (same as alcohol_fatality_sf)
counties <- st_transform(counties, crs = st_crs(alcohol_fatality_sf))


# Perform spatial join (collision data joined with counties)
# Step 1: Perform a spatial join that attaches county information to collision points.
collisions_within_counties <- st_join(alcohol_fatality_sf, counties, join = st_within)


# Step 2: Summarize collision data by county while preserving county geometries.
county_collision_summary <- counties %>%
  left_join(
    collisions_within_counties %>%
      st_drop_geometry() %>%  # Drop point geometries from collision data
      group_by(NAME) %>%       # Group by county name (from the counties shapefile)
      summarise(
        total_collisions = n(),
        
        # Total collisions per day of the week
        total_monday = sum(collision_day == "1"),
        total_tuesday = sum(collision_day == "2"),
        total_wednesday = sum(collision_day == "3"),
        total_thursday = sum(collision_day == "4"),
        total_friday = sum(collision_day == "5"),
        total_saturday = sum(collision_day == "6"),
        total_sunday = sum(collision_day == "7"),
        
        # Summarize fatal collisions
        fatal_collisions = sum(fatal == 1),
        
        # Fatal collisions per day of the week
        fatal_monday = sum(fatal == 1 & collision_day == "1"),
        fatal_tuesday = sum(fatal == 1 & collision_day == "2"),
        fatal_wednesday = sum(fatal == 1 & collision_day == "3"),
        fatal_thursday = sum(fatal == 1 & collision_day == "4"),
        fatal_friday = sum(fatal == 1 & collision_day == "5"),
        fatal_saturday = sum(fatal == 1 & collision_day == "6"),
        fatal_sunday = sum(fatal == 1 & collision_day == "7"),
        
        # Total collisions involving alcohol
        total_collisions_alcohol = sum(`alcohol_involved` == 1),
        
        # Total collisions per day of the week involving alcohol
        total_monday_alcohol = sum(`alcohol_involved` == 1 & collision_day == "1"),
        total_tuesday_alcohol = sum(`alcohol_involved` == 1 & collision_day == "2"),
        total_wednesday_alcohol = sum(`alcohol_involved` == 1 & collision_day == "3"),
        total_thursday_alcohol = sum(`alcohol_involved` == 1 & collision_day == "4"),
        total_friday_alcohol = sum(`alcohol_involved` == 1 & collision_day == "5"),
        total_saturday_alcohol = sum(`alcohol_involved` == 1 & collision_day == "6"),
        total_sunday_alcohol = sum(`alcohol_involved` == 1 & collision_day == "7"),
        
        # Fatal collisions involving alcohol
        fatal_collisions_alcohol = sum(fatal == 1 & `alcohol_involved` == 1),
        
        # Fatal collisions per day of the week involving alcohol
        fatal_monday_alcohol = sum(fatal == 1 & `alcohol_involved` == 1 & collision_day == "1"),
        fatal_tuesday_alcohol = sum(fatal == 1 & `alcohol_involved` == 1 & collision_day == "2"),
        fatal_wednesday_alcohol = sum(fatal == 1 & `alcohol_involved` == 1 & collision_day == "3"),
        fatal_thursday_alcohol = sum(fatal == 1 & `alcohol_involved` == 1 & collision_day == "4"),
        fatal_friday_alcohol = sum(fatal == 1 & `alcohol_involved` == 1 & collision_day == "5"),
        fatal_saturday_alcohol = sum(fatal == 1 & `alcohol_involved` == 1 & collision_day == "6"),
        fatal_sunday_alcohol = sum(fatal == 1 & `alcohol_involved` == 1 & collision_day == "7")
      ),
    by = "NAME"  # Join on county name
  )

# Clean county names to remove any unwanted newlines or spaces
county_collision_summary$NAME <- gsub("\\s*<br>\\s*", " ", county_collision_summary$NAME)

glimpse(county_collision_summary)

# Now `county_collision_summary` contains county geometries (MULTIPOLYGON) and summarized collision data.



library(shiny)
library(leaflet)
library(dplyr)
library(shinyjs)  # Load shinyjs for dynamic styling
library(bslib)

# Define custom Bootstrap theme using bslib
custom_theme <- bs_theme(
  bootswatch = "lux",  # You can try other themes like "minty", "darkly", "flatly"
  primary = "#007bff",  # Blue color for primary elements
  secondary = "#6c757d",  # Grey color for secondary elements
  bg = "#F5F5F5",  # Background color for the whole app
  fg = "#333333"  # Font color
)

# Define UI with bslib custom theme
ui <- fluidPage(
  theme = custom_theme,  # Apply the custom theme
  useShinyjs(),  # Initialize shinyjs
  
  tags$head(
    tags$link(
      href = "https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap", 
      rel = "stylesheet"
    ),
    tags$title("California County Collisions"),
    # Custom CSS for buttons and map background, and layout centering
    tags$style(HTML("
      .body {
      font-family: 'Poppins', sans-serif; 
      }
      .day-button {
        width: 150px;  /* Fixed width for all buttons */
        margin: 5px;
        font-family: 'Poppins', sans-serif;
        border-radius: 12px;  /* Round button corners */
        font-weight: bold;
        background-color: transparent;  /* Default to transparent */
        border: 2px solid #007bff;  /* Outline for unselected buttons */
        color: #007bff;
        text-align: center;  /* Center text horizontally */
        padding: 12px;  /* Ensure enough padding for vertical centering */
        display: inline-block;  /* Ensure inline-block layout for consistency */
      }
      .day-button.active {
        background-color: #007bff !important;  
        color: white !important;
        border: none;  /* Remove border when active */
      }
      #map {
        background-color: #F5F5F5;  /* Set the map background to white */
        border-radius: 8px;  /* Rounded corners */
        position: relative;  /* Make the map container relative for absolute positioning inside it */
      }
      .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }
      .main-container {
        text-align: center;
        margin: 0 auto;
      }
      h1 {
        text-align: center;
        padding-top: 20px;
        padding-bottom: 20px;
        margin: 0 auto;
        font-family: 'Poppins', sans-serif;
        font-size: 32px;
      }
    .toggle-container {
    position: fixed; /* Use fixed positioning */
    text-align: left;
    bottom: 20vh;  /* Position at the bottom of the screen */
    left: 20px;  /* Align to the right side of the screen */
    z-index: 999;  /* Ensure the toggles are above the map content */
    background-color: rgba(255, 255, 255, 0.7);  /* Semi-transparent white background */
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0px 0px 5px rgba(0,0,0,0.2);  /* Add a light shadow for visibility */
}

      .toggle-container hr {
        border: 1px solid #cccccc;  /* Style for the horizontal bar */
        margin: 10px 0;  /* Add some spacing around the bar */
      }
    "))
  ),

  
  
  
  div(class = "main-container",
      h1("California County Collisions"),
      div(class = "center-content",
          
          # Leaflet output with checkboxes inside the map
          leafletOutput("map", height = "600px"),
          
          # Toggle buttons within the map panel
          div(class = "toggle-container",
              checkboxInput("fatal_toggle", "Show Fatal Collisions", value = FALSE),
              checkboxInput("alcohol_toggle", "Show Collisions with Alcohol Involved", value = FALSE),
              
              # Horizontal separator between independent and percentage toggles
              tags$hr(),
              
              # Radio buttons for percentage toggles
              radioButtons("percent_toggle", "Percentage View:",
                           choices = list(
                             "None" = "none",
                             "Fatality Rate" = "percent_fatal_total",
                             "Alcohol-Involvement Rate (Fatal Collisions)" = "percent_fatal_alcohol",
                             "Alcohol-Involvement Rate (All Collisions)" = "percent_total_alcohol"
                           ),
                           selected = "none", 
                           inline = FALSE)  # Keep them vertically aligned
          ),
          
          # Buttons for days of the week centered below the map
          div(style = "display: flex; justify-content: center; margin-top: 10px;",
              actionButton("monday", "Monday", class = "day-button btn-primary"),
              actionButton("tuesday", "Tuesday", class = "day-button btn-primary"),
              actionButton("wednesday", "Wednesday", class = "day-button btn-primary"),
              actionButton("thursday", "Thursday", class = "day-button btn-primary"),
              actionButton("friday", "Friday", class = "day-button btn-primary"),
              actionButton("saturday", "Saturday", class = "day-button btn-primary"),
              actionButton("sunday", "Sunday", class = "day-button btn-primary")
          )
      )
  )
)

# Define server logic
server <- function(input, output, session) {
  
  # Reactive variable to track selected days of the week
  selected_days <- reactiveVal(character())
  
  observe({
    if (input$percent_toggle != "none") {
      updateCheckboxInput(session, "fatal_toggle", value = FALSE)
      updateCheckboxInput(session, "alcohol_toggle", value = FALSE)
      disable("fatal_toggle")
      disable("alcohol_toggle")
    } else {
      enable("fatal_toggle")
      enable("alcohol_toggle")
    }
  })
  
  # Observe events for each day button and toggle selection
  observeEvent(input$monday, {
    toggle_day("total_monday", "monday")
  })
  
  observeEvent(input$tuesday, {
    toggle_day("total_tuesday", "tuesday")
  })
  
  observeEvent(input$wednesday, {
    toggle_day("total_wednesday", "wednesday")
  })
  
  observeEvent(input$thursday, {
    toggle_day("total_thursday", "thursday")
  })
  
  observeEvent(input$friday, {
    toggle_day("total_friday", "friday")
  })
  
  observeEvent(input$saturday, {
    toggle_day("total_saturday", "saturday")
  })
  
  observeEvent(input$sunday, {
    toggle_day("total_sunday", "sunday")
  })
  
  # Function to toggle the selected day and give visual feedback to the buttons
  toggle_day <- function(day, button_id) {
    current_days <- selected_days()
    
    if (day %in% current_days) {
      current_days <- setdiff(current_days, day)  # Remove day from the selected list
      runjs(paste0("$('#", button_id, "').css('background-color', 'transparent').css('color', '#007bff').css('border', '2px solid #007bff');"))  # Reset button color
    } else {
      current_days <- c(current_days, day)  # Add day to the selected list
      runjs(paste0("$('#", button_id, "').css('background-color', '#007bff').css('color', 'white').css('border', 'none');"))  # Highlight button
    }
    
    selected_days(current_days)
  }
  
  # Reactive expression for calculating the collision data
  reactive_collision_data <- reactive({
    data <- county_collision_summary
    
    # Debug print statements to track what is happening
    print(paste("Fatal Toggle:", input$fatal_toggle))
    print(paste("Alcohol Toggle:", input$alcohol_toggle))
    print(paste("Percent Toggle:", input$percent_toggle))
    print(paste("Selected Days:", paste(selected_days(), collapse = ", ")))
    
    # Step 1: Determine the appropriate base columns depending on the toggle state
    if (input$fatal_toggle && input$alcohol_toggle) {
      print("Using Fatal + Alcohol Data")
      base_fatal_column <- "fatal_collisions_alcohol"
      base_total_column <- "total_collisions_alcohol"
      day_fatal_columns <- c("fatal_monday_alcohol", "fatal_tuesday_alcohol", "fatal_wednesday_alcohol", "fatal_thursday_alcohol", 
                             "fatal_friday_alcohol", "fatal_saturday_alcohol", "fatal_sunday_alcohol")
      day_total_columns <- c("total_monday_alcohol", "total_tuesday_alcohol", "total_wednesday_alcohol", "total_thursday_alcohol", 
                             "total_friday_alcohol", "total_saturday_alcohol", "total_sunday_alcohol")
    } else if (input$fatal_toggle) {
      print("Using Fatal Data Only")
      base_fatal_column <- "fatal_collisions"
      base_total_column <- "total_collisions"
      day_fatal_columns <- c("fatal_monday", "fatal_tuesday", "fatal_wednesday", "fatal_thursday", 
                             "fatal_friday", "fatal_saturday", "fatal_sunday")
      day_total_columns <- c("total_monday", "total_tuesday", "total_wednesday", "total_thursday", 
                             "total_friday", "total_saturday", "total_sunday")
    } else if (input$alcohol_toggle) {
      print("Using Alcohol Data Only")
      base_fatal_column <- "fatal_collisions_alcohol"
      base_total_column <- "total_collisions_alcohol"
      day_fatal_columns <- c("fatal_monday_alcohol", "fatal_tuesday_alcohol", "fatal_wednesday_alcohol", "fatal_thursday_alcohol", 
                             "fatal_friday_alcohol", "fatal_saturday_alcohol", "fatal_sunday_alcohol")
      day_total_columns <- c("total_monday_alcohol", "total_tuesday_alcohol", "total_wednesday_alcohol", "total_thursday_alcohol", 
                             "total_friday_alcohol", "total_saturday_alcohol", "total_sunday_alcohol")
    } else {
      print("Using Total Data")
      base_fatal_column <- "fatal_collisions"
      base_total_column <- "total_collisions"
      day_fatal_columns <- c("fatal_monday", "fatal_tuesday", "fatal_wednesday", "fatal_thursday", 
                             "fatal_friday", "fatal_saturday", "fatal_sunday")
      day_total_columns <- c("total_monday", "total_tuesday", "total_wednesday", "total_thursday", 
                             "total_friday", "total_saturday", "total_sunday")
    }
    
    # Step 2: Handle Percentage Toggles using the radio button value
    if (input$percent_toggle == "percent_fatal_total") {
      print("Calculating Percentage of Fatal to Total")
      # Calculate percentage fatal of total
      if (length(selected_days()) == 0) {
        data <- data %>% 
          mutate(collisions = ifelse(.data[[base_total_column]] > 0, 
                                     (.data[[base_fatal_column]] / .data[[base_total_column]]) * 100, 0))
      } else {
        # Selected days
        selected_columns_fatal <- day_fatal_columns[match(selected_days(), c("total_monday", "total_tuesday", "total_wednesday", 
                                                                             "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        selected_columns_total <- day_total_columns[match(selected_days(), c("total_monday", "total_tuesday", "total_wednesday", 
                                                                             "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        
        data <- data %>%
          rowwise() %>%
          mutate(
            total_selected = sum(c_across(all_of(selected_columns_total)), na.rm = TRUE),
            fatal_selected = sum(c_across(all_of(selected_columns_fatal)), na.rm = TRUE),
            collisions = ifelse(total_selected > 0, (fatal_selected / total_selected) * 100, 0)
          ) %>%
          ungroup()
      }
    } else if (input$percent_toggle == "percent_fatal_alcohol") {
      print("Calculating Percentage of Fatal Collisions Involving Alcohol")
      # Calculate percentage of fatal collisions that involved alcohol
      if (length(selected_days()) == 0) {
        data <- data %>%
          mutate(collisions = ifelse(.data[["fatal_collisions"]] > 0, 
                                     (.data[["fatal_collisions_alcohol"]] / .data[["fatal_collisions"]]) * 100, 0))
      } else {
        # Selected days for fatal and alcohol
        selected_columns_fatal <- day_fatal_columns[match(selected_days(), c("total_monday", "total_tuesday", "total_wednesday", 
                                                                             "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        selected_columns_fatal_alcohol <- c("fatal_monday_alcohol", "fatal_tuesday_alcohol", "fatal_wednesday_alcohol", 
                                            "fatal_thursday_alcohol", "fatal_friday_alcohol", "fatal_saturday_alcohol", "fatal_sunday_alcohol")[match(selected_days(), 
                                                                                                                                                      c("total_monday", "total_tuesday", "total_wednesday", 
                                                                                                                                                        "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        
        data <- data %>%
          rowwise() %>%
          mutate(
            fatal_selected = sum(c_across(all_of(selected_columns_fatal)), na.rm = TRUE),
            fatal_alcohol_selected = sum(c_across(all_of(selected_columns_fatal_alcohol)), na.rm = TRUE),
            collisions = ifelse(fatal_selected > 0, (fatal_alcohol_selected / fatal_selected) * 100, 0)
          ) %>%
          ungroup()
      }
    } else if (input$percent_toggle == "percent_total_alcohol") {
      print("Calculating Percentage of Total Collisions Involving Alcohol")
      # Calculate percentage of total collisions that involved alcohol
      if (length(selected_days()) == 0) {
        data <- data %>%
          mutate(collisions = ifelse(.data[["total_collisions"]] > 0, 
                                     (.data[["total_collisions_alcohol"]] / .data[["total_collisions"]]) * 100, 0))
      } else {
        # Selected days for total and alcohol
        selected_columns_total <- day_total_columns[match(selected_days(), c("total_monday", "total_tuesday", "total_wednesday", 
                                                                             "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        selected_columns_total_alcohol <- c("total_monday_alcohol", "total_tuesday_alcohol", "total_wednesday_alcohol", 
                                            "total_thursday_alcohol", "total_friday_alcohol", "total_saturday_alcohol", "total_sunday_alcohol")[match(selected_days(), 
                                                                                                                                                      c("total_monday", "total_tuesday", "total_wednesday", 
                                                                                                                                                        "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        
        data <- data %>%
          rowwise() %>%
          mutate(
            total_selected = sum(c_across(all_of(selected_columns_total)), na.rm = TRUE),
            total_alcohol_selected = sum(c_across(all_of(selected_columns_total_alcohol)), na.rm = TRUE),
            collisions = ifelse(total_selected > 0, (total_alcohol_selected / total_selected) * 100, 0)
          ) %>%
          ungroup()
      }
    } else {
      # Step 3: Show total/fatal collisions depending on the selected days and toggle state
      if (length(selected_days()) == 0) {
        # No days selected, use the base column
        data <- data %>% mutate(collisions = .data[[if (input$fatal_toggle) base_fatal_column else base_total_column]])
      } else {
        # Specific days selected
        selected_columns <- if (input$fatal_toggle) {
          day_fatal_columns[match(selected_days(), c("total_monday", "total_tuesday", "total_wednesday", 
                                                     "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        } else {
          day_total_columns[match(selected_days(), c("total_monday", "total_tuesday", "total_wednesday", 
                                                     "total_thursday", "total_friday", "total_saturday", "total_sunday"))]
        }
        
        data <- data %>%
          rowwise() %>%
          mutate(collisions = sum(c_across(all_of(selected_columns)), na.rm = TRUE)) %>%
          ungroup()
      }
    }
    
    return(data)
  })
  
  
  # Render the Leaflet map
  output$map <- renderLeaflet({
    data <- reactive_collision_data()
    
    is_percentage_view <- input$percent_toggle != "none"
    pal <- colorNumeric(palette = "Blues", domain = data$collisions, na.color = "transparent")
    
    # Create a leaflet map using county geometries
    leaflet(data = counties, options = leafletOptions(minZoom = 6, maxZoom = 15)) %>%
      setView(lng = -121, lat = 37.3, zoom = 5.75) %>%
      addPolygons(
        fillColor = ~pal(data$collisions),
        weight = 2,
        opacity = 1,
        color = "#999999",
        dashArray = "1",
        fillOpacity = 0.9,
        highlight = highlightOptions(
          weight = 4,
          color = "#999999",
          dashArray = "",
          fillOpacity = 0.9,
          bringToFront = TRUE
        ),
        label = ~paste(NAME, "–", round(data$collisions, 2), ifelse(input$percent_toggle %in% c("percent_fatal_total", "percent_fatal_alcohol", "percent_total_alcohol"), "%", " collisions")),
        labelOptions = labelOptions(
          style = list("font-weight" = "normal", padding = "3px 8px"),
          textsize = "15px",
          direction = "auto"
        )
      ) %>%
      addLegend(
        pal = pal,
        values = ~data$collisions,
        opacity = 0.7,
        title = if (is_percentage_view) "Percentage (%)" else "Collisions",
        labFormat = labelFormat(suffix = if (is_percentage_view) "%" else ""),  # Add % suffix for percentage view
        position = "bottomright"
      )
  })
}

# Run the app
shinyApp(ui = ui, server = server)
