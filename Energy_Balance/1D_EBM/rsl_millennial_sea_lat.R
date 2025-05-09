# install package

if (!"dplyr" %in% installed.packages()) install.packages("dplyr")
library(dplyr)

if (!"readxl" %in% installed.packages()) install.packages("readxl")
library(readxl)

# read data 

Data <- read_xlsx("rsl_millennial_sea_lat.xlsx", .name_repair = "minimal") 


Data <- as.data.frame(Data)

Data[,2] <- as.numeric(Data[,2])
Data[,5] <- as.numeric(Data[,5])
Data[,7] <- as.numeric(Data[,7])
Data[,11] <- as.numeric(Data[,11])

norm_age <- (Data[[1]] - min(Data[[1]])) / (max(Data[[1]]) - min(Data[[1]]))





# --------------------------------------------
# 1.1 PLOT 65 Summer vs 65 Seasonal Difference

colors_violet <- rgb(0.93, 0.51, 0.93, alpha = norm_age) 
plot(Data[,6], Data[,2], type='p', pch=16, cex=1.5, 
     ylab="Rate of Sea Level Change", xlab="Insolation", col=colors_violet)


title( main="65 Summer VS Seasonal Differences", line = 4)

colors_blue <- rgb(0, 0.5, 1, alpha = norm_age)
points(Data[,8], Data[,2], pch=16, cex=1.5, col=colors_blue)

axis(side=3, col="darkblue", col.axis="darkblue")
mtext("Seasonal Differences", side=3, line=2.5, col="darkblue")

grid()


# --------------------------------------------
# 1.2 PLOT 65 Summer vs Latitudinal Difference



# Colors with transparency based on age
colors_violet <- rgb(0.93, 0.51, 0.93, alpha = norm_age)
colors_green <- rgb(0.5, 0.75, 0.5, alpha = norm_age)

# Adjust margins
par(mar=c(5, 4, 6, 4))

# Plot the first dataset (Insolation vs Rate of Sea Level Change)
plot(Data[,6], Data[,2], type='p', pch=16, cex=1.5, 
     ylab="Rate of Sea Level Change", xlab="65N summer", 
     col=colors_violet)

title(main="65 Summer VS Latitudinal Differences", line=4)

# Overlay the second dataset (Latitudinal Differences vs Rate of Sea Level Change)
par(new=TRUE)  # Allow overlaying a new plot

# Plot the second dataset without axes or labels
plot(Data[,11], Data[,2], type='p', pch=16, cex=1.5, col=colors_green, 
     axes=FALSE, xlab="", ylab="")

# Add a second x-axis at the top (side=3)
axis(side=3, col="darkgreen", col.axis="darkgreen")
mtext("Latitudinal Differences", side=3, line=2.5, col="darkgreen")

# Add grid if needed (optional)
grid()


# -------------------------------------------
# 2.1 Determine thresholds


data$High_65Summer <- data$`summer solstice` > quantile(data$`summer solstice`, 0.75)
# data$Low_65Summer <- data$`summer solstice` < quantile(data$`summer solstice`, 0.15)
data$High_Sea_Diff <- data$`seasonal diff` > quantile(data$`seasonal diff`, 0.9)
data$Low_Sea_Diff <- data$`seasonal diff` < quantile(data$`seasonal diff`, 0.85)


# Condition 1: High 65°N Summer Insolation and High Seasonal Insolation Difference
High_65Summer_High_Sea_Diff <- subset(data, High_65Summer & High_Sea_Diff)

# Condition 2: High 65°N Summer Insolation and Low Seasonal Insolation Difference
High_65Summer_Low_Sea_Diff <- subset(data, High_65Summer & Low_Sea_Diff)



# Regression for Condition 1
model_H65_HSea <- lm(`rsl` ~ `summer solstice` + `seasonal diff`, data = High_65Summer_High_Sea_Diff)
summary(model_H65_HSea)

# Regression for Condition 2
model_H65_LSea <- lm(`rsl` ~ `summer solstice` + `seasonal diff`, data = High_65Summer_Low_Sea_Diff)
summary(model_H65_LSea)


# ---------------------------------------------
# 2.2 VISUALIZATION

# Set up the plotting area to have 2 rows and 1 column
par(mfrow = c(2, 1))

# Plot for Condition 1: High 65°N Summer Insolation and High Seasonal Insolation Difference
plot(High_65Summer_High_Sea_Diff$`summer solstice`, High_65Summer_High_Sea_Diff$rsl,
     pch = 16, # Solid circle for points
     cex = 1.5, # Increase the size of the points
     xlab = "Summer Solstice Insolation",
     ylab = "RSL",
     main = "High 65°N Summer Insolation and Great Seasonal Insolation Difference (>90%)")

# Plot for Condition 2: High 65°N Summer Insolation and Low Seasonal Insolation Difference
plot(High_65Summer_Low_Sea_Diff$`summer solstice`, High_65Summer_Low_Sea_Diff$rsl,
     pch = 16, # Solid circle for points
     cex = 1.5, # Increase the size of the points
     xlab = "Summer Solstice Insolation",
     ylab = "RSL",
     main = "High 65°N Summer Insolation and Minor Seasonal Insolation Difference (<85%)")
