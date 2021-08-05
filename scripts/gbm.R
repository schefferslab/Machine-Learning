## David Klinges
## 2021-08-05

## This script walks through a simple example exercise of employing Gradient 
## Boosting Machines (GBM), and comparing model performance to a single 
## Decision Tree and a Random Forest (given that these methods are all 
## conceptually similar)

## Sources:
# https://www.youtube.com/watch?v=GZafoSe5DvI
# https://statistik-dresden.de/archives/14967

# Think of Random forest as: "bootstrapping to generate a bunch of decision trees"

# Think of GBM as: "start with one tree, create additional "stumps" (single-split trees) that model the residual error from the first tree.
# We find the cases in which the model predicted poorly before, and give these
# cases a higher weight in the next tree

## Workspace prep ---------

# Do you want to start fresh and train models yourself, or load already-trained
# models?
fresh_start <- FALSE

# Set the seed to generate reproducible results
set.seed(2018)

## Load packages -----------

# caret package: Max Kuhn, v1.0 was in 2008 but has really picked up steam 
# in the past few years:
# https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=TMDDykAAAAAJ
library(caret)
# For building decision trees
library(rpart)
# For plotting decision trees
library(rpart.plot)
# For use of example data `Boston`
library(MASS)
# For tidying
library(tidyverse)

## Load data ----------

data(Boston) # load data
glimpse(Boston) # Look at the structure of the data set

## Load trained models --------

if (!fresh_start) {
  rpart_tree <- read_rds("models/gbm/rpart_tree.rds")
  rf_tree <- read_rds("models/gbm/rf_tree.rds")
  gbm_tree_auto <- read_rds("models/gbm/gbm_tree_auto.rds")
  gbm_tree_tune <- read_rds("models/gbm/gbm_tree_tune.rds")
  gbm_best_tree <- read_rds("models/gbm/gbm_best_tree.rds")
  runtimes <- read_rds("models/gbm/runtimes.rds")
  list2env(runtimes, envir = globalenv())
}

## Prep Cross-Validation -------

# Throughout, we'll be using Cross-Validation to train and validate the models.
# We'll use the trainControl() function to designate a template number of folds
# (k = 10) by which we perform traning
trainctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

## single Decision Tree ----------

# Construct a single Decision Tree
tree <- rpart(medv ~ ., data = Boston, method = "anova", cp = 0.05)
# Plot this tree
rpart.plot(tree)

# CV training of single decision tree
if (fresh_start) {
  decision_tree_time <- system.time({
    rpart_tree <- train(medv ~., data = Boston, method = "rpart", trControl = trainctrl)
  })
}


cat("Training single Decision Tree:", decision_tree_time[[3]] / 60, "minutes")

## Random Forest ------------

if (fresh_start) {
  rf_train_time <- system.time({
    ## CV training of Random Forest
    rf_tree <- train(medv ~., data = Boston, method = "rf",
                     trControl = trainctrl)
  })
}

cat("Training Random Forest:", rf_train_time[[3]] / 60, "minutes")

## Compare single Decision Tree with Random Forest
resamps <- resamples(list(rpart_tree = rpart_tree, randomForest = rf_tree))
summary(resamps)

## GBM with default parameters --------------

if (fresh_start) {
  gbm_auto_train_time <- system.time({
    gbm_tree_auto <- train(medv ~., data = Boston, method = "gbm", distribution = "gaussian", trControl = trainctrl, verbose = FALSE)
    gbm_tree_auto
  })
}


cat("Training GBM with default parameters:", gbm_auto_train_time[[3]] / 60, "minutes")

## Compare single Decision Tree, Random Forest, and default GBM
resamps <- resamples(list(rpart_tree = rpart_tree, randomForest = rf_tree,
                          gbm_tree_auto = gbm_tree_auto))
summary(resamps)

## GBM with tuned parameters -------------

# Let's first check out what parameters one can adjust for a GBM
getModelInfo()$gbm$parameters

# n.trees: the number of boosting iterations (or number of trees generated). I.e.,
# how many times are we modeling the residuals from the last tree

# interaction.depth: the max tree depth. I presume this is how many splits are 
# allowed per tree

# shrinkage: relates to how fast the learning curve is, and therefore the 
# tree-weighting process. I presume this determines how much we weigh cases in 
# which the model set performs poorly

# n.minobsinnode: minimal terminal node size. As in, what is the minimum number 
# of observations we would allow a terminal bin to have (the smallest allowed bin)

# By default, caret holds shrinkage and n.minobsinnode constant, and only explored
# models that had variable n.trees and interaction.depth. But let's try tuning
# all 4 of these parameters

# Here's a grid of parameter values. 4 possible n.trees, 5 possible interaction.depths,
# 5 possible shrinkages, 4 possible n.minobsinnode
# 4 x 5 x 5 x 4 = 400 models, for each of which we run CV. A lot of model training!
myGrid <- expand.grid(n.trees = c (150, 175, 200, 225),
                       interaction.depth = c (5, 6, 7, 8, 9),
                       shrinkage = c (0.075, 0.1, 0.125, 0.15, 0.2),
                       n.minobsinnode = c (7, 10, 12, 15))

# Train all 400 models
if (fresh_start) {
  gbm_tune_train_time <- system.time({
    gbm_tree_tune <- train(medv ~., data = Boston, method = "gbm", 
                           distribution = "gaussian",
                           trControl = trainctrl, verbose = FALSE,
                           tuneGrid = myGrid)
  })
}

cat("Training GBM with tuned parameters:", gbm_tune_train_time[[3]] / 60, "minutes")

gbm_tree_tune

# Now, we just choose the best set of parameters from all 400 models (default
# behavior is to decide by RMSE)
myGrid <- gbm_tree_tune$bestTune
# And now train the model that corresponds to this parameter grid
# I don't quite get why we need to do this again, because I assume this tree
# is buried somewhere in the`gbm_tree_tune` object.
if (fresh_start) {
  gbm_best_train_time <- system.time({
    gbm_best_tree <- train(medv ~., data = Boston, method = "gbm",
                           trControl = trainctrl,
                           tuneGrid = myGrid, verbose = FALSE)
  })
}

cat("Training GBM with optimal parameters:", gbm_best_train_time[[3]] / 60, "minutes")

gbm_best_tree

## Model comparison -----------

resamps <- resamples(list(rpart_tree = rpart_tree, 
                          randomForest = rf_tree, 
                          gbm_tree_auto = gbm_tree_auto, 
                          gbm_best_tree = gbm_best_tree))
summary(resamps)

# Visualize model comparison with a dot plot
dotplot(resamps, metric = "RMSE", main = "model comparison")

## Save models ----------

write_rds(rpart_tree, "models/gbm/rpart_tree.rds")
write_rds(rf_tree, "models/gbm/rf_tree.rds")
write_rds(gbm_tree_auto, "models/gbm/gbm_tree_auto.rds")
write_rds(gbm_tree_tune, "models/gbm/gbm_tree_tune.rds")
write_rds(gbm_best_tree, "models/gbm/gbm_best_tree.rds")

# Save runtimes
runtimes <- list(decision_tree_time = decision_tree_time,
                 rf_train_time = rf_train_time,
                 gbm_auto_train_time = gbm_auto_train_time,
                 gbm_tune_train_time = gbm_tune_train_time,
                 gbm_best_train_time = gbm_best_train_time)

write_rds(runtimes, "models/gbm/runtimes.rds")
