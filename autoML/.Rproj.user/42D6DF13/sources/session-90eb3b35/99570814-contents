# load libraries
library(h2o)
library(iForecast)

library(tidyverse)
library(readxl)
library(zoo)

# Initialize H2O
h2o.init(max_mem_size = "8g")

# open data and specify variable classes
data_r <- read_xlsx('PJMdatatest1.xlsx') %>% 
  
  # remove extra date variables
  select(-c('Date',
            'Day',
            'Month',
            'Year',
            'Days Back')
         ) %>% 
  
  # convert to time series structure
  mutate(time = as.Date(`Date/Time`)) %>% 
  select(-`Date/Time`)

# specify target and exogenous predictors
y_zoo <- zoo(data_r$`DA LMP`,
             order.by = data_r$time)

x_zoo <- data_r[, 
                setdiff(colnames(data_r), 
                        c("time", "DA LMP")),
                drop = FALSE] %>% 
  zoo(x = .,
      order.by = data_r$time)

# Choosing the dates of training and testing data
train.end <- "2025-07-30"

# fit autoML
model <- 
  tts.autoML(y = y_zoo,
             x = x_zoo,
             train.end = train.end,
             arOrder   = c(1, 2, 7),   # 1, 2, 7 day lag of y
             xregOrder = c(0, 1, 2, 7), # 0, 1, 2, 7 day lag of x
             type = 'none', # stupid setting
             initial = F) # don't autoinitialize

# leaderboard
model$modelsUsed %>% print()

# out of sample cross-validation
testData2 <- 
  window(model$dataused,
         start = "2025-07-30",
         end= end(model$data))

lb <- model$modelsUsed  # leaderboard H2OFrame
model_ids <- as.vector(lb$model_id)

results <- lapply(model_ids, function(mid) {
  m <- h2o.getModel(mid)
  perf <- h2o.performance(m, newdata = model$testData)
  data.frame(model_id = mid,
             RMSE = h2o.rmse(perf),
             MAE  = h2o.mae(perf))
})
do.call(rbind, results)
        
#testData2 <- window(autoML$dataused,start="2009-01-01",end=end(autoML$data))
#P1<-iForecast(Model=autoML,newdata=testData2,type="static")

h2o.varimp_heatmap(object = model$modelsUsed, 
                   top_n = 10, 
                   num_of_features = 20)



