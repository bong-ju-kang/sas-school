# Bong Ju Kang
# for for random forest sample with missing imputation
# 5/8/2019
 
library(randomForest)

print(dm_model_formula)
print(dm_input)

head(dm_traindf)
str(dm_traindf)

dm_model <- randomForest(dm_model_formula, ntree=100, mtry=5, data=dm_traindf, importance=T)

# 예측
predicted <- predict(dm_model, dm_inputdf,type='prob')
dm_scoreddf <- data.frame(predicted)
colnames(dm_scoreddf)

# SAS 지정양식으로 만들기
values <- levels(unique(dm_inputdf[,c(dm_dec_target)]))

namelist <- array()
for (i in 1:length(values)){
  namelist[i] <- paste0('P_', dm_dec_target, values[i])
}

colnames(dm_scoreddf) <- namelist

# 출력물 보내고 받기
png("rpt_rf_mse_plot.png")
plot(dm_model, main='Random Forest MSE Plot')
dev.off()
