library(pacman)

p_load(data.table)
p_load(ggplot2)
p_load(readxl)
p_load(DataExplorer)
p_load(janitor)
p_load(naniar)
p_load(vip)
p_load(tidymodels)
p_load(e1071)
p_load(esquisse)
p_load(robustbase)
p_load(outliers)
p_load(plotly)
p_load(corrplot)
p_load(Boruta)
p_load(factoextra)
p_load(FactoMineR)



#1) LOADING DATA ----

#setting directory
setwd("C:\\Users\\pedro_jw08iyg\\OneDrive\\Área de Trabalho\\DSA\\Projetos\\Big Data Analytics com R e Microsoft Azure Machine Learning\\Projeto2")

readxl::excel_sheets('data/Acoustic_Extinguisher_Fire_Dataset.xlsx')
raw_dataframe = readxl::read_excel('data/Acoustic_Extinguisher_Fire_Dataset.xlsx',sheet = 'A_E_Fire_Dataset')

data.table::setDT(raw_dataframe)

summary(raw_dataframe)

#change variable's types 
raw_dataframe[,STATUS:=ifelse(STATUS==0,'non_extinction','extinction')]
raw_dataframe[,STATUS:=factor(STATUS, levels = c('extinction','non_extinction'))]
raw_dataframe[,SIZE:=as.integer(SIZE)]
raw_dataframe[,DISTANCE:=as.integer(DISTANCE)]

#checking levels order for target
levels(raw_dataframe[,STATUS])

#balance for target variable
table(raw_dataframe[,STATUS])


# 2) EDA ----


#2.1) size vs fuel
ggplot(raw_dataframe) +
  aes(x = SIZE, fill = FUEL) +
  geom_boxplot() +
  scale_fill_hue(direction = 1) +
  theme_minimal()


summary(raw_dataframe[FUEL=='lpg',.(SIZE)])

#we have a clear difference in size range between liquid and gas fuels.
#as we are dealing with different physical properties, may size is not
#directly comparable with among liquid and gas fuels.


#2.2) airflow vs status
ggplot(raw_dataframe) +
  aes(x = AIRFLOW, fill = STATUS, alpha = 0.6) +
  geom_density(adjust = 1L) +
  scale_fill_hue(direction = 1) +
  theme_minimal()

#we have a high amount of register where airflow = 0.
#as we increase airflow, there's more probability of put out the fire.


#2.3) decibel x status
ggplotly(ggplot(raw_dataframe) +
  aes(x = DESIBEL, fill = STATUS, alpha = 0.6) +
  geom_density(adjust = 1L) +
  scale_fill_hue(direction = 1) +
  theme_minimal())
  
ggplotly(ggplot(raw_dataframe) +
           aes(y = DESIBEL, x = STATUS, fill = STATUS) +
           geom_boxplot()+
           theme_minimal())

#desibel variable has a bivariate distribution.


#2.4) frequency x distance x desibel

ggplotly(  ggplot(raw_dataframe) +
           aes(x = FREQUENCY, y = DISTANCE, color = DESIBEL) +
           geom_jitter(alpha = 0.6) +
           scale_color_viridis_c()
         )


#there's some irregular patterns. For example, for distance = 50 and 
#frequency in [38,44] there's a cluster of low desibel measures



#2.5) frequency x distance x airflow

ggplotly(
          ggplot(raw_dataframe) +
           aes(x = FREQUENCY, y = DISTANCE, color = AIRFLOW) +
           geom_jitter(alpha = 0.6)+
           scale_color_viridis_c() 
         
         )

#there's a clearly pattern here. As we increse distance, the air flow tends to be lower.
#frequency is more intense in the airflow around 45 Hz. 
#there's an anomaly airflow cluster in frequency == 55 Hz and distance = 50cm 


#2.6) airflow x desibel x status

ggplotly(
  ggplot(raw_dataframe) +
    aes(x = AIRFLOW, y = DESIBEL, color = STATUS) +
    geom_jitter(alpha = 0.4)+
    scale_color_viridis_d() 
  
)

#We can clearly see that desibel performs a partiton in our scatterplot at 99db 


# 3) ELIGIBILITY CRITERIA FOR MODELING ----

#3.1) Size variable does not have the same meaning in liquid and gas fuels, then we have to split our data before using this variable
#3.2) Distance and frequency are pre-fixed experimental values, as opposed to desibel and airflow, which are measured variables. 
#3.3) We'll discretize desibel variable because of the patterns we aleady discussed


#feature selection
cleaned_dataframe = copy(raw_dataframe)
cleaned_dataframe[,DESIBEL_DISCRETE:=as.integer(ifelse(DESIBEL<99,0,1))]
cleaned_dataframe[,DESIBEL:=NULL]


#splitting dataset into gas and liquid fuels
cleaned_dataframe_gas = copy(cleaned_dataframe)[FUEL=='lpg'] #gas dataframe
cleaned_dataframe_gas[,FUEL:=NULL]
cleaned_dataframe_gas[,SIZE:=as.factor(SIZE)] #size is not numeric for gas LPG

cleaned_dataframe_liquid = copy(cleaned_dataframe)[FUEL!='lpg'] #liquid dataframe
rm(cleaned_dataframe)


#saving our pre-modeling datasets 
#fwrite(cleaned_dataframe_gas,paste0(getwd(),"/data/cleaned_dataframe_gas.txt"))
#fwrite(cleaned_dataframe_liquid,paste0(getwd(),"/data/cleaned_dataframe_liquid.txt"))

# 5) DATA RESAMPLING ----

set.seed(123)
cleaned_dataframe_split = rsample::initial_split(data = cleaned_dataframe_liquid,
                                                 prop = 0.75,
                                                 strata = STATUS) 

cleaned_dataframe_training = cleaned_dataframe_split |> training()
cleaned_dataframe_testing = cleaned_dataframe_split |> testing()

#6) MODEL SPECIFICATION ----

baseline_model = decision_tree() |> 
  # Set the engine
  set_engine("rpart") |> 
  # Set the mode
  set_mode("classification")



#7) FEATURE ENGINEERING ----

recipe_baseline = recipes::recipe(STATUS~.,
                                  data = cleaned_dataframe_training) |>
  step_range(all_numeric(),-all_outcomes(),-DESIBEL_DISCRETE) |>
  step_dummy(all_nominal(),-all_outcomes())


#8) RECIPE TRAINING ----

recipe_prep_baseline = recipe_baseline |> 
  prep(training = cleaned_dataframe_training)

#9) PREPROCESS TRAINING DATA ----

cleaned_dataframe_training_prep = recipe_prep_baseline |>
  recipes::bake(new_data = NULL)


#10) PREPROCESS TEST DATA ----

cleaned_dataframe_testing_prep = recipe_prep_baseline |> 
  recipes::bake(new_data = cleaned_dataframe_testing)


#11) MODELS FITTING ----

#modeling only with experimental fixed variables 

baseline_model_fit_fixed_variables = baseline_model |>
  parsnip::fit(STATUS ~ FREQUENCY + DISTANCE + SIZE + FUEL_kerosene + FUEL_thinner,
               data = cleaned_dataframe_training_prep)



#modeling only on measured variables by sensors 

baseline_model_fit_measured_variables = baseline_model |>
  parsnip::fit(STATUS ~ DESIBEL_DISCRETE + AIRFLOW + SIZE + FUEL_kerosene + FUEL_thinner,
               data = cleaned_dataframe_training_prep)



#parsnip::tidy(baseline_model_fit)
#parsnip::glance(baseline_model_fit)

#12) PREDICTIONS ON TEST DATA ----

#12.1) Predictions for fixed experimental variables
predictions_fixed_variables = predict(baseline_model_fit_fixed_variables,
                      new_data = cleaned_dataframe_testing_prep)


setDT(predictions_fixed_variables)

predictions_fixed_variables = data.table(predictions_fixed_variables,true_class = cleaned_dataframe_testing_prep$STATUS)

#confusion matrix
conf_mat(data = predictions_fixed_variables,
         estimate = .pred_class,
         truth = true_class)

#probability
proba_temp = predict(baseline_model_fit_fixed_variables,
                                      new_data = cleaned_dataframe_testing_prep,
                                      type = 'prob')


predictions_fixed_variables = data.table(predictions_fixed_variables,predicted_proba = proba_temp$.pred_extinction)
rm(proba_temp)
#AUC = 0.949
yardstick::pr_auc(data = predictions_fixed_variables,
                  estimate = predicted_proba,
                  truth = true_class)

autoplot(yardstick::roc_curve(data = predictions_fixed_variables,
                              estimate = predicted_proba,
                              truth = true_class))


#12.2) Predictions for sensor measured variables
predictions_measured_variables = predict(baseline_model_fit_measured_variables,
                                      new_data = cleaned_dataframe_testing_prep)

setDT(predictions_measured_variables)

predictions_measured_variables = data.table(predictions_measured_variables,true_class = cleaned_dataframe_testing_prep$STATUS)

conf_mat(data = predictions_measured_variables,
         estimate = .pred_class,
         truth = true_class)

#probability
proba_temp = predict(baseline_model_fit_measured_variables,
                     new_data = cleaned_dataframe_testing_prep,
                     type = 'prob')


predictions_measured_variables = data.table(predictions_measured_variables,predicted_proba = proba_temp$.pred_extinction)

#AUC = 0.904
yardstick::pr_auc(data = predictions_measured_variables,
                  estimate = predicted_proba,
                  truth = true_class)

autoplot(yardstick::roc_curve(data = predictions_measured_variables,
                              estimate = predicted_proba,
                              truth = true_class))

#13) CONCLUSIONS
#Both simple decision tree models had a great performance (AUC > 0.90).
#The measured variables was slightly less accurate, maybe because of the noise embedded in the data

