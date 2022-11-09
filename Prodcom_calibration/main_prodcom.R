# Necessary packages ---- 

library(tidyverse)
library(readxl)
library(openxlsx) # writting xlsx results

# Reading the prodcom database

prodcom_2017_QNTUNIT = read_xlsx("Prodcom_calibration/data/Prodcom_data_2017.xlsx", 
                                 sheet = "QNTUNIT")
prodcom_2017_impexp = read_xlsx("Prodcom_calibration/data/Prodcom_data_2017.xlsx", 
                                 sheet = "TRADE") %>%
  mutate(across(indicators, ~ str_split_fixed(.x, " -",2)[,1])) %>%
  rename(PRCCODE = indicators) %>%
  rename(prodcom_digit_8_code = PRCCODE) %>%
  arrange(prodcom_digit_8_code)


prodcom_2017 = read_xlsx("Prodcom_calibration/data/Prodcom_data_2017.xlsx") %>% 
  bind_rows(prodcom_2017_imports) %>%
  select(-DECL, -PERIOD) %>%
  rename(prodcom_digit_8_code = PRCCODE) %>%
  distinct() %>%
  group_by(prodcom_digit_8_code, indicators) %>%
  fill(Value, .direction = "updown") %>%
  ungroup() %>%
  distinct %>%
  pivot_wider(names_from = indicators,
              values_from = Value) %>%
  left_join(prodcom_2017_QNTUNIT %>%
              select(PRCCODE, Value) %>%
              rename(prodcom_digit_8_code = PRCCODE, 
                     QNTUNIT = Value)) %>%
  select(-PRCCODE_LABEL) %>%
  left_join(prodcom_2017_impexp) %>%
  arrange(prodcom_digit_8_code) %>%
  relocate("QNTUNIT", .after = EXPVAL) %>%
  group_by(prodcom_digit_8_code) %>%
  fill(PRODVAL:QNTUNIT, .direction = "updown") %>%
  distinct() %>%
  ungroup()

# Values are in million euros






# Material content Prodcom ---------------------------

prodcom_mat_content <- read_excel("Prodcom_calibration/data/ProdCom_Material_Content_20210603.xlsx", 
                                  sheet = "IoC_MaterialContent_NACEv2") %>%
  # reading the Material content database %>%
  rename(prodcom_digit_8_code = "NACEv2_Code [fields highlighted in red are not counted as they represent aggregates of already listed groups]") %>%
  rename(Prodcom_digit_8_label = NACEv2_Label)
# renaming columns


prodcom_8 = prodcom_2017 %>%
  select(prodcom_digit_8_code) %>%
  unique() %>% 
  pull

prodcom_mat_content_useful = prodcom_mat_content[-1,] 

mat_content = prodcom_mat_content_useful %>%
  filter(prodcom_digit_8_code %in% prodcom_8) %>%
  select(Prodcom_digit_8_label:`Cement...11`, 
         `Grouping of 4047 ProdCom categories into the following four groups: 0: not included/relevant, 1: basic material, 2: pure material product, 3: components of products, 4: final goods/products.`) %>%
  rename(dummy_group = `Grouping of 4047 ProdCom categories into the following four groups: 0: not included/relevant, 1: basic material, 2: pure material product, 3: components of products, 4: final goods/products.`) %>%
  janitor::clean_names() %>%
  
  rename(steel = steel_6,
         aluminium = al_metal_7, 
         copper = cu_metal_8, 
         plastic = plastics_9,
         paper = paper_10, 
         cement = cement_11) %>%
  select(-x5)



# Carbon emissions to produce basic metals -------------------------------------

Approximate_carbon_emissions <- read_excel("Prodcom_calibration/data/Approximate_carbon_emissions.xlsx") %>%
  select(-`Benchmark ETS`) %>%
  group_by(Common_Material) %>%
  mutate(across(where(is.numeric), ~ mean(.x, na.rm = T)))%>%
  ungroup() %>%
  drop_na((where(is.numeric))) %>%
  select(-Assumption) %>% 
  distinct
#unit: tCO2eq

prodcom_2017 %>%
  mutate(cons = PRODVAL - EXPVAL + IMPVAL) %>%
  filter(cons > 0)
  select(cons) %>%
  summary
# Printing output in xlsx format -----

prodcom_2017 %>%
  left_join(mat_content) %>%
  relocate(prodcom_digit_8_label, .after = prodcom_digit_8_code) %>%
  write.xlsx("Prodcom_calibration/ProdCom_and_MatContent.xlsx")

Approximate_carbon_emissions %>%
  rename(`10% least emissive installations 2016-2017` = `10% more efficient 2016-2017`) %>%
  write.xlsx("Prodcom_calibration/carbon_emissions_MatContent.xlsx")
