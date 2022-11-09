# Necessary packages ---- 

library(tidyverse)
library(readxl)
library(openxlsx) # writting xlsx results


carbon_tax = 100 # € per ton of CO2eq

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


# Printing output in xlsx format -----

prodcom_2017 %>%
  left_join(mat_content) %>%
  relocate(prodcom_digit_8_label, .after = prodcom_digit_8_code) %>%
  write.xlsx("Prodcom_calibration/ProdCom_and_MatContent.xlsx")

Approximate_carbon_emissions %>%
  rename(`10% least emissive installations 2016-2017` = `10% more efficient 2016-2017`) %>%
  write.xlsx("Prodcom_calibration/carbon_emissions_MatContent.xlsx")

final_df = prodcom_2017 %>%
  left_join(mat_content) %>%
  relocate(prodcom_digit_8_label, .after = prodcom_digit_8_code) %>% 
  mutate(across(kg_per_unit, as.numeric)) %>%
  # Get quantities in tons
  mutate(across(c(PRODQNT, EXPQNT, IMPQNT), ~ kg_per_unit * .x / 1e3)) %>%
  mutate(imp_deflator = IMPVAL / IMPQNT, 
         exp_deflator = EXPVAL / EXPQNT, 
         prod_deflator = PRODVAL / PRODQNT, 
         deflator_exist = ifelse(!is.na(imp_deflator) |
                                   !is.na(exp_deflator) | 
                                   !is.na(prod_deflator), 
                                 TRUE, 
                                 FALSE)) %>%
  mutate(deflator = ifelse(!is.na(prod_deflator), prod_deflator, NA), 
         deflator_used = ifelse(!is.na(deflator), "prod", NA)) %>%
  mutate(deflator = ifelse(is.na(deflator), exp_deflator, deflator), 
         deflator_used = ifelse(!is.na(deflator) & is.na(deflator_used),
                                "exp", 
                                deflator_used
                                )
         ) %>%
  
  mutate(deflator = ifelse(is.na(deflator), imp_deflator, deflator), 
         deflator_used = ifelse(!is.na(deflator) & is.na(deflator_used), 
                                "imp", 
                                deflator_used)) %>%
  # Note that this deflator of imports is never used
  mutate(across(c(steel,
                  aluminium, 
                  #copper,
                  #plastic,
                  paper,
                  cement), 
                as.numeric)) %>%
  rename_with(.cols = c(steel,
                        aluminium, 
                        #copper,
                        #plastic, 
                        paper,
                        cement), 
              ~ paste0(.x, "_content_intensity")
  ) %>%
  #mutate(deflated_carbon_price = carbon_tax / deflator_price) %>%
  mutate(steel_carbon_content = Approximate_carbon_emissions$`Average emissions 2016-2017`[1], 
         aluminium_carbon_content = Approximate_carbon_emissions$`Average emissions 2016-2017`[2], 
         cement_carbon_content = Approximate_carbon_emissions$`Average emissions 2016-2017`[3], 
         paper_carbon_content = Approximate_carbon_emissions$`Average emissions 2016-2017`[5]) %>%
  mutate(emission_coef_pton = steel_content_intensity * steel_carbon_content + 
           aluminium_content_intensity * aluminium_carbon_content + 
           cement_content_intensity * cement_carbon_content + 
           paper_content_intensity * paper_content_intensity) %>%
  
  mutate(eff_tax = carbon_tax/deflator  * emission_coef_pton ) %>%
  mutate(eff_tax_pos = ifelse(eff_tax >0, TRUE, FALSE))



final_df %>%
  select(prodcom_digit_8_code, 
         prodcom_digit_8_label, 
         deflator, 
         contains("_content_intensity")) %>%
  drop_na(deflator) %>%
  write.xlsx("prodcom_input_matlab.xlsx")

# Plot of the biais induced by sample selection

final_df %>%
  mutate(deflator_available = ifelse(is.na(deflator_used), FALSE, TRUE)) %>%
  select(prodcom_digit_8_code, prodcom_digit_8_label,
         deflator,
         deflator_used, 
         deflator_available, 
         contains("content_intensity"), 
         contains("carbon_content"),
         emission_coef_pton,
         eff_tax) %>%
  ggplot(aes(x = emission_coef_pton)
         ) + 
  geom_density() +
  geom_density(aes(color = deflator_available)) +
  theme_bw() + 
  xlab("Emission coefficient (using the average emissions values)")


# Plot the effective tax rate -----

final_df %>%
  mutate(deflator_available = ifelse(is.na(deflator_used), FALSE, TRUE)) %>%
  select(prodcom_digit_8_code, prodcom_digit_8_label,
         deflator,
         deflator_used, 
         deflator_available, 
         emission_coef_pton) %>%
  mutate(eff_tax_10 = 10 / deflator * emission_coef_pton, 
         eff_tax_25 = 25 / deflator * emission_coef_pton, 
         eff_tax_50 = 50 / deflator * emission_coef_pton, 
         eff_tax_75 = 75 / deflator * emission_coef_pton, 
         eff_tax_100 = 100 / deflator * emission_coef_pton) %>%
  pivot_longer(contains("eff_tax"), 
               names_to = "tax_level", 
               values_to = "eff_tax") %>%
  mutate(tax_level = str_sub(tax_level, -2)) %>%
  mutate(tax_level = ifelse(tax_level == "00", "100", tax_level
                            )) %>%
  mutate(across(tax_level, ~ paste0(.x, " €/t CO2eq"))) %>%
  ggplot(aes(x = eff_tax, 
             color = tax_level)) + 
  stat_ecdf() +
  xlim(0,1) +
  theme_bw() + 
  xlab("Effective tax rate")
