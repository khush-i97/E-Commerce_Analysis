# Data Model Documentation

## 1. Table Catalog

| Table Name | Description | Source | Row Count |
|------------|-------------|--------|-----------|
| Actuals Rank |  |  |  |
| Brand Status |  |  |  |
| C Missing Local SKU |  |  |  |
| C Missing TC |  |  |  |
| CEE_HTP_LE_Plans |  |  |  |
| CSS_Consumption |  |  |  |
| CT_Base |  |  |  |
| CT_DIM_Brand |  |  |  |
| CT_DIM_CategoryMapping |  |  |  |
| CT_DIM_Country |  |  |  |
| CT_DIM_Responder |  |  |  |
| CT_FACT_All_KPI |  |  |  |
| Calculation group |  |  |  |
| Core lookup |  |  |  |
| Currency |  |  |  |
| Currency Lookup |  |  |  |
| D Account |  |  |  |
| D Account Category |  |  |  |
| D Datatype |  |  |  |
| D Datatype Base |  |  |  |
| D Datatype Reference |  |  |  |
| D Date |  |  |  |
| D Market |  |  |  |
| D Market Category |  |  |  |
| D Page Information |  |  |  |
| D Pod System Only |  |  |  |
| D Product |  |  |  |
| D Ratio |  |  |  |
| D Source |  |  |  |
| D Timeserie |  |  |  |
| DIM_BrandMarket |  |  |  |
| DIM_Country |  |  |  |
| DIM_Date |  |  |  |
| Date List |  |  |  |
| DateTableTemplate_dac13467-a1aa-49ac-b30d-9806bdb60c69 |  |  |  |
| Engagement_Master |  |  |  |
| Exchange Rates |  |  |  |
| F Job Information |  |  |  |
| F Reconciling Items |  |  |  |
| F Sales |  |  |  |
| FY Period Mapping |  |  |  |
| Forecast Info |  |  |  |
| Forecast or Actual |  |  |  |
| HT_Clean_Source_Sell-Out |  |  |  |
| KASVD |  |  |  |
| KPI Definition |  |  |  |
| LocalDateTable_22cb4713-f4d7-42d9-989b-35884298de73 |  |  |  |
| LocalDateTable_3429052b-3fd9-40eb-b621-d8ca06070f45 |  |  |  |
| LocalDateTable_391d2f5b-82e0-4968-81b8-cb4cede57c42 |  |  |  |
| LocalDateTable_499062b9-316a-4931-9410-360c06531878 |  |  |  |
| LocalDateTable_5ffdc18e-72bc-4b92-b26d-d1bce385abfb |  |  |  |
| LocalDateTable_681f5efc-2622-4256-b685-7a8c4b6bbd42 |  |  |  |
| LocalDateTable_75473b53-2bf6-40d3-a0d5-29aed8473b90 |  |  |  |
| LocalDateTable_77688c5d-dd24-418b-ae13-f04bf1748700 |  |  |  |
| LocalDateTable_7a50f0b0-f04a-4f58-be60-ba4d9d056dcb |  |  |  |
| LocalDateTable_7ff71c68-363a-4bf8-a6c0-1c476b402d5f |  |  |  |
| LocalDateTable_80fbbf71-1b3e-4973-bf8f-78f4c68ba1ae |  |  |  |
| LocalDateTable_884cce44-7ba9-4b69-9b20-e3ff8d4970bb |  |  |  |
| LocalDateTable_a465266c-b5c8-4d14-91fe-c76ae5629a78 |  |  |  |
| LocalDateTable_b3734f61-882a-4a1c-bf86-600cd9cb213b |  |  |  |
| LocalDateTable_b3bfabbe-86a5-49da-ba6d-d99bd0f9a504 |  |  |  |
| LocalDateTable_bd21775f-c11a-4e03-8c12-b7093c07da41 |  |  |  |
| LocalDateTable_da8a3fc5-a868-4241-8711-b906773ce43a |  |  |  |
| LocalDateTable_e43a7498-7b1c-4a5c-83da-67991187df0c |  |  |  |
| LocalDateTable_f31b4e0f-979d-455c-b061-35d944f27a2f |  |  |  |
| LocalDateTable_f6868e92-ac5c-4d67-9c26-399e5f471052 |  |  |  |
| LocalDateTable_f862f110-8940-4c8c-a2c0-0aeaf5e3e2b7 |  |  |  |
| M Refresh |  |  |  |
| M Report URL |  |  |  |
| MSS ProductMarket |  |  |  |
| MSS Select Brand |  |  |  |
| MSS Select Date |  |  |  |
| MSS Select Forecast |  |  |  |
| MSS Select Geography |  |  |  |
| MSS Select Market Region |  |  |  |
| MSS Select Measure |  |  |  |
| MSS Select Measure Type |  |  |  |
| MSS Select Measure Unit |  |  |  |
| MSS Select Product |  |  |  |
| MSS Unrelated Date |  |  |  |
| MSS Unrelated Forecast |  |  |  |
| MSS Values |  |  |  |
| MSS Version Release Date |  |  |  |
| MSS factmss |  |  |  |
| NetSuite File Existence |  |  |  |
| S Amount Type Switch |  |  |  |
| S Measure Switch |  |  |  |
| act_account |  |  |  |
| act_account_loyalty_voucher_redemption |  |  |  |
| act_campaignemails_new |  |  |  |
| act_consumer_referral_code |  |  |  |
| act_orders |  |  |  |
| act_regdevices |  |  |  |
| active_doi_accounts |  |  |  |
| ecommerce |  |  |  |
| factmss (2) |  |  |  |
| factmss (3) |  |  |  |
| ga4_sessions |  |  |  |
| ga4_ua_engagement_rate |  |  |  |
| new_engagement_score |  |  |  |
| rep_newsletter_history |  |  |  |
| sm_pulzecare |  |  |  |

## 2. Column Details

### Actuals Rank

- **Actuals Rank** (int64)
- **Concat** (string)
- **Latest Actuals** (int64)
- **Min Rank** (int64)
- **Month** (string)
- **Rank** (int64)
- **VersionName** (string)

### Brand Status

- **Brand Status** (string)
- **Concat 2** (string)
- **Concat** (string)
- **Market** (string)
- **MeasureType** (string)
- **Product** (string)
- **Status** (string)

### C Missing Local SKU

- **Date ID** (int64)
- **Local SKU Code** (string)
- **Market ID** (int64)
- **Quantity** (double)
- **Revenue** (double)

### C Missing TC

- **DataType ID** (int64)
- **Date ID** (int64)
- **Product ID** (int64)
- **Volume** (double)

### CEE_HTP_LE_Plans

- **Date** (dateTime)
- **Metric** (string)
- **Value** (double)
- **bp_le** (string)
- **date_updated** (dateTime)
- **market_id** (string)
- **sub_metric** (string)

### CSS_Consumption

- **AGE** (int64)
- **Category** (string)
- **Country** (string)
- **Country_Key** (string)
- **DIM_Country.Country** (string)
- **GENDER** (int64)
- **Quantity** (int64)
- **User** (int64)
- **Weight** (int64)
- **Year** (int64)
- **Year_Ago** (string)
- **record** (string)

### CT_Base

- **AGE** (int64)
- **ResponderID** (int64)
- **WEIGHT** (double)
- **WeightAge** (double)

### CT_DIM_Brand

- **BrandCode** (string)
- **BrandCode_CountryCode** (string)
- **BrandName** (string)
- **Country Code** (string)
- **EVP** (string)
- **ImperialBrandOrder** (int64)
- **IsImperialBrand** (string)
- **LocalBrandLookup** (string)
- **LocalCode** (string)
- **LocalCodeWith0s** (string)
- **Manufact** (string)
- **MasterBrand** (string)
- **SourceCategoryCode** (string)
- **UniqueBrandCode** (string)
- **UniqueCode** (string)

### CT_DIM_CategoryMapping

- **CategoryCode** (string)
- **CategoryDisplayLabel** (string)
- **CategoryFullName** (string)
- **CategoryLabel** (string)
- **CategoryOrder** (string)
- **FlagActive** (int64)
- **MainCategoryCode** (string)
- **MainCategoryDisplayLabel** (string)
- **MainCategoryLabel** (string)
- **MainCategoryOrder** (int64)

### CT_DIM_Country

- **Code** (int64)
- **Label** (string)

### CT_DIM_Responder

- **AGE** (int64)
- **Date** (dateTime)
- **Date_SortKey_CT** (string)
- **Gender** (int64)
- **Region** (string)
- **RegionDisplayLabel** (string)
- **ResponderID** (int64)
- **hidCountry** (int64)

### CT_FACT_All_KPI

- **BrandCode_CountryCode** (string)
- **CategoryCode** (string)
- **Date** (dateTime)
- **KPI** (string)
- **Order** (int64)
- **ResponderID** (int64)

### Calculation group

- **Calculation group column** (string)
- **Ordinal** (int64)

### Core lookup

- **Brand Status** (string)
- **Brand** (string)
- **Category** (string)
- **Concat 2** (string)
- **Concat** (string)
- **Market** (string)
- **Status** (string)

### Currency

- **CurrencyName** (string)
- **CurrencySymbol** (string)
- **Local Currency** (string)

### Currency Lookup

- **Currency** (string)
- **Market** (string)

### D Account

- **Account_ID** (int64)
- **Controller_Account** (string)
- **Detail_Account_Code** (string)
- **Detail_Account_Name** (string)
- **Detail_Account_Short_Name** (string)
- **Dim_Account_Key** (string)
- **Dim_Account_Level** (string)
- **Operational_Investment_Category** (string)
- **Summary_Account_Code** (string)
- **Summary_Account_NGP_Code** (string)
- **Summary_Account_NGP_Code_L0** (string)
- **Summary_Account_NGP_Code_L1** (string)
- **Summary_Account_NGP_Code_L2** (string)
- **Summary_Account_NGP_Code_L3** (string)
- **Summary_Account_NGP_Name** (string)
- **Summary_Account_NGP_Name_L0** (string)
- **Summary_Account_NGP_Name_L1** (string)
- **Summary_Account_NGP_Name_L2** (string)
- **Summary_Account_NGP_Name_L3** (string)
- **Summary_Account_NGP_Short_Name** (string)
- **Summary_Account_NGP_Short_Name_L0** (string)
- **Summary_Account_NGP_Short_Name_L1** (string)
- **Summary_Account_NGP_Short_Name_L2** (string)
- **Summary_Account_NGP_Short_Name_L3** (string)
- **Summary_Account_Name** (string)
- **Summary_Account_Short_Name** (string)

### D Account Category

- **Account_Category_Code** (string)
- **Account_Category_Level** (string)
- **Account_Category_Main** (string)
- **Account_Category_Name** (string)
- **Account_Category_Operator** (int64)
- **Account_Category_Short_Name** (string)
- **Account_Category_Sort_Code** (string)
- **Account_ID** (int64)
- **Gross Revenue Breakdown** (string)
- **Operating Profit Breakdown** (string)

### D Datatype

- **Anaplan_Rec_Ind** (boolean)
- **DataType_FAD_Flag** (string)
- **DataType_FAD_SortKey** (string)
- **DataType_ID** (int64)
- **DataType_REC_Flag** (string)
- **Data_Type_Code** (string)
- **Data_Type_Identifier_Code** (string)
- **Data_Type_Identifier_Name** (string)
- **Data_Type_Identifier_Short_Name** (string)
- **Data_Type_Name** (string)
- **Data_Type_Short_Name** (string)
- **Data_Type_Status_Name** (string)
- **Data_Type_Text_Key** (string)
- **Dim_DataType_Key** (string)
- **Dim_DataType_Level** (string)
- **FAD_Ind** (boolean)
- **GSM_Rec_Ind** (boolean)
- **SFPD_Ind** (boolean)

### D Datatype Base

- **D_Datatype_Base_Anaplan_Rec_Ind** (boolean)
- **D_Datatype_Base_Category** (string)
- **D_Datatype_Base_Code** (string)
- **D_Datatype_Base_FAD_Ind** (boolean)
- **D_Datatype_Base_GSM_Rec_Ind** (boolean)
- **D_Datatype_Base_ID** (int64)
- **D_Datatype_Base_Name** (string)
- **D_Datatype_Base_Name_Short** (string)
- **D_Datatype_Base_SFPD_Ind** (boolean)
- **Sorting_Key** (string)

### D Datatype Reference

- **D_Datatype_Reference_Anaplan_Rec_Ind** (boolean)
- **D_Datatype_Reference_Category** (string)
- **D_Datatype_Reference_Code** (string)
- **D_Datatype_Reference_FAD_Ind** (boolean)
- **D_Datatype_Reference_GSM_Rec_Ind** (boolean)
- **D_Datatype_Reference_ID** (int64)
- **D_Datatype_Reference_Name** (string)
- **D_Datatype_Reference_Name_Short** (string)
- **D_Datatype_Reference_SFPD_Ind** (boolean)
- **Sorting_Key** (string)

### D Date

- **BD_Number_Current** (int64)
- **BD_Number_In_Month** (int64)
- **BD_Number_Running** (int64)
- **Calendar_Month_Code** (string)
- **Calendar_Month_End_Period** (dateTime)
- **Calendar_Month_Number** (string)
- **Calendar_Month_Of_Year_Name** (string)
- **Calendar_Month_Start_Period** (dateTime)
- **Calendar_Quarter_Code** (string)
- **Calendar_Quarter_End_Period** (string)
- **Calendar_Quarter_Of_Year_Name** (string)
- **Calendar_Quarter_Start_Period** (string)
- **Calendar_Year_Code** (string)
- **Calendar_Year_End_Period** (string)
- **Calendar_Year_Start_Period** (string)
- **Current_Previous_Month_Ind** (string)
- **Current_Previous_Year_Ind** (string)
- **Date_ID** (int64)
- **Day_As_Date** (dateTime)
- **Day_Code** (string)
- **Day_Date_Number** (string)
- **Day_Of_Month_Name** (string)
- **Day_Of_Week_Name** (string)
- **Dim_Date_Key** (string)
- **Dim_Date_Level** (string)
- **FY_Month_Ago** (double)
- **FY_Year_Ago** (double)
- **Financial_Closing_Year** (string)
- **Financial_Period_Code** (string)
- **Financial_Period_End_Period** (dateTime)
- **Financial_Period_Of_Financial_Year_Name** (string)
- **Financial_Period_Of_Financial_Year_Number** (string)
- **Financial_Period_Of_Financial_Year_Short_Name** (string)
- **Financial_Period_Start_Period** (dateTime)
- **Financial_Quarter_Code** (string)
- **Financial_Quarter_End_Period** (string)
- **Financial_Quarter_Of_Financial_Year_Name** (string)
- **Financial_Quarter_Of_Financial_Year_Short_Name** (string)
- **Financial_Quarter_Start_Period** (string)
- **Financial_Reporting_Year** (string)
- **Financial_Year_Code** (string)
- **Financial_Year_End_Period** (string)
- **Financial_Year_Name** (string)
- **Financial_Year_Start_Period** (string)
- **IsWorkingDay** (int64)
- **Past_Future_Ind** (string)
- **Rolling_18M_Ind** (string)
- **Rolling_18M_SortKey** (string)

### D Market

- **Business_Area_Code** (string)
- **Business_Area_Name** (string)
- **Business_Area_Short_Name** (string)
- **Cluster_Code** (string)
- **Cluster_Name** (string)
- **Cluster_Short_Name** (string)
- **Controller_Code** (string)
- **Dim_Market_Key** (string)
- **Dim_Market_Level** (string)
- **Division_Code** (string)
- **Division_Name** (string)
- **Division_Short_Name** (string)
- **External_Reporting_Segment_Name** (string)
- **External_Reporting_Segment_Short_Name** (string)
- **Focus_Market_Code** (string)
- **Focus_Market_Name** (string)
- **Focus_Market_Short_Name** (string)
- **ISO Country New Market** (string)
- **ISO_Country_Code_Alpha3** (string)
- **ISO_Country_Code_Num3** (string)
- **ISO_Country_Name** (string)
- **ISO_Country_Short_Name** (string)
- **ISO_Currency_Name** (string)
- **Market_All** (string)
- **Market_Code** (string)
- **Market_Constraint_Name** (string)
- **Market_Constraint_Short_Name** (string)
- **Market_Derivation_Rule_Name** (string)
- **Market_Duty_Type_Name** (string)
- **Market_Duty_Type_Short_Name** (string)
- **Market_ID** (int64)
- **Market_Name** (string)
- **Market_Short_Name** (string)
- **New_Market_Category** (string)
- **New_Market_Category_Code** (string)
- **New_Market_Category_Name** (string)
- **New_Market_Code** (string)
- **New_Market_Name** (string)
- **New_Market_Short_Name** (string)
- **New_Market_Sub_Category_Code** (string)
- **New_Market_Sub_Category_Name** (string)
- **Ngp_Channel_Name** (string)
- **Reporting_Market_Code** (string)
- **Reporting_Market_FootPrint** (string)
- **Reporting_Market_Name** (string)
- **Reporting_Market_Short_Name** (string)
- **Sales_Territory_Name** (string)
- **Sales_Territory_Short_Name** (string)
- **Tier_Code** (string)
- **Tier_Name** (string)

### D Market Category

- **Market_Category** (string)
- **Market_Category_Sort_Key** (string)
- **Market_ID** (int64)

### D Page Information

- **Page Description** (string)
- **Page ID** (int64)
- **Page Name** (string)
- **Page Subtitle** (string)
- **Page Title** (string)

### D Pod System Only

- **Brand Family Code** (string)
- **Brand Family Short Name** (string)
- **Pod System Only** (string)

### D Product

- **APO_Location_Name** (string)
- **APO_Location_Short_Name** (string)
- **APO_Location_Type_Name** (string)
- **APO_Procurement_Type** (string)
- **Asset_Brand** (string)
- **Brand_Family_Code** (string)
- **Brand_Family_Name** (string)
- **Brand_Family_Short_Name** (string)
- **Brand_House_Code** (string)
- **Brand_House_Name** (string)
- **Brand_House_Short_Name** (string)
- **Brand_Pack_Variant_Code** (string)
- **Brand_Pack_Variant_Name** (string)
- **Brand_Pack_Variant_Short_Name** (string)
- **Brand_Segment** (string)
- **Brand_Sub_Family_Code** (string)
- **Brand_Sub_Family_Name** (string)
- **Brand_Sub_Family_Short_Name** (string)
- **Brand_Variant_Code** (string)
- **Brand_Variant_Name** (string)
- **Brand_Variant_Short_Name** (string)
- **Chassis_Brand** (string)
- **Cigar_Product_Segment_Name** (string)
- **Cigar_Product_Segment_Short_Name** (string)
- **Consumable_Content_Type_Name** (string)
- **Consumable_Content_Type_Short_Name** (string)
- **Conversion_Factor_Stick_Equivalent** (string)
- **Dim_Product_Key** (string)
- **Dim_Product_Level** (string)
- **Frango_Product_Group_Code** (string)
- **Frango_Product_Group_Name** (string)
- **Frango_Product_Group_Short_Name** (string)
- **ITG_Unit_Of_Measure_Code** (string)
- **ITG_Unit_Of_Measure_Code_Collection** (string)
- **Liquid_Recipe_Owner_Name** (string)
- **Manufacturer_Identifier_Brand_Sub_Family** (string)
- **Manufacturer_Identifier_Brand_Variant** (string)
- **Market_Constraint_Name** (string)
- **Market_Constraint_Short_Name** (string)
- **PR_Reporting_Unit_Of_Measure** (string)
- **PR_Reporting_Unit_Of_Measure_Conversion_Factor** (int64)
- **Product_All** (string)
- **Product_Device_Content** (int64)
- **Product_Group_Code** (string)
- **Product_Group_In_OCA** (string)
- **Product_Group_In_PIP** (string)
- **Product_Group_Name** (string)
- **Product_Group_Short_Name** (string)
- **Product_ID** (int64)
- **Product_Liquid_Content_ML** (string)
- **Product_Type_Name** (string)
- **Production_Planning_Strategy_Name** (string)
- **Production_Planning_Strategy_Short_Name** (string)
- **Reporting_Product_Group_Short_Name** (string)
- **Retail_Blend_Type_In_OCA** (string)
- **Retail_Blend_Type_Name** (string)
- **Retail_Blend_Type_Short_Name** (string)
- **Retail_Bundle_Shape_Type_MFG_Code** (string)
- **Retail_Bundle_Shape_Type_Name** (string)
- **Retail_Bundle_Shape_Type_Short_Name** (string)
- **Retail_Bundle_Shape_Type_Status_Type_Code** (string)
- **Retail_Diameter_Type_Name** (string)
- **Retail_Filter_Type_Name** (string)
- **Retail_Filter_Type_Short_Name** (string)
- **Retail_Flavour_Type_Name** (string)
- **Retail_Flavour_Type_Short_Name** (string)
- **Retail_Flavour_Type_Status_Type_Code** (string)
- **Retail_Length_Band_MFG_Code** (string)
- **Retail_Length_Band_Name** (string)
- **Retail_Length_Band_Short_Name** (string)
- **Retail_Length_Band_Status_Type_Code** (string)
- **Retail_Pack_Consumer_Unit_Content** (double)
- **Retail_Pack_Content** (double)
- **Retail_Pack_Contents_Type_Name** (string)
- **Retail_Pack_Contents_Type_Short_Name** (string)
- **Retail_Pack_Type_Name** (string)
- **Retail_Pack_Type_Short_Name** (string)
- **Retail_Pack_Type_Status_Type_Code** (string)
- **Retail_Product_Content_Type_Name** (string)
- **Retail_Product_Content_Type_Short_Name** (string)
- **Retail_Product_Generation_Name** (string)
- **Retail_Product_Generation_Short_Name** (string)
- **Retail_SKU_Code** (string)
- **Retail_SKU_Forecast_Type_Name** (string)
- **Retail_SKU_Name** (string)
- **Retail_SKU_Outer_Contents** (double)
- **Retail_SKU_Short_Name** (string)
- **Retail_SKU_Status_Name** (string)
- **Retail_SKU_Status_Short_Name** (string)
- **Retail_Taste_Type_In_OCA** (string)
- **Retail_Taste_Type_Name** (string)
- **Retail_Taste_Type_Short_Name** (string)
- **Retail_Taste_Type_Status_Type_Code** (string)
- **Retail_Wrap_Type_Name** (string)
- **Retail_Wrap_Type_Short_Name** (string)
- **Retail_Wrap_Type_Status_Type_Code** (string)
- **Standard_SKU_Code** (string)
- **Standard_SKU_Name** (string)
- **Standard_SKU_Short_Name** (string)
- **System_Type_Name** (string)
- **System_Type_Short_Name** (string)
- **Trademark_Owner_Name** (string)
- **Trademark_Owner_Short_Name** (string)
- **Volume_Type** (string)

### D Ratio

- **Ratio_Denominator** (string)
- **Ratio_Description** (string)
- **Ratio_ID** (int64)
- **Ratio_Numerator** (string)
- **Ratio_Sortkey** (string)

### D Source

- **Source_Category** (string)
- **Source_ID** (int64)
- **Source_Name** (string)

### D Timeserie

- **Calculate_Ind** (int64)
- **Current_Previous_Month_Ind** (string)
- **Current_Previous_Year_Ind** (string)
- **Date_ID** (int64)
- **Date_Label** (string)
- **Date_Sortkey** (int64)
- **Financial_Year_Label** (string)
- **Financial_Year_Selection** (string)
- **Past_Future_Ind** (string)
- **Period_Desc** (string)
- **Period_Group** (string)
- **Period_Group_Desc** (string)
- **Period_Label** (string)
- **Period_Selection** (string)
- **Period_Sortkey** (int64)
- **Period_Year_Month_Desc** (string)
- **Timeserie_Name** (string)
- **Timeserie_S_Name** (string)
- **Timeserie_Sortkey** (int64)
- **Timeserie_Sortkey_Granular** (string)
- **Timeserie_Type** (string)

### DIM_BrandMarket

- **Brand** (string)
- **MarketID** (string)
- **SurveyID** (string)
- **U_Market_ID** (string)

### DIM_Country

- **Brand** (string)
- **CSS_Key** (string)
- **CT_Country** (string)
- **Country currency** (string)
- **Country** (string)
- **Currency** (string)
- **Ex_rate** (double)
- **Finance_Country** (string)
- **HT_Clean_Source_Country** (string)
- **MSS_Key** (string)
- **Market_ID** (string)
- **Name** (string)

### DIM_Date

- **CALENDAR_DATE** (dateTime)
- **CAL_DAY_IN_MONTH** (int64)
- **CAL_DAY_IN_WEEK** (string)
- **CAL_DAY_IN_WEEK_NO** (int64)
- **CAL_DAY_IN_YEAR** (int64)
- **CAL_MONTH** (int64)
- **CAL_MONTH_END** (dateTime)
- **CAL_MONTH_NAME** (string)
- **CAL_MONTH_NO** (int64)
- **CAL_MONTH_START** (dateTime)
- **CAL_MTH_TRADING_DAYS** (int64)
- **CAL_QUARTER** (int64)
- **CAL_QUARTER_NO** (int64)
- **CAL_SEMESTER** (int64)
- **CAL_SEMESTER_NO** (int64)
- **CAL_TRADING_DAYS_MTD** (int64)
- **CAL_WEEK_END** (dateTime)
- **CAL_WEEK_IN_YEAR** (int64)
- **CAL_WEEK_START** (dateTime)
- **CAL_YEAR** (int64)
- **CAL_YEAR_END** (dateTime)
- **CAL_YEAR_START** (dateTime)
- **DIM_DATE_KEY** (int64)
- **DIM_DateSortKey_for_timeseries** (string)
- **DSS_UPDATE_TIME** (dateTime)
- **Date** (string)
- **Date_Abs_Mo_Ago** (int64)
- **Date_Fin_Month_Ago** (int64)
- **Date_Fin_Year_Ago** (int64)
- **Date_Quarter_Text** (string)
- **Date_Year_Ago** (int64)
- **Day Rank Desc** (int64)
- **FINANCIAL_DATE** (dateTime)
- **FIN_DAY_IN_MONTH** (int64)
- **FIN_DAY_IN_WEEK** (string)
- **FIN_DAY_IN_WEEK_NO** (int64)
- **FIN_DAY_IN_YEAR** (int64)
- **FIN_MONTH** (int64)
- **FIN_MONTH_END** (dateTime)
- **FIN_MONTH_NAME** (string)
- **FIN_MONTH_NO** (int64)
- **FIN_MONTH_START** (dateTime)
- **FIN_MTH_TRADING_DAYS** (int64)
- **FIN_PERIOD** (string)
- **FIN_QUARTER** (int64)
- **FIN_QUARTER_NO** (int64)
- **FIN_SEMESTER** (int64)
- **FIN_SEMESTER_NO** (int64)
- **FIN_TRADING_DAYS_MTD** (int64)
- **FIN_WEEKS_IN_PERIOD** (int64)
- **FIN_WEEK_END** (dateTime)
- **FIN_WEEK_IN_MONTH** (int64)
- **FIN_WEEK_IN_YEAR** (int64)
- **FIN_WEEK_START** (dateTime)
- **FIN_YEAR** (int64)
- **FIN_YEAR_END** (dateTime)
- **FIN_YEAR_START** (dateTime)
- **FIN_YEAR_TXT** (string)
- **HOLIDAY_DESC** (string)
- **HOLIDAY_FLAG** (string)
- **Month Rank Desc** (int64)
- **Month_Year** (string)
- **NON_WORKING_DESC** (string)
- **NON_WORKING_TYPE** (string)
- **TRADING_DAYS_SO_FAR** (int64)
- **TRADING_DAY_FLAG** (string)
- **WEEK_DAY_FLAG** (string)
- **WEEK_END_FLAG** (string)
- **Week Day** (int64)
- **Week Number Name** (string)
- **Week Rank Desc** (int64)
- **YYYYMM** (int64)
- **YYYYWW** (int64)

### Date List

- **Month** (string)

### DateTableTemplate_dac13467-a1aa-49ac-b30d-9806bdb60c69

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### Engagement_Master

- **ChoiceFlag** (string)
- **ChoiceId** (string)
- **KPIID** (int64)
- **KPIName** (string)
- **MarketID** (string)
- **QuestionID** (string)
- **SurveyID** (int64)

### Exchange Rates

- **Exchange Rate** (double)
- **FY** (string)
- **Local Currency** (string)

### F Job Information

- **Job Status Code** (int64)
- **Job Status** (string)
- **Job** (string)
- **Last Succesful Run Date Time UTC** (string)
- **Next Run Date** (string)
- **Run Date Time UTC** (string)
- **Status Description** (string)

### F Reconciling Items

- **Item Description** (string)
- **Item ID** (int64)

### F Sales

- **Account_ID** (int64)
- **Datatype_ID** (int64)
- **Date_ID** (int64)
- **Local_Currency_ID** (int64)
- **Market_ID** (int64)
- **Product_ID** (int64)
- **Source_ID** (int64)
- **Value_Reporting_Currency** (double)

### FY Period Mapping

- **Month** (string)
- **Period** (string)

### Forecast Info

- **Column1** (string)
- **ForecastCreationDate** (string)
- **Latest Actuals** (int64)
- **Latest BP** (string)
- **Latest LE** (string)
- **ModifiedOn Date** (dateTime)
- **ModifiedOn** (int64)
- **Sort** (double)
- **Submission Type** (string)
- **Version Name** (string)
- **Version Short Name** (string)
- **_1** (string)

### Forecast or Actual

- **Actuals Status** (string)

### HT_Clean_Source_Sell-Out

- **Base** (string)
- **Brand.Family** (string)
- **CB_Non.CB** (string)
- **Date.M.Y** (dateTime)
- **Date.Rank.Market** (int64)
- **Date.Rank** (int64)
- **Date** (dateTime)
- **Date_Abs_Mo_Ago** (int64)
- **Date_Abs_Qtr_Ago** (int64)
- **Date_FY_Month** (int64)
- **Date_FY_Year** (int64)
- **Date_FY_Year_Ago** (int64)
- **Date_Month_Text** (string)
- **Date_Quarter_Text** (string)
- **Date_Year_Ago** (int64)
- **Emerging.Type.Rank.3m** (int64)
- **Emerging.Type.Volume.change.3m** (double)
- **Emerging.Type.Volume.pecent.change.3m** (double)
- **Exclusion** (string)
- **Flavour** (string)
- **Fruit_Non.Fruit** (string)
- **HC.Share** (double)
- **Harmonized.Brand.Name** (string)
- **Hints** (string)
- **Holding.Company** (string)
- **Is.CFB** (string)
- **Is.Divice** (string)
- **Is.New.Launch** (string)
- **Market.Emerging.Type.Volume.change.3m** (string)
- **Market.Short** (string)
- **Market** (string)
- **Menthol_Non.Menthol** (string)
- **Monthly.Volume** (double)
- **Notes** (string)
- **Numeric.Distribution** (string)
- **Open.Closed** (string)
- **Project** (string)
- **Raw.Value** (double)
- **Raw.Volume** (double)
- **SKU.Tenure** (int64)
- **SKU** (string)
- **SKU.min.date.idx** (int64)
- **SortKey_for_timeseries** (string)
- **Source.Name** (string)
- **Start.date** (dateTime)
- **Stick.multiple** (int64)
- **Type.Rank** (int64)
- **Type** (string)
- **Value** (double)
- **Volume** (double)
- **Weighted.Distribution** (string)

### KASVD

- **Date Key** (int64)
- **Date** (dateTime)
- **Manufacturer** (string)
- **Market_ID** (string)
- **Month** (string)
- **Network** (string)
- **Product** (string)
- **ProductGroup** (string)
- **WhiteStick** (double)
- **Year** (int64)

### KPI Definition

- **Calculation** (string)
- **Definition** (string)
- **Target KPI Name** (string)
- **Target Page** (string)
- **Target Section** (string)

### LocalDateTable_22cb4713-f4d7-42d9-989b-35884298de73

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_3429052b-3fd9-40eb-b621-d8ca06070f45

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_391d2f5b-82e0-4968-81b8-cb4cede57c42

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_499062b9-316a-4931-9410-360c06531878

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_5ffdc18e-72bc-4b92-b26d-d1bce385abfb

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_681f5efc-2622-4256-b685-7a8c4b6bbd42

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_75473b53-2bf6-40d3-a0d5-29aed8473b90

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_77688c5d-dd24-418b-ae13-f04bf1748700

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_7a50f0b0-f04a-4f58-be60-ba4d9d056dcb

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_7ff71c68-363a-4bf8-a6c0-1c476b402d5f

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_80fbbf71-1b3e-4973-bf8f-78f4c68ba1ae

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_884cce44-7ba9-4b69-9b20-e3ff8d4970bb

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_a465266c-b5c8-4d14-91fe-c76ae5629a78

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_b3734f61-882a-4a1c-bf86-600cd9cb213b

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_b3bfabbe-86a5-49da-ba6d-d99bd0f9a504

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_bd21775f-c11a-4e03-8c12-b7093c07da41

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_da8a3fc5-a868-4241-8711-b906773ce43a

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_e43a7498-7b1c-4a5c-83da-67991187df0c

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_f31b4e0f-979d-455c-b061-35d944f27a2f

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_f6868e92-ac5c-4d67-9c26-399e5f471052

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### LocalDateTable_f862f110-8940-4c8c-a2c0-0aeaf5e3e2b7

- **Date** (dateTime)
- **Day** (int64)
- **Month** (string)
- **MonthNo** (int64)
- **Quarter** (string)
- **QuarterNo** (int64)
- **Year** (int64)

### M Refresh

- **Last_Refresh** (string)
- **Object** (string)

### M Report URL

- **Report** (string)
- **URL** (string)

### MSS ProductMarket

- **MSSPM_PreKey** (string)
- **Market** (string)
- **MarketKey** (string)
- **Product** (string)
- **ProductKey** (string)

### MSS Select Brand

- **Brand** (string)

### MSS Select Date

- **End Date Key - Value** (int64)
- **EndDateKey - Start of Month** (string)
- **EndDateKey** (string)
- **External FY & HY:** (string)
- **External FY & Qtr:** (string)
- **External FY:** (string)
- **FY Period** (string)
- **Internal FY & HY:** (string)
- **Internal FY & Qtr:** (string)
- **Internal FY:** (string)
- **Month Rank** (int64)
- **Month** (dateTime)
- **Previous Internal FY** (string)
- **YearMonthCode** (int64)

### MSS Select Forecast

- **DateRank** (int64)
- **Latest BP** (string)
- **Latest LE** (string)
- **Sort** (double)
- **Version Name** (string)
- **Version Short Name** (string)
- **VersionDate** (dateTime)
- **VersionName** (string)
- **Versions** (string)

### MSS Select Geography

- **Cluster** (string)
- **Division / Tier** (string)
- **External Division FY23** (string)
- **External Division** (string)
- **INT / EXT** (string)
- **Internal Division** (string)
- **Market** (string)
- **Tier** (int64)

### MSS Select Market Region

- **Market Region** (string)

### MSS Select Measure

- **Measure** (string)
- **Measures** (string)

### MSS Select Measure Type

- **Measure Type** (string)

### MSS Select Measure Unit

- **Measure Unit** (string)
- **VolumeUnitOfMeasureKey** (string)

### MSS Select Product

- **Product** (string)

### MSS Unrelated Date

- **EndDateKey - Start of Mont** (string)
- **EndDateKey** (string)
- **External FY & HY:** (string)
- **External FY & Qtr:** (string)
- **External FY:** (string)
- **Internal FY & HY:** (string)
- **Internal FY & Qtr:** (string)
- **Internal FY:** (string)
- **Month Rank** (int64)
- **Month** (dateTime)
- **Previous Internal FY** (string)
- **YearMonthCode** (int64)

### MSS Unrelated Forecast

- **EndDateKey** (string)
- **MaxActualDate** (string)
- **SUBMISSION DEADLINE** (dateTime)
- **VERSION NAME** (string)
- **VERSION SCENARIO** (string)
- **VERSION SHORT NAME** (string)

### MSS Values

- **a** (string)

### MSS Version Release Date

- **Release Date** (string)

### MSS factmss

- **Actuals Status** (string)
- **Concat 2** (string)
- **Exchange Rate** (double)
- **Local Currency** (string)
- **MSSPM_PreKey** (string)
- **Market** (string)
- **MarketRegion** (string)
- **Measure Type 2** (string)
- **Measure** (string)
- **MeasureType** (string)
- **MeasureUnit** (string)
- **MeasureValue** (double)
- **Month** (string)
- **Product** (string)
- **Rank** (int64)
- **Value Converted** (double)
- **Version & Month Concat** (string)
- **Version Market Concat** (string)
- **VersionName** (string)

### NetSuite File Existence

- **File_Per_Day** (int64)
- **NetSuite File Availability** (string)

### S Amount Type Switch

- **Amount Type Switch Code** (string)
- **Amount Type Switch Desc** (string)
- **Amount Type Switch ID** (int64)

### S Measure Switch

- **Measure_Switch_Code** (string)
- **Measure_Switch_Description** (string)
- **Measure_Switch_Option** (string)

### act_account

- **SourceOfRegistration** (string)
- **double_opt_in** (boolean)
- **ecosystem_entry_date** (dateTime)
- **ecosystem_user_type** (string)
- **inceptioninfo_purchasedproduct** (string)
- **is_converted** (string)
- **market_id** (string)
- **purchasedproduct** (string)
- **registered_status** (boolean)
- **sourceid** (string)
- **uuid** (string)
- **valid** (boolean)

### act_account_loyalty_voucher_redemption

- **RedemptionMode** (string)
- **market_id** (string)
- **redemptiondate** (dateTime)
- **status** (string)
- **voucher_createdat** (dateTime)
- **voucher_id** (string)
- **voucher_id_index** (int64)

### act_campaignemails_new

- **Market_ID** (string)
- **TemplateID** (string)
- **bounceddate** (dateTime)
- **clickeddate** (dateTime)
- **droppeddate** (dateTime)
- **id_index** (string)
- **senddate** (dateTime)
- **spamreportdate** (dateTime)

### act_consumer_referral_code

- **CreatedAt** (dateTime)
- **RAF_unique_referral_code** (string)
- **insertdate** (dateTime)
- **updatedate** (dateTime)

### act_orders

- **First_Order_Date** (dateTime)
- **Is First Purchase** (int64)
- **OrderType** (string)
- **Referral Discount Code** (string)
- **UUID** (string)
- **brand** (string)
- **deliveredat** (dateTime)
- **discountCode** (string)
- **market_id** (string)
- **order_Status** (string)
- **order_date** (dateTime)
- **shippedat** (dateTime)
- **sourceid** (string)

### act_regdevices

- **Brand** (string)
- **Country** (string)
- **Date** (dateTime)
- **Market_ID** (string)
- **TotalRegDevices** (int64)

### active_doi_accounts

- **Activity** (string)
- **Date_field** (dateTime)
- **Market_ID** (string)
- **uuid** (string)

### ecommerce

- **Market_ID** (string)
- **SessionId** (int64)
- **Visits** (int64)
- **eCommerceAction** (string)
- **process_date** (dateTime)

### factmss (2)

- **Actuals Status** (string)
- **Concat** (string)
- **Exchange Rate** (double)
- **Local Currency** (string)
- **Market** (string)
- **MarketRegion** (string)
- **Measure Type 2** (string)
- **Measure** (string)
- **MeasureType** (string)
- **MeasureUnit** (string)
- **MeasureValue** (double)
- **Month** (string)
- **Product** (string)
- **Value Converted** (double)
- **Version Market Concat** (string)
- **VersionName** (string)

### factmss (3)

- **Actuals Status** (string)
- **Concat** (string)
- **Exchange Rate** (double)
- **Local Currency** (string)
- **Market** (string)
- **MarketRegion** (string)
- **Measure** (string)
- **MeasureType** (string)
- **MeasureUnit** (string)
- **MeasureValue** (double)
- **Month** (string)
- **Product** (string)
- **Value Converted** (double)
- **Version & Month Concat** (string)
- **VersionName** (string)

### ga4_sessions

- **Market_ID** (string)
- **event_name** (string)
- **events_params_transaction_id_index** (string)
- **id_index** (int64)
- **process_date** (dateTime)

### ga4_ua_engagement_rate

- **Market_ID** (string)
- **engaged_sessions** (int64)
- **process_date** (dateTime)
- **sessions** (int64)

### new_engagement_score

- **Market_ID** (string)
- **brand** (string)
- **index** (string)
- **market** (string)
- **weight** (int64)

### rep_newsletter_history

- **email_optin** (boolean)
- **newsletter_opt_in** (boolean)
- **phone_optin** (boolean)
- **sms_optin** (boolean)
- **sourceid** (string)

### sm_pulzecare

- **DateCreated** (dateTime)
- **QuestionId** (string)
- **SurveyID** (string)
- **market_id** (string)
- **normalized_answer** (double)

## 3. Measures

| Measure Name | Table | DAX Expression |
|--------------|-------|----------------|
| CY_AvgCons | CSS_Measures | C A L C U L A T E ( A V E R A G E ( C S S _ C o n s u m p t i o n [ Q u a n t i t y ] ) , C S S _ C o n s u m p t i o n [ U s e r ] = 1 , C S S _ C o n s u m p t i o n [ Y e a r _ A g o ] = " 0 " ) |
| CY_incidence | CSS_Measures | VAR _UserBase = CALCULATE(SUM(CSS_Consumption[Weight]), CSS_Consumption[User] = 1, CSS_Consumption[Year_Ago] = "0") VAR _base = CALCULATE(SUM(CSS_Consumption[Weight]), CSS_Consumption[Year_Ago] = "0") RETURN DIVIDE(_UserBase, _base) |
| PY_AvgCons | CSS_Measures | VAR _py = CALCULATE(AVERAGE(CSS_Consumption[Quantity]), CSS_Consumption[User] = 1, CSS_Consumption[Year_Ago] = "1") VAR _cy = [CY_AvgCons] RETURN DIVIDE((_cy - _py), _py) |
| PY_Incidence | CSS_Measures | R O U N D ( [ P Y _ I n c i d e n c e _ C o l o r ] * 1 0 0 , 0 ) & " p p " |
| PY_Incidence_Color | CSS_Measures | VAR _UserBase = CALCULATE(SUM(CSS_Consumption[Weight]), CSS_Consumption[User] = 1, CSS_Consumption[Year_Ago] = "1") VAR _base = CALCULATE(SUM(CSS_Consumption[Weight]), CSS_Consumption[Year_Ago] = "1") VAR _py = DIVIDE(_UserBase, _base) VAR _cy = [CY_incidence] RETURN (_cy - _py) |
| Year_Title | CSS_Measures | C A L C U L A T E ( A V E R A G E ( C S S _ C o n s u m p t i o n [ Y e a r ] ) , C S S _ C o n s u m p t i o n [ Y e a r _ A g o ] = " 0 " ) |
| CbyA | CT_Measures | D I V I D E ( [ W e i g h t e d B r a n d F u n n e l P c t C o n s i ] , [ W e i g h t e d B r a n d F u n n e l P c t P A ] ) |
| Max_Date_CT | CT_Measures | VAR _date = CALCULATE(MAX(CT_DIM_Responder[Date]), ALL()) RETURN FORMAT(_date, "mmm") & "-" & YEAR(_date) |
| MobyR | CT_Measures | D I V I D E ( [ W e i g h t e d B r a n d F u n n e l P c t M O ] , [ W e i g h t e d B r a n d F u n n e l P c t R e ] ) |
| PFY_Weighted_BF_Pct_Consi_Diff | CT_Measures | IF(ISBLANK([PFY_Weighted_BF_Pct_Consi_Diff_c]), "--", CONVERT(ROUND([PFY_Weighted_BF_Pct_Consi_Diff_c]* 100,0),STRING)  & "pp") |
| PFY_Weighted_BF_Pct_Consi_Diff_c | CT_Measures | VAR _pfy_sa =      CALCULATE([Weighted BrandFunnel Pct Consi],     ALL(DIM_Date), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE])     ) VAR _cfy_sa = CALCULATE([Weighted BrandFunnel Pct Consi]) RETURN _cfy_sa - _pfy_sa |
| PFY_Weighted_BF_Pct_MO_Diff | CT_Measures | IF(ISBLANK([PFY_Weighted_BF_Pct_MO_Diff_c]), "--", CONVERT(ROUND([PFY_Weighted_BF_Pct_MO_Diff_c]* 100,0),STRING)  & "pp") |
| PFY_Weighted_BF_Pct_MO_Diff_c | CT_Measures | VAR _pfy_sa =      CALCULATE([Weighted BrandFunnel Pct MO],     ALL(DIM_Date), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE])     ) VAR _cfy_sa = CALCULATE([Weighted BrandFunnel Pct MO]) RETURN _cfy_sa - _pfy_sa |
| PFY_Weighted_BF_Pct_PA_Diff | CT_Measures | IF(ISBLANK([PFY_Weighted_BF_Pct_PA_Diff_c]), "--", CONVERT(ROUND([PFY_Weighted_BF_Pct_PA_Diff_c] * 100, 0), STRING) & "PP") |
| PFY_Weighted_BF_Pct_PA_Diff_c | CT_Measures | VAR _pfy_sa =      CALCULATE([Weighted BrandFunnel Pct PA],     ALL(DIM_Date), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE])     ) VAR _cfy_sa = CALCULATE([Weighted BrandFunnel Pct PA]) RETURN _cfy_sa - _pfy_sa |
| PFY_Weighted_BF_Pct_Re_Diff | CT_Measures | IF(ISBLANK([PFY_Weighted_BF_Pct_Re_Diff_c]), "--", CONVERT(ROUND([PFY_Weighted_BF_Pct_Re_Diff_c]  * 100,0), STRING) & "pp") |
| PFY_Weighted_BF_Pct_Re_Diff_c | CT_Measures | VAR _pfy_sa =      CALCULATE([Weighted BrandFunnel Pct Re],     ALL(DIM_Date), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE])     ) VAR _cfy_sa = CALCULATE([Weighted BrandFunnel Pct Re]) RETURN _cfy_sa - _pfy_sa |
| PFY_Weighted_BF_Pct_SA_Diff | CT_Measures | I F ( I S B L A N K ( [ P F Y _ W e i g h t e d _ B F _ P c t _ S A _ D i f f _ c ] ) , " - - " , C O N V E R T ( R O U N D ( [ P F Y _ W e i g h t e d _ B F _ P c t _ S A _ D i f f _ c ] * 1 0 0 , 0 ) , S T R I N G ) & " p p " ) |
| PFY_Weighted_BF_Pct_SA_Diff_c | CT_Measures | VAR _pfy_sa =      CALCULATE([Weighted BrandFunnel Pct SA],     ALL(DIM_Date), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE])     ) VAR _cfy_sa = CALCULATE([Weighted BrandFunnel Pct SA]) RETURN _cfy_sa - _pfy_sa |
| PFY_Weighted_BF_Pct_Tried_Diff | CT_Measures | IF(ISBLANK([PFY_Weighted_BF_Pct_Tried_Diff_c]),"--", CONVERT(ROUND([PFY_Weighted_BF_Pct_Tried_Diff_c] * 100 ,0), STRING)  & "pp") |
| PFY_Weighted_BF_Pct_Tried_Diff_c | CT_Measures | VAR _pfy_sa =      CALCULATE([Weighted BrandFunnel Pct Tried],     ALL(DIM_Date), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE])     ) VAR _cfy_sa = CALCULATE([Weighted BrandFunnel Pct Tried]) RETURN _cfy_sa - _pfy_sa |
| RbyT | CT_Measures | D I V I D E ( [ W e i g h t e d B r a n d F u n n e l P c t R e ] , [ W e i g h t e d B r a n d F u n n e l P c t T r i e d ] ) |
| TbyC | CT_Measures | D I V I D E ( [ W e i g h t e d B r a n d F u n n e l P c t T r i e d ] , [ W e i g h t e d B r a n d F u n n e l P c t C o n s i ] ) |
| Total WBase BrandFunnel PA | CT_Measures | VAR FinalTotalWBaseBrandFunnel =     CALCULATE (         [WBase],CROSSFILTER(         CT_Base[ResponderID],         CT_FACT_All_KPI[ResponderID],         BOTH),         ALL(CT_DIM_Brand),         All(CT_FACT_ALL_KPI[KPI],CT_FACT_ALL_KPI[Order]), CT_FACT_ALL_KPI[KPI]="Prompted Awareness"     ) RETURN     FinalTotalWBaseBrandFunnel |
| Total WBase BrandFunnel | CT_Measures | VAR FinalTotalWBaseBrandFunnel =     CALCULATE (         [WBase],CROSSFILTER(         CT_Base[ResponderID],         CT_FACT_All_KPI[ResponderID],         BOTH),         ALL(CT_DIM_Brand)     ) RETURN     FinalTotalWBaseBrandFunnel |
| WBase BrandFunnel | CT_Measures | VAR FinalWBaseBrandFunnel =     CALCULATE (         [WBase],         CROSSFILTER(         CT_Base[ResponderID],         CT_FACT_All_KPI[ResponderID],         Both)     ) RETURN     FinalWBaseBrandFunnel |
| WBase | CT_Measures | VAR final =     CALCULATE(SUM(CT_Base[WEIGHT])) RETURN final |
| Weighted BrandFunnel Pct Consi | CT_Measures | VAR _wbbf =  CALCULATE([WBase BrandFunnel], FILTER(CT_FACT_All_KPI, CT_FACT_All_KPI[KPI] = "Consideration"))  VAR _twbf =  CALCULATE([Total WBase BrandFunnel], All(CT_FACT_ALL_KPI[KPI],CT_FACT_ALL_KPI[Order]), CT_FACT_ALL_KPI[KPI]="Prompted Awareness")  RETURN DIVIDE(_wbbf, _twbf) |
| Weighted BrandFunnel Pct MO | CT_Measures | VAR _wbbf =  CALCULATE([WBase BrandFunnel], FILTER(CT_FACT_All_KPI, CT_FACT_All_KPI[KPI] = "Most often used"))  VAR _twbf =  CALCULATE([Total WBase BrandFunnel], All(CT_FACT_ALL_KPI[KPI],CT_FACT_ALL_KPI[Order]), CT_FACT_ALL_KPI[KPI]="Prompted Awareness")   RETURN DIVIDE(_wbbf, _twbf) |
| Weighted BrandFunnel Pct PA | CT_Measures | VAR _wbbf =  CALCULATE([WBase BrandFunnel], FILTER(CT_FACT_All_KPI, CT_FACT_All_KPI[KPI] = "Prompted Awareness"))  VAR _twbf =  CALCULATE([Total WBase BrandFunnel], All(CT_FACT_ALL_KPI[KPI],CT_FACT_ALL_KPI[Order]), CT_FACT_ALL_KPI[KPI]="Prompted Awareness")  RETURN DIVIDE(_wbbf, _twbf) |
| Weighted BrandFunnel Pct Re | CT_Measures | VAR _wbbf =  CALCULATE([WBase BrandFunnel], FILTER(CT_FACT_All_KPI, CT_FACT_All_KPI[KPI] = "Repertoire"))  VAR _twbf =  CALCULATE([Total WBase BrandFunnel], All(CT_FACT_ALL_KPI[KPI],CT_FACT_ALL_KPI[Order]), CT_FACT_ALL_KPI[KPI]="Prompted Awareness")  RETURN DIVIDE(_wbbf, _twbf) |
| Weighted BrandFunnel Pct SA | CT_Measures | VAR _wbbf =  CALCULATE([WBase BrandFunnel], FILTER(CT_FACT_All_KPI, CT_FACT_All_KPI[KPI] = "Spontaneous Awareness"))  VAR _twbf =  CALCULATE([Total WBase BrandFunnel], All(CT_FACT_ALL_KPI[KPI],CT_FACT_ALL_KPI[Order]), CT_FACT_ALL_KPI[KPI]="Spontaneous Awareness")  RETURN DIVIDE(_wbbf, _twbf) |
| Weighted BrandFunnel Pct Tried | CT_Measures | VAR _wbbf =  CALCULATE([WBase BrandFunnel], FILTER(CT_FACT_All_KPI, CT_FACT_All_KPI[KPI] = "Tried"))  VAR _twbf =  CALCULATE([Total WBase BrandFunnel], All(CT_FACT_ALL_KPI[KPI],CT_FACT_ALL_KPI[Order]), CT_FACT_ALL_KPI[KPI]="Prompted Awareness")  RETURN DIVIDE(_wbbf, _twbf) |
| Weighted BrandFunnelMain(%) | CT_Measures | if(max(CT_FACT_All_KPI[KPI])="Spontaneous Awareness", DIVIDE([WBase BrandFunnel],[Total WBase BrandFunnel]), DIVIDE([WBase BrandFunnel],[Total WBase BrandFunnel PA] )) |
| IsBaseOrReference | D Datatype | E X T E R N A L M E A S U R E ( " I s B a s e O r R e f e r e n c e " , I N T E G E R , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Base Rank SSKU Top 80 Ind | D Product | E X T E R N A L M E A S U R E ( " B a s e R a n k S S K U T o p 8 0 I n d " , I N T E G E R , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Base Rank SSKU | D Product | E X T E R N A L M E A S U R E ( " B a s e R a n k S S K U " , I N T E G E R , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio 12MT | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o 1 2 M T " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Base Denominator | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o B a s e D e n o m i n a t o r " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Base Numerator | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o B a s e N u m e r a t o r " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Base vs Ratio Reference (%) | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o B a s e v s R a t i o R e f e r e n c e ( % ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Base vs Ratio Reference | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o B a s e v s R a t i o R e f e r e n c e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Base | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o B a s e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Denominator 12MT | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o D e n o m i n a t o r 1 2 M T " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Denominator | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o D e n o m i n a t o r " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Exit Rate | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o E x i t R a t e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Numerator 12MT | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o N u m e r a t o r 1 2 M T " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Numerator | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o N u m e r a t o r " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Reference Denominator | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o R e f e r e n c e D e n o m i n a t o r " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Reference Numerator | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o R e f e r e n c e N u m e r a t o r " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Reference | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o R e f e r e n c e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio Type Switch | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o T y p e S w i t c h " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Ratio | D Ratio | E X T E R N A L M E A S U R E ( " R a t i o " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| test measure | DIM_Date | MAX(DIM_Date[FIN_YEAR_TXT]) |
| % Engaged Sessions | Engagement_Measures | VAR _total_sessions = SUM(ga4_ua_engagement_rate[sessions]) VAR _engaged_sessions = SUM(ga4_ua_engagement_rate[engaged_sessions]) RETURN  DIVIDE(_engaged_sessions, _total_sessions) |
| Click Rate | Engagement_Measures | VAR _Unique_Clicks =  CALCULATE(DISTINCTCOUNT(act_campaignemails_new[id_index]),      NOT(ISBLANK(act_campaignemails_new[clickeddate])),     act_campaignemails_new[TemplateID] <> "-1",     DATEDIFF(act_campaignemails_new[senddate], act_campaignemails_new[clickeddate],DAY) <= 7) VAR _Delivered =  CALCULATE(DISTINCTCOUNT(act_campaignemails_new[id_index]),     NOT(ISBLANK(act_campaignemails_new[senddate]))     ,act_campaignemails_new[TemplateID] <> "-1"     ,ISBLANK(act_campaignemails_new[droppeddate])     ,ISBLANK(act_campaignemails_new[spamreportdate])     ,ISBLANK(act_campaignemails_new[bounceddate]) ) RETURN DIVIDE(_Unique_Clicks, _Delivered) |
| NPS | Engagement_Measures | VAR _promoters = CALCULATE(COUNTROWS(sm_pulzecare), Engagement_Master[KPIID]=1,sm_pulzecare[normalized_answer] in {9,10}) VAR _allresponses = CALCULATE(COUNTROWS(sm_pulzecare), Engagement_Master[KPIID] = 1, NOT(sm_pulzecare[normalized_answer]) in {(BLANK())}) VAR _pro_percent = DIVIDE(_promoters, _allresponses, 0) * 100 VAR _detractors = CALCULATE(COUNTROWS(sm_pulzecare), Engagement_Master[KPIID] = 1, sm_pulzecare[normalized_answer] in {0,1,2,3,4,5,6}) VAR _det_percent = DIVIDE(_detractors, _allresponses, 0) * 100 RETURN  _pro_percent - _det_percent |
| Shopping Transaction Sessions % Actual | Engagement_Measures | Var _A = CALCULATE(SUM('eCommerce'[SessionId]),eCommerce[eCommerceAction] in {"TRANSACTION"},eCommerce[Visits] > 0) Var _B = CALCULATE(DISTINCTCOUNT('ga4_sessions'[id_index]),GA4_sessions[event_name] in {"purchase"} && NOT(ISBLANK(ga4_sessions[events_params_transaction_id_index])))    VAR _Shopping_Transaction_Sessions_Actual = _A + _B var _C = CALCULATE(SUM('eCommerce'[SessionId]), eCommerce[Visits]>0 ,eCommerce[eCommerceAction]= "ALL_VISITS") var _D = CALCULATE(COUNTROWS(SUMMARIZE(GA4_sessions,ga4_sessions[id_index],GA4_sessions[event_name])),GA4_sessions[event_name] in {"purchase","begin_checkout","add_to_cart","view_item","view_item_list"})  VAR _Shopping_All_Visits_Sessions_Actual = _C + _D RETURN  DIVIDE(     _Shopping_Transaction_Sessions_Actual,     _Shopping_All_Visits_Sessions_Actual    ) |
| engagement_score | Engagement_Measures | var es_percent = CALCULATE(AVERAGE('new_engagement_score'[weight]),'new_engagement_score'[index] = "engaged_sessions") var nps_percent = CALCULATE(AVERAGE('new_engagement_score'[weight]),'new_engagement_score'[index] = "nps") var email_percent = CALCULATE(AVERAGE('new_engagement_score'[weight]),'new_engagement_score'[index] = "email_ctr") var repurc_percent = CALCULATE(AVERAGE('new_engagement_score'[weight]),'new_engagement_score'[index] = "repurchase") var cr_percent = CALCULATE(AVERAGE('new_engagement_score'[weight]),'new_engagement_score'[index] = "conversion_rate") RETURN  ([% engaged sessions]*es_percent) + (([NPS]/100)*nps_percent) + ([Click Rate]*email_percent) + ([Repurchase Rate act]*repurc_percent) + ([Shopping Transaction Sessions % Actual] * cr_percent) |
| Amount (xOper - Corrected) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t ( x O p e r - C o r r e c t e d ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount (xOper) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t ( x O p e r ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount 12MT | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t 1 2 M T " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base (% of NR) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e ( % o f N R ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base (Future Act as Blank) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e ( F u t u r e A c t a s B l a n k ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base (Map) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e ( M a p ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base (Per Pack) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e ( P e r P a c k ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base FormatValue | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e F o r m a t V a l u e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base vs Amount Reference (%) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e v s A m o u n t R e f e r e n c e ( % ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base vs Amount Reference (Future Act as Blank) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e v s A m o u n t R e f e r e n c e ( F u t u r e A c t a s B l a n k ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base vs Amount Reference (Per Pack) (%) Actual Variance (%) Per Pack | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e v s A m o u n t R e f e r e n c e ( P e r P a c k ) ( % ) A c t u a l V a r i a n c e ( % ) P e r P a c k " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base vs Amount Reference (Per Pack) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e v s A m o u n t R e f e r e n c e ( P e r P a c k ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base vs Amount Reference Switch | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e v s A m o u n t R e f e r e n c e S w i t c h " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base vs Amount Reference | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e v s A m o u n t R e f e r e n c e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Base | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t B a s e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Exit Rate | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t E x i t R a t e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Reference (% of NR) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t R e f e r e n c e ( % o f N R ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Reference (Future Act as Blank) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t R e f e r e n c e ( F u t u r e A c t a s B l a n k ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Reference (Map) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t R e f e r e n c e ( M a p ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Reference (Per Pack) | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t R e f e r e n c e ( P e r P a c k ) " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Reference FormatValue | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t R e f e r e n c e F o r m a t V a l u e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Reference | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t R e f e r e n c e " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Type Switch | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t T y p e S w i t c h " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount Variance FormatValue | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t V a r i a n c e F o r m a t V a l u e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Amount | F Sales | E X T E R N A L M E A S U R E ( " A m o u n t " , D O U B L E , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Finance Label_Page_Title | F Sales | E X T E R N A L M E A S U R E ( " F i n a n c e L a b e l _ P a g e _ T i t l e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Label Forecast Analysis | F Sales | E X T E R N A L M E A S U R E ( " L a b e l F o r e c a s t A n a l y s i s " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Label_Page_Title | F Sales | E X T E R N A L M E A S U R E ( " L a b e l _ P a g e _ T i t l e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Product BF Visual Title | F Sales | E X T E R N A L M E A S U R E ( " P r o d u c t B F V i s u a l T i t l e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Product BSF Visual Title | F Sales | E X T E R N A L M E A S U R E ( " P r o d u c t B S F V i s u a l T i t l e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Product NT Visual Title | F Sales | E X T E R N A L M E A S U R E ( " P r o d u c t N T V i s u a l T i t l e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Trend&Channel Visual Title | F Sales | E X T E R N A L M E A S U R E ( " T r e n d & C h a n n e l V i s u a l T i t l e " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| BP_Gross_Margin_Pct | Finance | VAR _NR =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")	      ) VAR _GM =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Gross Margin"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _pfy_gmp = _GM/_NR VAR _cfy_gmp = [CFY_Gross_Margin_Pct] RETURN  IF(_pfy_gmp = BLANK(), "--", ROUND((_cfy_gmp - _pfy_gmp) * 100,0) & "pp") |
| BP_Gross_Margin_Pct_Color | Finance | VAR _NR =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _GM =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Gross Margin"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _pfy_gmp = _GM/_NR VAR _cfy_gmp = [CFY_Gross_Margin_Pct] RETURN  ROUND((_cfy_gmp - _pfy_gmp ) * 100,0) |
| BP_Investments_Diff | Finance | VAR _pfy_in =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] IN {"Discount & Rebates", "IFRS 15 Discounts", "Advertising & Promotions"}),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _cfy_in = [CFY_Investments] RETURN DIVIDE((_cfy_in - _pfy_in ), ABS(_pfy_in)) |
| BP_Net_Brand_Contribution_Diff | Finance | VAR _pfy_nbc =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Brand Contribution"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _cfy_nbc = [CFY_Net_Brand_Contribution] RETURN DIVIDE((_cfy_nbc - _pfy_nbc), ABS(_pfy_nbc)) |
| BP_Net_Revenue_Diff | Finance | VAR _pfy_nr =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")	     ) VAR _cfy_nr = [CFY_Net_Revenue] RETURN DIVIDE(( _cfy_nr - _pfy_nr) , ABS(_pfy_nr)) |
| BP_Operating_Profit_Diff | Finance | VAR _pfy_op =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Operating Profit"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _cfy_op = [CFY_Operating_Profit] RETURN DIVIDE((_cfy_op - _pfy_op) , ABS(_pfy_op)) |
| BP_S2D_Diff | Finance | VAR _sticks =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA"}),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _devices =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] = "Pulze"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _BPstd = DIVIDE(_sticks, _devices) VAR _cfy = [CFY_S2D] RETURN  DIVIDE((_cfy - _BPstd), ABS(_BPstd) ) |
| BP_Total_Devices_Diff | Finance | VAR _pfy_nr =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] = "Pulze"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _cfy_nr = [CFY_Total_Devices] RETURN DIVIDE((_cfy_nr - _pfy_nr) , ABS(_pfy_nr)) |
| BP_Total_Sticks_Diff | Finance | VAR _pfy_nr =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA"}),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "BP")     ) VAR _cfy_nr = [CFY_Total_Sticks] RETURN DIVIDE((_cfy_nr - _pfy_nr) , ABS(_pfy_nr)) |
| CFY_Gross_Margin_Pct | Finance | VAR _NR =  CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue")     ) VAR _GM =  CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Gross Margin")     ) RETURN  _GM/_NR /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD"), */ |
| CFY_Investments | Finance | CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] IN {"Discount & Rebates", "IFRS 15 Discounts", "Advertising & Promotions"})     ) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| CFY_Net_Brand_Contribution | Finance | CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Brand Contribution")     ) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| CFY_Net_Revenue | Finance | CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue")     ) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| CFY_Operating_Profit | Finance | CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Operating Profit")     ) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| CFY_S2D | Finance | D I V I D E ( [ C F Y _ T o t a l _ S t i c k s ] , [ C F Y _ T o t a l _ D e v i c e s ] ) |
| CFY_Total_Devices | Finance | CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] = "Pulze")     ) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| CFY_Total_Sticks | Finance | CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA"})     ) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| CLE_Gross_Margin_Pct | Finance | VAR _NR =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")	      ) VAR _GM =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Gross Margin"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _pfy_gmp = _GM/_NR VAR _cfy_gmp = [CFY_Gross_Margin_Pct] RETURN  IF(_pfy_gmp = BLANK(), "--", ROUND((_cfy_gmp - _pfy_gmp) * 100,0) & "pp") |
| CLE_Gross_Margin_Pct_Color | Finance | VAR _NR =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _GM =  CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Gross Margin"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _pfy_gmp = _GM/_NR VAR _cfy_gmp = [CFY_Gross_Margin_Pct] RETURN  ROUND((_cfy_gmp - _pfy_gmp) * 100,0) |
| CLE_Investments_Diff | Finance | VAR _pfy_in =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] IN {"Discount & Rebates", "IFRS 15 Discounts", "Advertising & Promotions"}),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _cfy_in = [CFY_Investments] RETURN DIVIDE((_cfy_in - _pfy_in), ABS(_pfy_in)) |
| CLE_Net_Brand_Contribution_Diff | Finance | VAR _pfy_nbc =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Brand Contribution"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _cfy_nbc = [CFY_Net_Brand_Contribution] RETURN DIVIDE((_cfy_nbc - _pfy_nbc), ABS(_pfy_nbc)) |
| CLE_Net_Revenue_Diff | Finance | VAR _pfy_nr =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _cfy_nr = [CFY_Net_Revenue] RETURN DIVIDE((_cfy_nr - _pfy_nr), ABS(_pfy_nr)) |
| CLE_Operating_Profit_Diff | Finance | VAR _pfy_op =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Operating Profit"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _cfy_op = [CFY_Operating_Profit] RETURN DIVIDE((_cfy_op - _pfy_op), ABS(_pfy_op)) |
| CLE_S2D_Diff | Finance | VAR _sticks =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA"}),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _devices =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] = "Pulze"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _clestd = DIVIDE(_sticks, _devices) VAR _cfy = [CFY_S2D] RETURN  DIVIDE((_cfy - _clestd ), ABS(_clestd)) |
| CLE_Total_Devices_Diff | Finance | VAR _pfy_nr =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] = "Pulze"),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _cfy_nr = [CFY_Total_Devices] RETURN DIVIDE((_cfy_nr - _pfy_nr ),ABS( _pfy_nr)) |
| CLE_Total_Sticks_Diff | Finance | VAR _pfy_nr =      CALCULATE([Amount],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Sales Volume"),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA"}),     FILTER('D Datatype','D Datatype'[DataType_FAD_Flag] = "Current LE")     ) VAR _cfy_nr = [CFY_Total_Sticks] RETURN DIVIDE((_cfy_nr - _pfy_nr), ABS(_pfy_nr)) |
| Coming Soon | Finance | " C o m i n g S o o n . . . " |
| Info_icon | Finance | U N I C H A R ( 1 2 8 7 1 2 ) |
| Max_Date_Finance | Finance | L E F T ( R I G H T ( M A X ( ' M R e f r e s h ' [ L a s t _ R e f r e s h ] ) , 2 6 ) , 1 2 ) |
| PFY_Gross_Margin_Pct | Finance | VAR _NR =  CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _GM =  CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Gross Margin"),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _pfy_gmp = DIVIDE(_GM, _NR) VAR _cfy_gmp = CALCULATE([CFY_Gross_Margin_Pct], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN  ROUND((_cfy_gmp - _pfy_gmp) * 100,0) & "pp" /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD"), */ |
| PFY_Gross_Margin_Pct_Color | Finance | VAR _NR =  CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _GM =  CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Gross Margin"),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _pfy_gmp = DIVIDE(_GM,_NR) VAR _cfy_gmp = CALCULATE([CFY_Gross_Margin_Pct], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN  ROUND((_cfy_gmp - _pfy_gmp) * 100,0) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD"), */ |
| PFY_Investments_Diff | Finance | VAR _pfy_in =      CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] IN {"Discount & Rebates", "IFRS 15 Discounts", "Advertising & Promotions"}),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _cfy_in = CALCULATE([CFY_Investments], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN DIVIDE((_cfy_in - _pfy_in), ABS(_pfy_in)) |
| PFY_Net_Brand_Contribution_Diff | Finance | VAR _pfy_nbc =      CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Brand Contribution"),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _cfy_nbc = CALCULATE([CFY_Net_Brand_Contribution], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN DIVIDE((_cfy_nbc - _pfy_nbc), ABS(_pfy_nbc)) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| PFY_Net_Revenue_Diff | Finance | VAR _pfy_nr =      CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Net Revenue"),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _cfy_nr = CALCULATE([CFY_Net_Revenue], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN DIVIDE((_cfy_nr - _pfy_nr), ABS(_pfy_nr)) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| PFY_Operating_Profit_Diff | Finance | VAR _pfy_op =      CALCULATE([Amount Base],      FILTER('D Account Category', 'D Account Category'[Account_Category_Name] = "Operating Profit"),     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _cfy_op = CALCULATE([CFY_Operating_Profit], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN DIVIDE((_cfy_op - _pfy_op), ABS(_pfy_op)) /* Please use the following Filters in the Filter Pane Due to Memory Constrain     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Date', 'D Date'[Dim_Date_Level] = "Month"),     FILTER('D Market', 'D Market'[Ngp_Channel_Name] IN {"Online", "Retail"}),     FILTER('D Product', 'D Product'[Brand_House_Name] IN {"iD", "iSENZIA", "Pulze"}),     FILTER('D Timeserie', 'D Timeserie'[Calculate_Ind] = 1),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual"),     FILTER('D Pod System Only', 'D Pod System Only'[Pod System Only] = "Off"),     FILTER('D Timeserie', 'D Timeserie'[Timeserie_Type] = "FPD") */ |
| PFY_S2D | Finance | VAR _pfy_in =      CALCULATE([CFY_S2D],     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _cfy_in = CALCULATE([CFY_S2D], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN DIVIDE((_cfy_in - _pfy_in), ABS(_pfy_in)) |
| PFY_Total_Devices | Finance | VAR _pfy_in =      CALCULATE([CFY_Total_Devices],     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _cfy_in = CALCULATE([CFY_Total_Devices], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN DIVIDE((_cfy_in - _pfy_in), ABS(_pfy_in)) |
| PFY_Total_Sticks | Finance | VAR _pfy_in =      CALCULATE([CFY_Total_Sticks],     SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date),     FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual Restated")     ) VAR _cfy_in = CALCULATE([CFY_Total_Sticks], FILTER('D Datatype Base', 'D Datatype Base'[D_Datatype_Base_Name_Short] = "Actual")) RETURN DIVIDE((_cfy_in - _pfy_in), ABS(_pfy_in)) |
| Avg_Retail_Price_Title | HT_Clean_Source_Sell-Out | "AVERAGE RETAIL PRICE PER BOX (20 STICKS) IN " & SELECTEDVALUE(DIM_Country[Currency]) |
| IMB_Market_Share_Sticks | HT_Clean_Source_Sell-Out | VAR _total_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume])) var _imb_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]),         'HT_Clean_Source_Sell-Out'[Holding.Company] = "IMB") RETURN DIVIDE(_imb_sticks, _total_sticks) |
| IMB_Market_Share_Sticks_Chart | HT_Clean_Source_Sell-Out | VAR _total_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume])) var _imb_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]),         'HT_Clean_Source_Sell-Out'[Holding.Company] = "IMB") RETURN IF(      ISFILTERED('HT_Clean_Source_Sell-Out'[Date_Quarter_Text]),      _imb_sticks / _total_sticks,      BLANK() ) |
| Market_Size_Volume | HT_Clean_Source_Sell-Out | S U M ( ' H T _ C l e a n _ S o u r c e _ S e l l - O u t ' [ V o l u m e ] ) |
| Market_Size_Volume_Text | HT_Clean_Source_Sell-Out | R O U N D ( [ M a r k e t _ S i z e _ V o l u m e ] , 0 ) & " M " |
| Max_Date_HT_Sell_Out | HT_Clean_Source_Sell-Out | VAR _date = CALCULATE(MAX('HT_Clean_Source_Sell-Out'[Date.M.Y]), ALL()) RETURN FORMAT(_date, "mmm") & "-" & YEAR(_date) |
| PFY_IMB_Market_Share_Sticks | HT_Clean_Source_Sell-Out | VAR _total_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _imb_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]),         'HT_Clean_Source_Sell-Out'[Holding.Company] = "IMB",         SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _pfy_ms = DIVIDE(_imb_sticks, _total_sticks) VAR _cfy_ms = [IMB_Market_Share_Sticks] RETURN ROUND((_cfy_ms - _pfy_ms) * 100,1) & "pp" |
| PFY_IMB_Market_Share_Sticks_Color | HT_Clean_Source_Sell-Out | VAR _total_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _imb_sticks = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]),         'HT_Clean_Source_Sell-Out'[Holding.Company] = "IMB",         SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _pfy_ms = DIVIDE(_imb_sticks, _total_sticks) VAR _cfy_ms = [IMB_Market_Share_Sticks] RETURN _cfy_ms - _pfy_ms |
| PY_Stick_Volume | HT_Clean_Source_Sell-Out | VAR _pfy_sv = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]),'HT_Clean_Source_Sell-Out'[Holding.Company] = "IMB", SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cfy_sv = [Stick_Volume] RETURN (_cfy_sv - _pfy_sv) / _pfy_sv |
| PY_TMS | HT_Clean_Source_Sell-Out | VAR _pfy_tms = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume]), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cfy_tms = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume])) RETURN (_cfy_tms - _pfy_tms) / _pfy_tms |
| Stick_Volume | HT_Clean_Source_Sell-Out | C A L C U L A T E ( S U M ( ' H T _ C l e a n _ S o u r c e _ S e l l - O u t ' [ V o l u m e ] ) , ' H T _ C l e a n _ S o u r c e _ S e l l - O u t ' [ H o l d i n g . C o m p a n y ] = " I M B " ) |
| Stick_Volume_Text | HT_Clean_Source_Sell-Out | R O U N D ( [ S t i c k _ V o l u m e ] , 1 ) & " M " |
| Value_per_Stick | HT_Clean_Source_Sell-Out | DIVIDE( SUM('HT_Clean_Source_Sell-Out'[Value]) , SUM('HT_Clean_Source_Sell-Out'[Volume])) * 20 |
| URL Reconciliation Report | M Report URL | E X T E R N A L M E A S U R E ( " U R L R e c o n c i l i a t i o n R e p o r t " , S T R I N G , " D i r e c t Q u e r y t o A S - N G P F i n a n c i a l I n s i g h t s D a t a s e t R L S " ) |
| Brand Market Share | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d M a r k e t S h a r e " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (All Time) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( A l l T i m e ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (FYTD - 1) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( F Y T D - 1 ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (FYTD vs. FYTD - 1 ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( F Y T D v s . F Y T D - 1 A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (FYTD) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( F Y T D ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (M6M - SPLY) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M 6 M - S P L Y ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (M6M vs. SPLY ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M 6 M v s . S P L Y A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (M6M) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M 6 M ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (MAT - 1) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M A T - 1 ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (MAT vs. MAT - 1 ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M A T v s . M A T - 1 A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (MAT) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M A T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (MQT - SPLY) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M Q T - S P L Y ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (MQT vs. SQLY ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M Q T v s . S Q L Y A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Share (MQT) | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d S h a r e ( M Q T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Brand Volume | MSS Values | E X T E R N A L M E A S U R E ( " B r a n d V o l u m e " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Local Currency Value | MSS Values | E X T E R N A L M E A S U R E ( " L o c a l C u r r e n c y V a l u e " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (All Time) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( A l l T i m e ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (FYTD - 1) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( F Y T D - 1 ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (FYTD vs. FYTD - 1 ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( F Y T D v s . F Y T D - 1 A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (FYTD) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( F Y T D ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (M6M - SPLY) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M 6 M - S P L Y ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (M6M vs. SPLY ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M 6 M v s . S P L Y A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (M6M) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M 6 M ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (MAT - 1) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M A T - 1 ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (MAT vs. MAT - 1 ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M A T v s . M A T - 1 A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (MAT) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M A T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (MQT - SPLY) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M Q T - S P L Y ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (MQT vs. SQLY ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M Q T v s . S Q L Y A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share (MQT) | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e ( M Q T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share Base Stage 1 | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e B a s e S t a g e 1 " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share Base | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e B a s e " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Market Share | MSS Values | E X T E R N A L M E A S U R E ( " M a r k e t S h a r e " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Actual | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - A c t u a l " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (FYTD - 1) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( F Y T D - 1 ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (FYTD) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( F Y T D ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (MAT - 1) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( M A T - 1 ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (MAT vs. MAT - 1 % MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( M A T v s . M A T - 1 % M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (MAT vs. MAT - 1 ACTUAL MOVEMENT) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( M A T v s . M A T - 1 A C T U A L M O V E M E N T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (MAT) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( M A T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (MQT - 1) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( M Q T - 1 ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined (MQT) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d ( M Q T ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Combined | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - C o m b i n e d " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values - Forecast | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s - F o r e c a s t " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| Values Combined (All Time) | MSS Values | E X T E R N A L M E A S U R E ( " V a l u e s C o m b i n e d ( A l l T i m e ) " , D O U B L E , " D i r e c t Q u e r y t o A S - M S S R e p o r t " ) |
| BP_IMBSticks | MSS_Measures | VAR _bp = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "BP",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "DERIVED (IMB)",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) * 1000 VAR _cfy_tms = [Stick_Volume] RETURN DIVIDE((_cfy_tms - _bp), _bp) |
| BP_MarketShare | MSS_Measures | VAR _bp_market = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "BP",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "MARKET",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_imb = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "BP",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "DERIVED (IMB)",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_ms = DIVIDE(_bp_imb, _bp_market) VAR _cfy_ms = [IMB_Market_Share_Sticks] VAR _ms_calc = _cfy_ms - _bp_ms RETURN IF(_bp_market = BLANK(), "--", ROUND((_ms_calc) * 100,1) & "pp") |
| BP_MarketShare_Color | MSS_Measures | VAR _bp_market = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "BP",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "MARKET",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_imb = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "BP",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "DERIVED (IMB)",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_ms = DIVIDE(_bp_imb, _bp_market) VAR _cfy_ms = [IMB_Market_Share_Sticks] RETURN _cfy_ms - _bp_ms |
| BP_MarketSticks | MSS_Measures | VAR _bp = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "BP",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "MARKET",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) * 1000 VAR _cfy_tms = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume])) RETURN DIVIDE((_cfy_tms - _bp), _bp) |
| LE_IMBSticks | MSS_Measures | VAR _le = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "LE",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "DERIVED (IMB)",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) * 1000 VAR _cfy_tms = [Stick_Volume] RETURN DIVIDE((_cfy_tms - _le), _le) |
| LE_MarketShare | MSS_Measures | VAR _bp_market = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "LE",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "MARKET",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_imb = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "LE",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "DERIVED (IMB)",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_ms = DIVIDE(_bp_imb, _bp_market) VAR _cfy_ms = [IMB_Market_Share_Sticks] VAR _ms_calc = _cfy_ms - _bp_ms RETURN IF(_bp_market = BLANK(), "--",  ROUND((_ms_calc) * 100,1) & "pp") |
| LE_MarketShare_Color | MSS_Measures | VAR _bp_market = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "LE",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "MARKET",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_imb = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "LE",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "DERIVED (IMB)",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) VAR _bp_ms = DIVIDE(_bp_imb, _bp_market) VAR _cfy_ms = [IMB_Market_Share_Sticks] RETURN _cfy_ms - _bp_ms |
| LE_MarketSticks | MSS_Measures | VAR _le = CALCULATE([Values - Combined],     'MSS Select Forecast'[Versions] = "LE",     'MSS Select Forecast'[DateRank] = 0,     'MSS Select Market Region'[Market Region] = "NGP Total Retail",     'MSS Select Measure'[Measure] = "LEGAL DOMESTIC SALES (SA)",     'MSS Select Measure Type'[Measure Type] = "MARKET",     DIM_Date[Date_Fin_Year_Ago] = 0,     'MSS Select Product'[Product] = "HTC"     ) * 1000 VAR _cfy_tms = CALCULATE(SUM('HT_Clean_Source_Sell-Out'[Volume])) RETURN DIVIDE((_cfy_tms - _le), _le) |
| % Godsons vs. Ordered Users | Referral_Measures | D I V I D E ( [ T o t a l P u r c h a s e u s i n g R e f e r r a l C o d e ] , [ 1 s t O r d e r A c c o u n t s ] , 0 ) |
| Total Purchase using Referral Code | Referral_Measures | CALCULATE(DISTINCTCOUNT(act_orders[UUID]),      act_orders[order_status] in {"COMPLETED", "Complete","confirmed"}      && act_orders[Referral Discount Code]="Y",     NOT(act_orders[orderType]) in {"CASH_REFUND","REPLACEMENT_ORDER","RETURN","SUBSCRIPTION_ORDER"}     --,USERELATIONSHIP(act_account[sourceid], act_orders[sourceid])))     ) |
| % Offline_Voucher | Voucher_Measures | VAR _redeemed = Voucher_Measures[Voucher Redemption] VAR _offline = CALCULATE([Voucher Redemption], act_account_loyalty_voucher_redemption[RedemptionMode] = "OfflineOrder") RETURN DIVIDE(_offline, _redeemed, 0) |
| % Online_Voucher | Voucher_Measures | VAR _redeemed = Voucher_Measures[Voucher Redemption] VAR _Online = CALCULATE([Voucher Redemption], act_account_loyalty_voucher_redemption[RedemptionMode] = "OnlineOrder") RETURN DIVIDE(_Online, _redeemed, 0) |
| Voucher Redemption | Voucher_Measures | CALCULATE(DISTINCTCOUNT(act_account_loyalty_voucher_redemption[voucher_id_index]) //,act_account_loyalty_voucher_redemption[Status] = "Redeemed"      //Imported Datasource Contains Redeemed vouchers only ) |
| 1st Order Accounts | act_account | CALCULATE(DISTINCTCOUNT(act_orders[UUID]),     act_orders[Is First Purchase] = 1,     act_orders[order_status] in {"COMPLETED","confirmed","Complete"},     act_orders[orderType] in {"STANDARD_ORDER"}     ) |
| Active Accounts | act_account | C A L C U L A T E ( D I S T I N C T C O U N T ( a c t i v e _ d o i _ a c c o u n t s [ u u i d ] ) , U S E R E L A T I O N S H I P ( a c t i v e _ d o i _ a c c o u n t s [ D a t e _ f i e l d ] , D I M _ D a t e [ C A L E N D A R _ D A T E ] ) ) |
| CFY_ConvtoMar | act_account | D I V I D E ( [ C F Y _ N e w M a r U s e r s ] , [ C F Y _ N e w U s e r s ] ) |
| CFY_DeviceReg | act_account | S U M ( a c t _ r e g d e v i c e s [ T o t a l R e g D e v i c e s ] ) |
| CFY_DeviceRegRate | act_account | D I V I D E ( [ C F Y _ D e v i c e R e g ] , [ C F Y _ D e v i c e s S o l d _ A c t S e l l O u t ] ) |
| CFY_DevicesSold_ActSellOut | act_account | VAR _kasvd = CALCULATE(SUM(KASVD[WhiteStick]), KASVD[Manufacturer] = "ITG" , KASVD[ProductGroup] = "HT Device" /*, KASVD[Network] IN {"GECO", "GG TABAK", "Traficon", "Valmont"}*/ ) VAR _acc = CALCULATE(     COUNT(act_account[uuid]),     CONTAINSSTRING(act_account[purchasedproduct], "pulze") \|\|  CONTAINSSTRING ( act_account[purchasedproduct], "welcome"),     act_account[ecosystem_user_type] = "CommerceTools" ) RETURN _kasvd + _acc |
| CFY_MainBrandConv | act_account | D I V I D E ( [ C F Y _ R e p e a t _ U s e r s ] , [ C F Y _ N e w M a r U s e r s ] ) |
| CFY_NewMarUsers | act_account | VAR _leads_moi = CALCULATE(     DISTINCTCOUNT(act_account[uuid])     ,(CONTAINSSTRING(act_account[purchasedproduct], "pulze") \|\|  CONTAINSSTRING ( act_account[purchasedproduct], "welcome"))     ,not(ISBLANK(act_account[ecosystem_entry_date]))     ,(rep_newsletter_history[newsletter_opt_in]=TRUE() \|\| rep_newsletter_history[email_optin] =TRUE() \|\| rep_newsletter_history[sms_optin] =TRUE() \|\| rep_newsletter_history[phone_optin] =TRUE())     ,act_account[ecosystem_user_type]="lead"     ,act_account[is_converted] = "0" ) VAR _acc_moi = CALCULATE(     DISTINCTCOUNT(act_account[uuid])     ,(CONTAINSSTRING(act_account[purchasedproduct], "pulze") \|\|  CONTAINSSTRING ( act_account[purchasedproduct], "welcome"))     ,not(ISBLANK(act_account[ecosystem_entry_date]))     ,(rep_newsletter_history[newsletter_opt_in]=TRUE() \|\| rep_newsletter_history[email_optin] =TRUE() \|\| rep_newsletter_history[sms_optin] =TRUE() \|\| rep_newsletter_history[phone_optin] =TRUE())     ,act_account[registered_Status]=TRUE()) RETURN _leads_moi + _acc_moi |
| CFY_NewUsers | act_account | CALCULATE(     COUNT(act_account[uuid]),     CONTAINSSTRING(act_account[purchasedproduct], "pulze") \|\|  CONTAINSSTRING ( act_account[purchasedproduct], "welcome") ) |
| CFY_Repeat_Users | act_account | CALCULATE(     COUNTROWS( VALUES( act_orders[UUID] ) ),     FILTER(         VALUES( act_orders[UUID] ),         CALCULATE(             COUNTROWS( act_orders ),             act_orders[order_status] IN {"COMPLETED", "Complete", "Confirmed"}                 && NOT( act_orders[OrderType] IN {"REPLACEMENT_ORDER", "RETURN", "CASH_REFUND"} )         ) > 1     ) ) |
| CLE_ConvtoMar | act_account | var _bp= CALCULATE(AVERAGE(CEE_HTP_LE_Plans[Value]), CEE_HTP_LE_Plans[Metric] = "Conversion to marketable (%)") var _cy = [CFY_ConvtoMar] var _diff = _cy-_bp RETURN ROUND(_diff * 100,0) & "pp" |
| CLE_DeviceRegRate | act_account | var _le = CALCULATE(AVERAGE(CEE_HTP_LE_Plans[Value]), CEE_HTP_LE_Plans[Metric] = "Registration rate (%)") var _cy = [CFY_DeviceRegRate] var _diff = _cy-_le RETURN ROUND(_diff * 100,0) & "pp" |
| CLE_DevicesSold_SellOut | act_account | var _bp = CALCULATE(SUM(CEE_HTP_LE_Plans[Value]), CEE_HTP_LE_Plans[Metric] = "Device sell-out plan") var _cy = [CFY_DevicesSold_ActSellOut] RETURN DIVIDE((_cy - _bp), _bp) |
| CLE_MainBrandConv | act_account | var _bp= CALCULATE(AVERAGE(CEE_HTP_LE_Plans[Value]), CEE_HTP_LE_Plans[Metric] = "Conversion into main brand (%)") var _cy = [CFY_MainBrandConv] var _diff = _cy-_bp RETURN ROUND(_diff * 100,0) & "pp" |
| CLE_NewMarUsers | act_account | var _le = CALCULATE(SUM(CEE_HTP_LE_Plans[Value]), CEE_HTP_LE_Plans[Metric] = "Marketable users (users that opt-in to marketing)") var _cfy = [CFY_NewMarUsers] RETURN DIVIDE((_cfy - _le), _le) |
| CLE_NewUsers | act_account | var _le = CALCULATE(SUM(CEE_HTP_LE_Plans[Value]), CEE_HTP_LE_Plans[Metric] = "Trialists (new users that purchased a device)") var _cfy = [CFY_NewUsers] RETURN DIVIDE((_cfy - _le), _le) |
| CLE_Repeat | act_account | var _bp = CALCULATE(SUM(CEE_HTP_LE_Plans[Value]), CEE_HTP_LE_Plans[Metric] = "Pulze consumers (main brand users)") var _cy = [CFY_Repeat_Users] RETURN DIVIDE((_cy - _bp), _bp) |
| Churn Disclaimer | act_account | VAR _max = CALCULATE(MAX(DIM_Date[Month_Year]), DIM_Date[CALENDAR_DATE] = MAX(DIM_Date[CALENDAR_DATE])) //VAR _max_but1 = CALCULATE(MAX(DIM_Date[Month_Year]), DIM_Date[CALENDAR_DATE] = DATEADD(MAX(DIM_Date[CALENDAR_DATE]),-1,MONTH)) RETURN MAX(DATEADD(DIM_Date[CALENDAR_DATE],-1,MONTH)) |
| Churned Users | act_account | VAR _max = MAX(DIM_Date[Calendar_Date]) VAR _min = (_max - 90) VAR UsersActiveBeforeWindow =     CALCULATETABLE(         VALUES(act_orders[uuid]),         act_orders[order_status] in {"COMPLETED","confirmed","Complete"},         act_orders[orderType] in {"STANDARD_ORDER"},         REMOVEFILTERS(DIM_Date),         DIM_Date[CALENDAR_DATE] <= _max     ) VAR UsersActiveDuringWindow =     CALCULATETABLE(         VALUES(act_orders[uuid]),         act_orders[order_status] in {"COMPLETED","confirmed","Complete"},         act_orders[orderType] in {"STANDARD_ORDER"},         FILTER(             ALL(DIM_Date),             DIM_Date[Calendar_Date] >= _min &&             DIM_Date[Calendar_Date] <= _max         )     ) VAR ChurnedUsers = COUNTROWS(EXCEPT(UsersActiveBeforeWindow, UsersActiveDuringWindow)) RETURN DIVIDE(ChurnedUsers, COUNTROWS(UsersActiveBeforeWindow)) |
| Converted into Accounts | act_account | CALCULATE(     DISTINCTCOUNT(act_account[uuid]),         act_account[ecosystem_user_type]="lead",         act_account[is_converted] IN {"1", "SelfConverted"},         NOT(ISBLANK('act_account'[ecosystem_entry_date]) )) |
| Leads DOI | act_account | CALCULATE(     DISTINCTCOUNT(act_account[uuid]),         act_account[ecosystem_user_type]="lead",         act_account[is_converted] = "0",         NOT(ISBLANK('act_account'[ecosystem_entry_date])),         act_account[double_opt_in] = TRUE()) |
| Leads MOI | act_account | CALCULATE(     DISTINCTCOUNT(act_account[uuid])     ,not(ISBLANK(act_account[ecosystem_entry_date]))     ,(rep_newsletter_history[newsletter_opt_in]=TRUE() \|\| rep_newsletter_history[email_optin] =TRUE() \|\| rep_newsletter_history[sms_optin] =TRUE() \|\| rep_newsletter_history[phone_optin] =TRUE())     ,act_account[ecosystem_user_type]="lead"     ,act_account[is_converted] = "0") |
| Leads | act_account | CALCULATE(     DISTINCTCOUNT(act_account[uuid]),         act_account[ecosystem_user_type]="lead",         act_account[is_converted] = "0",         NOT(ISBLANK('act_account'[ecosystem_entry_date]))) |
| Offline Reg Accounts | act_account | CALCULATE(DISTINCTCOUNT(act_account[uuid]),not(ISBLANK(act_account[ecosystem_entry_date])),act_account[registered_Status]=TRUE(), act_account[SourceOfRegistration] IN {"BA(Offline)", "SR(Offline)", "TradeApp(Offline)"}) |
| Online Reg Accounts | act_account | CALCULATE(DISTINCTCOUNT(act_account[uuid]),not(ISBLANK(act_account[ecosystem_entry_date])),act_account[registered_Status]=TRUE(), act_account[SourceOfRegistration] IN {"Online", BLANK()}) |
| Ordered Accounts | act_account | CALCULATE(DISTINCTCOUNT(act_account[uuid]),      FILTER(act_orders,act_orders[order_status] in {"COMPLETED", "Complete","Confirmed"}      && NOT(act_orders[orderType] in {"REPLACEMENT_ORDER","CASH_REFUND","RETURN"})),     USERELATIONSHIP(act_account[sourceid], act_orders[sourceid])     --,CROSSFILTER(DIM_Date[CALENDAR_DATE], act_orders[order_date],None) ) |
| Overall Records | act_account | [ L e a d s ] + [ R e g i s t e r e d A c c o u n t s ] |
| PFY_1st_Order_Acc | act_account | VAR _py = CALCULATE([1st Order Accounts], SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cy = [1st Order Accounts] RETURN  (_cy - _py) / _py |
| PFY_Active Accounts | act_account | VAR _py = CALCULATE([Active Accounts], SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cy = [Active Accounts] RETURN  (_cy - _py) / _py |
| PFY_Churn Rate Color | act_account | VAR _max = CALCULATE(MAX(DIM_Date[Calendar_Date]), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _min = (_max - 90) VAR UsersActiveBeforeWindow =     CALCULATETABLE(         VALUES(act_orders[uuid]),         act_orders[order_status] in {"COMPLETED","confirmed","Complete"},         act_orders[orderType] in {"STANDARD_ORDER"},         REMOVEFILTERS(DIM_Date),         DIM_Date[CALENDAR_DATE] <= _max     ) VAR UsersActiveDuringWindow =     CALCULATETABLE(         VALUES(act_orders[uuid]),         act_orders[order_status] in {"COMPLETED","confirmed","Complete"},         act_orders[orderType] in {"STANDARD_ORDER"},         FILTER(             ALL(DIM_Date),             DIM_Date[Calendar_Date] >= _min &&             DIM_Date[Calendar_Date] <= _max         )     ) VAR ChurnedUsers = COUNTROWS(EXCEPT(UsersActiveBeforeWindow, UsersActiveDuringWindow)) VAR _pfy_cr = DIVIDE(ChurnedUsers, COUNTROWS(UsersActiveBeforeWindow)) VAR _cy_cr = [Churned Users] RETURN (_cy_cr - _pfy_cr) |
| PFY_Converted into Acc | act_account | VAR _pfy_cia =      CALCULATE([Converted into Accounts],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_cia = [Converted into Accounts] RETURN (_cfy_cia - _pfy_cia) / _pfy_cia |
| PFY_Device_Reg | act_account | VAR _py = CALCULATE(SUM(act_regdevices[TotalRegDevices]), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cy = CALCULATE(SUM(act_regdevices[TotalRegDevices])) RETURN  (_cy - _py) / _py |
| PFY_Leads DOI | act_account | VAR _pfy =      CALCULATE([Leads DOI],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy = [Leads DOI] RETURN DIVIDE((_cfy - _pfy) , _pfy) |
| PFY_Leads | act_account | VAR _pfy_Leads =      CALCULATE([Leads],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_Leads = [Leads] RETURN (_cfy_Leads - _pfy_Leads) / _pfy_Leads |
| PFY_MOI_Leads | act_account | VAR _pfy_MOI_Leads =      CALCULATE([Leads MOI],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_MOI_Leads = [Leads MOI] RETURN (_cfy_MOI_Leads - _pfy_MOI_Leads) / _pfy_MOI_Leads |
| PFY_Offline_Reg_Acc_Diff | act_account | VAR _pfy_off =      CALCULATE([Offline Reg Accounts],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_off = [Offline Reg Accounts] RETURN (_cfy_off - _pfy_off) / _pfy_off |
| PFY_Online_Reg_Acc_Diff | act_account | VAR _pfy_on =      CALCULATE([Online Reg Accounts],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_on = [Online Reg Accounts] RETURN (_cfy_on - _pfy_on) / _pfy_on |
| PFY_Overall Records | act_account | VAR _pfy_or =      CALCULATE([Overall Records],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_or = [Overall Records] RETURN (_cfy_or - _pfy_or) / _pfy_or |
| PFY_Purchase Accounts | act_account | VAR _py = CALCULATE([Purchase Accounts], SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cy = [Purchase Accounts] RETURN  (_cy - _py) / _py |
| PFY_ReactivatedAccounts | act_account | VAR _py = CALCULATE([ReactivatedAccounts], SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cy = [ReactivatedAccounts] RETURN  (_cy - _py) / _py |
| PFY_Registered Accs | act_account | VAR _pfy_ra =      CALCULATE([Registered Accounts],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_ra = [Registered Accounts] RETURN (_cfy_ra - _pfy_ra) / _pfy_ra |
| PFY_Repurchase_Rate | act_account | ROUND([PFY_Repurchase_Rate_color] * 100,0) & "pp" |
| PFY_Repurchase_Rate_color | act_account | VAR _py = CALCULATE([Repurchase Rate act], SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cy = [Repurchase Rate act] RETURN  (_cy - _py) |
| PFY_Total Acc DOI | act_account | VAR _pfy_tad =      CALCULATE([Total Accounts DOI],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_tad = [Total Accounts DOI] RETURN (_cfy_tad - _pfy_tad) / _pfy_tad |
| PFY_Total Acc MOI DOI | act_account | VAR _pfy_tamd =      CALCULATE([Total Accounts MOI + DOI],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_tamd = [Total Accounts MOI + DOI] RETURN (_cfy_tamd - _pfy_tamd) / _pfy_tamd |
| PFY_Total Acc MOI | act_account | VAR _pfy_tam =      CALCULATE([Total Accounts MOI],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_tam = [Total Accounts MOI] RETURN (_cfy_tam - _pfy_tam) / _pfy_tam |
| PFY_Total Acc | act_account | VAR _pfy_ta =      CALCULATE([Total Accounts],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_ta = [Total Accounts] RETURN (_cfy_ta - _pfy_ta) / _pfy_ta |
| PFY_Total Marketable Database | act_account | VAR _pfy_or =      CALCULATE([Total Markatable Database],      SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)     ) VAR _cfy_or = [Total Markatable Database] RETURN (_cfy_or - _pfy_or) / _pfy_or |
| PFY_churn_rate | act_account | ROUND([PFY_Churn Rate Color] * 100,0) & "pp" |
| Purchase Accounts | act_account | CALCULATE(DISTINCTCOUNT(act_orders[UUID]),     --act_orders[Is First Purchase] = 1,     act_orders[order_status] in {"COMPLETED","confirmed","Complete"},     act_orders[orderType] in {"STANDARD_ORDER"}     ) |
| ReactivatedAccounts | act_account | VAR PeriodStart = MIN(DIM_Date[CALENDAR_DATE]) VAR PeriodEnd   = MAX(DIM_Date[CALENDAR_DATE]) RETURN SUMX(     VALUES(active_doi_accounts[uuid]),     VAR FirstActivityInPeriod =         CALCULATE(             MIN(active_doi_accounts[Date_field]),             active_doi_accounts[Date_field] >= PeriodStart,             active_doi_accounts[Date_field] <= PeriodEnd         )     VAR LastActivityBefore =         CALCULATE(             MAX(active_doi_accounts[Date_field]),             active_doi_accounts[Date_field] < PeriodStart         )     RETURN         IF(             NOT(ISBLANK(FirstActivityInPeriod)) &&             NOT(ISBLANK(LastActivityBefore)) &&             DATEDIFF(LastActivityBefore, FirstActivityInPeriod, DAY) >= 30,             1,             0         ) ) /* KPI Definition: Reactivated Accounts counts users whose earliest activity during the selected period occurs at least 30 days after their latest activity before the period, indicating they took a long break and then returned. EXAMPLE: If the selected period is Jan-2025 to Mar-2025. If the users first activity in this period was on 15 Jan AND the users last activity before 1st Jan (before the start of the selected period) was on 1st Dec 2025. PeriodStart & PeriodEnd: These variables fix the analysis period from Min and max of selected period Iterate Over Each User: We use VALUES(active_doi_accounts[uuid]) to iterate over each unique user. Determine the First Activity in the Period: For each user, we use CALCULATE(MIN(active_doi_accounts[date])) to get the earliest activity date between the selected period. Determine the Last Activity Before the Period: Similarly, we get the latest activity date before Min of selected period. Apply the Reactivation Logic: If both dates exist and the gap (using DATEDIFF) is 30 days or more, we count that user as reactivated (returning 1); otherwise, we return 0. Final Count: The SUMX function sums up the 1’s over all users, giving you the total number of reactivated accounts. */ /* This is the SQl query to QC the data for period of Jan 2025 and filtered for CZ market WITH PeriodActivity AS (     -- Get each user's first activity date in January 2025 for market 'PULZE_CZ'     SELECT          uuid,          MIN(Date_field) AS FirstActivityInPeriod     FROM glbl_rep.active_doi_accounts     WHERE Date_field BETWEEN '2025-01-01' AND '2025-01-31'       AND market_id = 'PULZE_CZ'     GROUP BY uuid ), PriorActivity AS (     -- Get each user's most recent activity date before January 2025 for market 'PULZE_CZ'     SELECT          uuid,          MAX(Date_field) AS LastActivityBefore     FROM glbl_rep.active_doi_accounts     WHERE Date_field < '2025-01-01'       AND market_id = 'PULZE_CZ'     GROUP BY uuid ), UserStatusCTE AS (     -- Combine the period and prior activity and compute GapInDays and AccountStatus     SELECT          p.uuid,          p.FirstActivityInPeriod,          pr.LastActivityBefore,          CASE              WHEN pr.LastActivityBefore IS NULL THEN 'New'              WHEN DATEDIFF(day, pr.LastActivityBefore, p.FirstActivityInPeriod) >= 30 THEN 'Reactivated'              ELSE 'Active'          END AS AccountStatus,          CASE              WHEN pr.LastActivityBefore IS NULL THEN NULL              ELSE DATEDIFF(day, pr.LastActivityBefore, p.FirstActivityInPeriod)          END AS GapInDays     FROM PeriodActivity p     LEFT JOIN PriorActivity pr ON p.uuid = pr.uuid ) -- Retrieve all activity records for reactivated users (for market 'PULZE_CZ') along with the computed fields SELECT     a.uuid,     a.Date_field,     a.activity,     a.market_id,     u.FirstActivityInPeriod,     u.LastActivityBefore,     u.GapInDays,     u.AccountStatus FROM glbl_rep.active_doi_accounts a INNER JOIN UserStatusCTE u ON a.uuid = u.uuid WHERE a.market_id = 'PULZE_CZ' ORDER BY a.uuid, a.Date_field; */ |
| Registered Accounts | act_account | CALCULATE(DISTINCTCOUNT(act_account[uuid]),not(ISBLANK(act_account[ecosystem_entry_date])),act_account[registered_Status]=TRUE()) |
| Repurchase Rate act | act_account | VAR _second = CALCULATE(     COUNTROWS( VALUES( act_orders[UUID] ) ),     FILTER(         VALUES( act_orders[UUID] ),         CALCULATE(             COUNTROWS( act_orders ),             act_orders[order_status] IN {"COMPLETED", "Complete", "Confirmed"}                 && NOT( act_orders[OrderType] IN {"REPLACEMENT_ORDER", "RETURN", "CASH_REFUND"} )         ) > 1     ) ) VAR _total = CALCULATE(     DISTINCTCOUNT(act_orders[UUID])     ,act_orders[order_Status] in {"COMPLETED","Confirmed","Complete"}      && NOT('act_orders'[OrderType] in {"REPLACEMENT_ORDER","RETURN","CASH_REFUND"})     ) RETURN  DIVIDE(_second, _total) |
| Title_Conversion_FullYear | act_account | " F u l l " & S E L E C T E D V A L U E ( D I M _ D a t e [ F I N _ Y E A R _ T X T ] ) & " ^ ^ " |
| Title_Conversion_Month | act_account | CALCULATE(MAX(DIM_Date[Month_Year]), DIM_Date[FIN_MONTH] = MAX(DIM_Date[FIN_MONTH])) |
| Title_Conversion_YTD | act_account | VAR _min_fin_month = CALCULATE(MIN(DIM_Date[FIN_MONTH]), ALL(DIM_Date), VALUES(DIM_Date[FIN_YEAR_TXT])) VAR _min_month_year = CALCULATE(MIN(DIM_Date[Month_Year]), ALL(DIM_Date), DIM_Date[FIN_MONTH] = _min_fin_month) VAR _max_mont_year = [Title_Conversion_Month] RETURN "YTD (" & _min_month_year & " to " & _max_mont_year & ")" |
| Total Accounts DOI | act_account | CALCULATE(     DISTINCTCOUNT(act_account[uuid])     ,not(ISBLANK(act_account[ecosystem_entry_date]))     ,act_account[double_opt_in] = TRUE()     ,act_account[registered_Status]=TRUE() ) |
| Total Accounts MOI + DOI | act_account | CALCULATE(     DISTINCTCOUNT(act_account[uuid])     ,not(ISBLANK(act_account[ecosystem_entry_date]))     ,(rep_newsletter_history[newsletter_opt_in]=TRUE() \|\| rep_newsletter_history[email_optin] =TRUE() \|\| rep_newsletter_history[sms_optin] =TRUE() \|\| rep_newsletter_history[phone_optin] =TRUE())     ,act_account[registered_Status]=TRUE()     ,act_account[double_opt_in]=TRUE() ) |
| Total Accounts MOI | act_account | /* VAR _convert = CALCULATE(     DISTINCTCOUNT(act_account[uuid])         ,(rep_newsletter_history[newsletter_opt_in]=TRUE() \|\| rep_newsletter_history[email_optin] =TRUE() \|\| rep_newsletter_history[sms_optin] =TRUE() \|\| rep_newsletter_history[phone_optin] =TRUE())         ,act_account[ecosystem_user_type]="lead"         ,act_account[is_converted] = "1") */ VAR _reg = CALCULATE(     DISTINCTCOUNT(act_account[uuid]),         not(ISBLANK(act_account[ecosystem_entry_date]))         ,(rep_newsletter_history[newsletter_opt_in]=TRUE() \|\| rep_newsletter_history[email_optin] =TRUE() \|\| rep_newsletter_history[sms_optin] =TRUE() \|\| rep_newsletter_history[phone_optin] =TRUE()),         act_account[registered_Status]=TRUE() ) RETURN _reg |
| Total Accounts | act_account | [ R e g i s t e r e d A c c o u n t s ] |
| Total Markatable Database | act_account | [ L e a d s M O I ] + [ T o t a l A c c o u n t s M O I ] |
| PFY_TotalRegDevices | act_regdevices | VAR _py = CALCULATE(SUM(act_regdevices[TotalRegDevices]), SAMEPERIODLASTYEAR(DIM_Date[CALENDAR_DATE]), ALL(DIM_Date)) VAR _cy = CALCULATE(SUM(act_regdevices[TotalRegDevices])) RETURN  (_cy - _py) / _py |
