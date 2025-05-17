# HotelBookingCancellations

## Introduction
The business objective of this project was to decrease the amount of hotel booking cancellations. Currently the percentage of non cancelled to cancelled hotel bookings is 60%/40%. To address this business objective, we defined our analytics problem of identifying factors that affect hotel booking cancellations. Through feature engineering, model free plots and machine learning we found that the most important factors are non refundable deposit type, and the total amount of special requests. This results in our recommendation of the partial refundable deposit type.

## Data Cleaning and Wrangling

### Data Gathering

The dataset was queried from a database resulting in a csv with 10 features related to the hotel bookings from both the hotel side and customer side. Features were then engineered such as date times, country continent encoding, and indicator features for variables that were important in the domain. Then for categorical variables, many levels had few samples so they were binned into an "other" level.

#### Data Dictionary
The resulting data dictionary is as follows:

hotel:  Type of hotel. Resort or city hotel    
is_canceled: Cancellation (1) or non cancelled (0)    
meal: Type of meal associated with the booking  
market_segment: The market segment of the booking    
distribution_channel: The distribution channel the booking was made through    
is_repeated_guest: BOolean value. whether the guest is a repeated guest or not    
previous_cancellations: Number of previous cancellations    
previous_bookings_not_canceled: Number of previous bookings not cancelled  
reserved_room_type: Reservation room type    
assigned_room_type: Actual assigned room type   
booking_changes: number of booking changes    
deposit_type: Type of deposit    
days_in_waiting_list: Number of days in waiting list    
customer_type: Customer type    
adr: Average daily rate    
required_car_parking_spaces: Required number of parking spaces for the customer    
total_of_special_requests: Total amount of special requests    
arrival_season: Arrival season for the booking    
arrival_weekday: Arrival day of the week for the booking    
length_of_stay: Length of stay for the booking    
expected_departure_season: Expected departure season for the booking    
expected_departure_weekday: Expected departure day of departure for the booking    
same_season_stay: Boolean value on whether the stay was in a single season    
continent: The continent the customer is from    
total_amount_of_guests: The total amount of guests associated with the booking    
babies_prop: The proportion of babies out of the total amount of guests    
adults_prop: The proportion of adults out of the total amount of guests    
solo_travel: Boolean value. Single guest    
previous_cancellations_ind: Boolean value. Whether the guest cancelled a previous booking   
day_use_ind: Boolean value. Whether the room was only used for one day or not    
lead_time_quartiles: Quartiles of lead time from when the booking was made until the expected date of arrival       

## Machine Learning

A standard random forest, logistic regression, and xgboost was used to investigate the most important variables related to hotel bookiongs.

From the random forest the 5 most important variables found are the average daily rarte, the non refundable deposit type, the total amount of special requests, the length of stay, and the previous cancellations.

From the stepwise regression model, the most important variables related to hotel cancellations is the non refundable deposit type, high lead times (Q2 - Q4) and the total amount of special requests. Of these the non refundable deposit type, and high lead times positively affect the log odds of cancellation. The total amount of special requests negatively affect the log odds of cancellations.

The accuracy for each of the models are as follows: 85.5% ,81.4%, and 84.7% for each of the random forest, logistic regression and xgboost models. As a result, as the interprability of the logistic regression model is much simpler, we presented the findings of the model in more depth. 

## Logistic Regression In Depth

<img width="600" alt="image" src="https://github.com/user-attachments/assets/92250e7b-cabc-4f1c-9d4b-ea175866a705" />  

<img width="1000" alt="image" src="https://github.com/user-attachments/assets/aa0716c9-c772-4685-89f6-05a5ce60f7a5" />
The above is the top 10 largest coefficients with the largest magnitude in terms of the statistically significant features. It is observed that the non-refundable deposit type is the most important with a positive value indicating that it causes individiduals to cancel their bookings.  

## Data Visualization

To futher verify our findings we investigated the trends found from the machine learning models using model free plots to visualize the findings. 

![image](https://github.com/user-attachments/assets/3dae67a8-5bb3-4eb3-9462-7d0c91b7e7e4)

Looking at the commonly found important feature of non refundable deposit type from both the random forest and logistic regression it is observed that the non refundable deposit type has purely cancelled bookings with basically no bookings that are not cancelled for this deposit type.

![image](https://github.com/user-attachments/assets/5b643b3c-4fa5-41e6-a1cc-e8c183a7ab4c)

Looking at the lead time quartiles, it is observed that initially at low lead times (Q1) the amount of not cancelled to cancelled is high. As the amount of lead time increases, the amount of cancelled to not cancelled incraeses until in Q4 where the amount of not cancelled gets surpassed by those who do cancel.


## Conclusion

Our recommendation to address the problem of hotel bookings is to introduce a partial refundable deposit type as the deposit type of non refundable deposit type is not flexible and a large majority of customers who book through this deposit type cancels their booking. 
