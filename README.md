# FNCE-449 Final Project

**Saad Abdullah**  
**UCID: 30142511**  
**November 4, 2024**  
**Python Version 3.11.9 --> Note that using a newer Python version will show FutureWarnings in the terminal when executing due to Python methods that are being modified in the future update, but this doesn't affect the execution of the code**  
**FNCE 449 Final Project**  
**Description:** This project looks to address which strategy (between the 50-day moving average and simple daily price momentum) is best to use in the technology equities market. It creates long-short portfolios as well as works with Sharpe Ratios to try and answer this question.

**Files on GitHub:**  
_FNCE449_Final_Project.py_ --> the code file to be executed which reads from the data files that are stored in this GitHub repository in the folder labelled "Data"  
_msft.csv_ --> a data file which stores the data for the Microsoft (MSFT) stock (pulled from Databento)  
_ea.csv_ --> a data file which stores the data for the Electronic Arts (EA) stock (pulled from Databento)  
_risk_free_rates.csv_ --> a data file which stores the daily risk risk-free rates based on the 10-Year US Treasury Note (pulled from MarketWatch)  
_Graph_FigureX.png_ --> png files stored in the "Outputs" folder which contain the output graphs (Figure1 - Figure5)

**Accessing the Program:**  
The user must download the Python file labelled FNCE449_Final_Project.py. The user must ensure that they have the modules (pandas, numpy, and matplotlib) installed. If they do not, this can be done using pip.

**Executing the Program:**  
The program reads from csv files stored on this GitHub repository so the user doesn't have to worry about downloading this data. If the program is being run on an IDE such as VSCode, the user may simply hit the "run" button to execute the program. If being run through the local terminal, after navigating to the correct directory, the following is an example of how the file can be run: C:\Users\saada\Downloads\FNCE449>python .\FNCE449_Final_Project.py  

**Results/Outputs:**  
One by one, the program will output a total of five graphs. When one graph is closed, the other will pop up. The first graph plots both stocks' prices over time, the second plots the 50-day moving average and the simple price momentum, the third plots the portfolio values for both strategies over time, the fourth plots the number of trades for both strategies over time, and the fifth graph uses a bar plot to depict each strategies' annualized Sharpe Ratio.  

Note that a report is also included in the submission on D2L which provides more detail about the question being addressed, the methodologies, an interpretation of the empirical results, and insightful conclusions.
