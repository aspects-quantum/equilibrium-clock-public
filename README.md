# equilibrium-clock-public

- Uncomment the desired function in main.py() to replot a figure from the paper
    - The data_prepro() function is there to read the original measurement data and convert it into discrete dates
    - The rates_postpro() function is there to extract the tunneling rates
    - The times_postpro() function is there to extract the left-right vs right-left waiting times
    - All following functions are for creating figures

- Further comments
    - main.py
    - data_loader.py contains code for reading measurement files
    - data_converter.py contains code for pre-processing the measurement data
    - num_stat.py contains code for analyzing jump statistics
    - reflecto_pwr.py contains code for determining dissipation of the rf method
    - theory_model.py contains code for modelling the ME model
    - visualizations.py contains code for plotting the paper figures

- Note: the code in data_converter.py for identifying states and counting LR vs RL cycles is based on David Craig's PhD thesis, doi:10.5287/ora-nbx0jzppy and shared here with his kind permission