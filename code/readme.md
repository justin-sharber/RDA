## Code Files
The code includes the following files, which may be treated as having a linear order, as later notebooks have some dependencies on earlier ones.
1. **data.py** - Imports and processes the data.  Creates these objects:
    * `df` - Pandas Dataframe of foods and nutrients.
    * `rda` - Pandas Series of nutrient RDA.
    * `food_finder()` - Helps the user locate a food by name in the dataframe.
    * `recal_df()` - Adjusts the dataframe to a new serving size, either in grams or calories.

2. **explore_functions.py** - Creates food analysis function for visualizing nutrients.
    * `analyze_food()` - Plots a foods top nutrients, and returns a sorted list of the nutrients it contains.
                                 
5. **Gut.py** - Creates the `Gut` class.  Tests satisfiability and constructs an ideal diet.

6. **RandomGut.py** - `RandomGut` is a subclass of `Gut` that chooses foods from food groups randomly, according to dietary plan.

7. **sim RandomGut.py** - Tests the MyPlate dietary recommendations by simulating `RandomGut` multiple times and aggregating results.
