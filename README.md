# Is It Possible to Satisfy the RDA? An Algorithmic Data Science Project

This project explores a fundamental, everyday problem through a unique data science lens: Can a human being actually satisfy 100% of their daily Recommended Dietary Allowances (RDAs) for all nutrients using only whole, natural foods, without exceeding a realistic caloric budget?

Rather than deploying a standard, out-of-the-box machine learning classifier or a cookie-cutter regression model, this project treats human nutrition as a classic computer science optimization problem. By reframing the human digestive tract as a **dynamic, multi-valued knapsack problem**, this repository evaluates both the satisfiability of the RDA themselves, and US MyPlate dietary guidelines.

### Project Orientation

This is a unique data science project.  It deploys algorithms to extract knowledge from existing data. But it does not utilize pre-built machine-learning models, and is not an instance of classification or prediction.   

*Technical components of the project:*

- A novel knapsack algorithm.
- A dynamic valuation function.
- Change of basis calculations (serving sizes).
- Python class-based (/object-oriented) simulations.
- Monte Carlo statistical estimation.

## The Question

Many foods are advertised as "excellent sources" of a nutrient, but provide very small amounts of that nutrient.  For example, avocados are listed among "Magnesium Rich Foods," while a medium avocado only provides 14% of the RDA ([Healthline: 10 Magnesium-Rich Foods That Are Super Healthy](https://www.healthline.com/nutrition/10-foods-high-in-magnesium#avocados)).  Statements like this raise the question of whether all RDA can be satisfied on a natural diet (particularly unprocessed "farm foods") without nutritional supplementation (vitamin pills).

This project sets out to answer two core questions:
1. **RDA Satisfiability:** Can all RDA be satisfied on a maintenance caloric budget, or are they jointly unsatisfiable?
2. **MyPlate Recommendations:** Will a person following USDA dietary guidelines naturally meet all (or most) RDA?

These questions may be considered data-scientific questions.  All the information required to answer these questions is available in existing data; no further empirical experimentation is necessary.  The information that determines these answers is contained in:
- RDA standards
- Caloric and nutritional profiles of various foods
- The average maintenance caloric load for an individual of a given age & sex

The questions are data-determined, and yet cannot be answered with elementary statistical / "pen & paper" methods. An algorithmic optimization approach is required.  However, these are *not* machine-learning problems.  They are not a question of classification or prediction.  

## The Framework
The two project questions are most similar to a classic "knapsack problem" in combinatorial optimization / computer science.  This project treats a person's gut as a knapsack, attempting to satisfy all nutrients in a given space or cost, here caloric load.  However, the standard knapsack definition is not sufficient for this problem: details in the next section.

This project utilizes the USDA National Nutrient Database for Standard Reference (SR28). 

**The Data Subset:** While the full SR28 contains over 6,000 items (many of which are branded, processed, or industrially fortified foods), this project uses a hand-curated subset of 136 core, whole natural foods (raw fruits, vegetables, meats, fish, grains, seeds, and dairy). This ensures we are testing the bounds of a natural human diet rather than synthetic fortification.

**The Change of Basis:** Standard nutritional data is presented per 100 grams. This creates a severe bias toward heavy, water-logged foods (like celery) over calorie-dense items (like nuts). To fix this, the entire dataset was re-calibrated onto a 50-calorie serving size basis. This normalizes the data, transforming every food item into a metric of *true nutrient density per calorie*.

**The Metrics:** The system explicitly tracks 26 distinct nutritional dimensions simultaneously, matching them against the baseline profile of a moderately active 28-year-old male.

## Details of the Solution
A standard knapsack problem algorithm needs two fundamental upgrades in order to tackle this project's questions.  

First, it needs to score steps to solution based on multiple values, one for each nutrient.  A standard knapsack problem might only optimize for the monetary value of items like gold bars and laptops, but this must optimize on the satisfaction for Vitamin A satisfaction, Vitamin B12, iron, etc.

Second and critically, the value of each nutrient must *change over time.*  Once the gut has met its RDA for Vitamin A, the marginal value of more Vitamin A needs to reduce to 0.  

The resulting system is a dynamic, multi-value knapsack algorithm.

Two key modules define that algorithm by defining new Python classes:

- `Gut` - A deterministic, greedy version of the algorithm.  Designed to answer the pure question of whether satisfying the RDA is possible in principle.

- `RandomGut` - A stochastic version of the algorithm.  Designed to approximate a person choosing foods according to MyPlate recommendations.

### `Gut`: The Deterministic Version
Again, this  models a dynamic value function that scores foods at each step based on the current state of nutritional deficiency. 

At any given calorie point, the algorithm computes an **Importance Factor** for each nutrient:

$$\text{Importance} = \frac{\text{Target RDA} - \text{Current Absorbed Nutrients}}{\text{Baseline RDA}}$$

The algorithm then calculates a **Utility Score** for every food item:

$$\text{Utility} = \frac{\sum (\text{Nutrient Load} \times \text{Importance})}{\text{Food Calories}}$$

This initial valuation function still led to dietary fixation on a single food, so a penalty for oversatisfaction was introduced - see the executive report for details.

The system steps through the day in 50-calorie increments, dynamically recalculating the deficit landscape and greedily choosing the single most efficient food item to plug the remaining nutritional gaps until the caloric ceiling is hit.

The `Gut` class defines a few core methods that run on a loop until satisfaction: 
- `choose_food()` - Value all foods based on current nutrient satisfaction, and choose the best.
- `add_food()` - Add a food to the "gut," storing its data in self-contained tracker.
- `continue()` - Determine whether to continue adding foods or complete the simulation.


```python
    def choose_food(self):
        '''
        Chooses food based on value function.
        t - target RDA level.
        r - scalar (<1) to ease penalty for oversatisfied nutrients.
        '''
        t = 1.5
        target = t * rda
        importance = (target - self.nutrients) / rda                 # Series
        importance = importance.map(lambda x: max(x, self.osf*x))
        increase = self.df[rda.index] / rda
        value = increase * importance                                    # DF
        value = value.mean(axis=1)                                   # Series
        value /= self.df.Calories
        value.sort_values(inplace=True)        
        food = value.index[-1]
        return food
```        
*The method for valuing and selecting a food in the Gut class.*

In addition, it determines a number of auxiliary functions for retrieving data and plotting results.  The object stores the history of food servings added, so the total nutritional progress can be plotted:

<img alt="det-gut-nutrient-sample" src="https://github.com/user-attachments/assets/137a1cb1-405b-48f0-95f6-e9635fe63545" />

*Growth of sampled nutrients in deterministic algorithm.*


### `RandomGut`: The Stochastic Person Simulator
While the mathematical optimizer is highly efficient, humans do not eat via dynamic linear programming. To test real-world behavior, the `RandomGut` class models a stochastic simulation of the official USDA MyPlate guidelines.  

`RandomGut` is a sub-class of `Gut`.  It changes the core food selection function, dispensing with iterative valuation of foods based on relative nutrient satisfaction.  It uses a simplified method that samples foods in one pass, representing an individual's diet for a single day.  A full simulation runs `RandomGut` many times and collects the results.

The 136 curated foods are mapped into structural dietary buckets: *Fruits, Vegetables, Grains, Protein, Dairy, and Snacks*. Caloric budgets are then assigned to each group based on federal guidelines (e.g., 1100 kcal for grains, 400 kcal for protein, 300 kcal for fruits). The `RandomGut` class randomly picks whole foods from within the allowed categorical buckets, logging cumulative nutritional ingestion in 50-calorie intervals until the specific category budgets—and the overall daily cap—are met. 

For example, the *MyPlate* recommendation for fruits is about 300 calories per day for the given individual.  `RandomGut` randomly chooses 6 fruits (at 50-calorie servings) for the day's diet and includes them in the total gut contents, adding their nutrients to the total accumulated nutrient load.

<img alt="rand-myplate-nutrient-growth" src="https://github.com/user-attachments/assets/f046fb41-040c-425c-9151-d03c8d959784" />

*Growth of sampled nutrients in random simulation algorithm.*

## Results
Contrary to initial expectations, this project shows that the RDA *are broadly satisfiable* on a maintenance diet.  The results also reveal interesting insights about nutritional guidelines.

### Idealized, Deterministic Gut
The mathematical optimizer demonstrates that with smart target-switching (incorporating highly dense items like spinach, morel mushrooms, and oysters), full satisfaction across 25/26 nutrients is achievable well within a standard 2,600-calorie diet.  Only fluorine could not be satisfied.

It is interesting to note that the deterministic gut exhibits four phases of eating foods.  The final honey phase demonstrates that RDA were largely satisfied in the first 1000 calories.

| Calorie Range     | Foods     |
| ---:              | ---        |
| 0 - 230           | A mix of spinach, asparagus, oysters, and mushrooms. |
| 230 - 350         | Mostly okra.
| 530 - 630         | Mostly turnips, with pickles and almonds. |
| 1020 - 2600       | Mostly honey, with a piece of pork and an egg. |

The resulting diet of this deterministic version could be considered  "Ideal Diet" in a certain sense.  

<img width="500" alt="ideal-diet" src="https://github.com/user-attachments/assets/6e4c98b9-7665-4200-8e80-4e470cb5b34d" />


### Random Gut / Human Simulation
When running a 1,000-iteration Monte Carlo simulation of the `RandomGut` (the USDA MyPlate paradigm), the average diet achieved a median nutrient satisfaction of 1.93 (nearly double the baseline requirement across all tracked nutrients). On average, **21.6 out of 26 nutrients** were satisfied by this diet. 

Unsatisfied nutrients had a mode of 4.  The diet exhibited 4 consistent nutritional shortfalls:
- Vitamin D
- Vitamin E
- Calcium
- Potassium

<img alt="random-gut-results" src="https://github.com/user-attachments/assets/4e66a8c3-bffd-44bd-a4d3-360046260c6b" />

## Why This Project is Unique

Most introductory data science repositories follow a predictable path: import a CSV, run `train_test_split`, fit a Scikit-Learn model, and plot a confusion matrix. 

This project demonstrates a completely different dimension of data science: **algorithmic problem formulation and simulation engineering.**
* It takes messy, multi-dimensional environmental data and transforms its basis to uncover hidden patterns of efficiency.
* It applies optimization theory (Knapsack constraints) to a biological domain.
* It builds a custom, state-dependent utility function to govern step-by-step decision making.
* It employs stochastic modeling to benchmark real-world public health policy against theoretical mathematical perfection.

## Repository Contents
- Executive Report, **[Is It Possible to Satisfy the RDA?](RDA_data_report.pdf)**  This report details the methodology and results of the project.

- A rough companion **[Jupyter Notebook](https://github.com/JShibby/RDA/blob/master/code/RDA_Jupyter.ipynb)**.  The notebook represents an abridged presentation of the project, and allows the user to review the information in more detail.
- 
- **Project codebase.**
