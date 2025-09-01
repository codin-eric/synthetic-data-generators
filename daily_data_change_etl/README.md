# Daily Data Change ETL

This generator simulates the behavior of sales data.  
When you run it, a daily dataset is created under the `data/` folder.  

- If no date is passed, it defaults to the current date.  
- On subsequent runs (simulating the next "day"), the generator produces a new daily export and randomly updates records from previous days.  

The challenge is to keep your **data warehouse** up to date while handling these ongoing historical changes.

---

## Potential Exercises

- **Evaluate extraction efficiency**  
  - How efficient is it to reprocess all files daily?  
  - How does this approach scale?  
  - When does it make sense to design a more incremental solution?  

- **Explore database approaches**  
  - Load the generated data into a transactional database.  
  - Experiment with solutions such as:  
    - tracking the last updated record
    - handling soft deletes  

- **Compare bulk exports vs. change data capture (CDC)**  
  - When does each approach make sense?  
  - What are the implementation and maintenance challenges?
  - How much time and cost are involved in building and supporting each approach?

---
## Usage
Check out [`etl.py`](etl.py) for a basic example of how to use the generator.

```bash
# Generate today's dataset
python generator.py

# Generate dataset for a specific date
python generator.py --start_date 2025-09-01
```

Note: --start_date will be ignored if a file already exists under the data/ folder.
In that case, the generator resumes from the latest file to simulate daily changes.
---

## Data behavior
The generator models daily sales using two **Gaussian functions**:
 simulating rush hours and slower periods.

![Gaussian distribution](images/daily_distribution.png)

Historical changes follow a **Sigmoid distribution**,  
so newer days experience more changes than older ones.

![Sigmoid distribution](images/sigmoid_1.1.2.png)

## Roadmap

- Add a database implementation that executes inserts, updates, and deletes.  
- Add support for delete operations.  
