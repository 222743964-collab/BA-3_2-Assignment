 The problem the website addresses

Lenders and applicants both want to know how likely a loan is to go unpaid, but judging that by hand is slow and inconsistent. Many factors interact: income, existing debt, loan size, credit utilization, missed payments, length of credit history and housing situation. A person can’t easily weigh them all at once, and the result is often a vague yes or no with no explanation.

The solution

The site takes those details through a simple form and returns a risk estimate in seconds.
Data: credit_data.csv holds 8,000 synthetic applicants, each labelled as having defaulted or not. I generated it so that risk rises with high debt, high utilization and missed payments, and falls with long credit history, stable employment and home ownership.
Model: a Gradient Boosting classifier (scikit-learn) learns those patterns from the data. It also calculates two ratios from the inputs, debt-to-income and loan size versus income.
Backend: a Flask server in app.py validates the inputs, runs the model and returns the default probability, a risk band, a 300–850 score and the top factors behind the result.
Frontend: a single page with a gauge, the percentage, the band, and a short list showing how many percentage points each factor adds to or removes from the risk compared with a typical applicant.
