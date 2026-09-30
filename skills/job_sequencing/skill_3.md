# Earliest deadlines first

Sort jobs by increasing deadline, breaking ties by higher profit. Scan that
order and append a job when its deadline is at least the next schedule
position. Skip jobs that no longer fit. Return the resulting feasible order.
