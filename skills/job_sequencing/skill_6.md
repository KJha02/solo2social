# Latest-slot profit scheduling

Sort all jobs by decreasing profit, breaking ties by job ID. Maintain slots
from 1 through the largest deadline. For each job, place it in the latest empty
slot no later than its deadline; skip it if no such slot exists. Return the
scheduled job IDs in increasing slot order, omitting empty slots.
