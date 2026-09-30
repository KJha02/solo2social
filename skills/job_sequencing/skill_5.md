# Profit-deadline heuristic

Rank jobs primarily by profit and secondarily by earlier deadline. Build a
feasible schedule incrementally. When a new job conflicts, compare its profit
with the lowest-profit scheduled job and keep the better one. Sort the retained
jobs by deadline, verify every position against its deadline, and drop any job
that still violates feasibility.
