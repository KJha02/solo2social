# Input-order scheduling

Read jobs in the order presented. Append a job when its deadline is at least
the next schedule position; otherwise skip it. Stop after filling the available
deadline horizon. Preserve the selected jobs' input order.
