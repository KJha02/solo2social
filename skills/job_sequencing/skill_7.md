# Verified latest-slot profit scheduling

Sort jobs by decreasing profit, breaking ties by job ID. Place each job in the
latest currently empty slot no later than its deadline, skipping jobs with no
available legal slot. Then independently verify that job IDs are unique and
that the job at output position t has deadline at least t. Remove no legal
profitable job unless required for feasibility. Return IDs in increasing slot
order with no empty placeholders.
