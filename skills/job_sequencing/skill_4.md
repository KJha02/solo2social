# Profit first, earliest placement

Sort jobs by decreasing profit. Process them in that order and append each job
to the earliest unused schedule position when that position meets its
deadline. Skip jobs that do not fit. This keeps high-profit jobs but does not
reconsider early placements that block tighter deadlines.
