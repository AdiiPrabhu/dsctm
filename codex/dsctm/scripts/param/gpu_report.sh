#!/bin/bash
# Complete GPU occupancy: every node, its state, why it is blocked, which reservation
# covers it, and the job holding it with time remaining.
#
#   bash scripts/param/gpu_report.sh
#
# Read-only scheduler queries. Output is kept under 100 columns so it screenshots cleanly.
set -uo pipefail

echo "==================== GPU NODES ===================="
printf "%-8s %-10s %-9s %-13s %-22s\n" NODE STATE GRES RESERVATION "REASON / BLOCKED BY"
printf '%.0s-' {1..76}; echo

# Reservation -> node expansion, so each node can be attributed.
RESMAP=$(scontrol show reservation 2>/dev/null | awk '
  /ReservationName=/ { name=$1; sub("ReservationName=","",name) }
  /Nodes=/ { for(i=1;i<=NF;i++) if($i ~ /^Nodes=/) { n=$i; sub("Nodes=","",n); print name"|"n } }')

for n in $(sinfo -h -p gpu -o "%N" | tr ',' '\n' | head -1 >/dev/null; \
           scontrol show hostnames "$(sinfo -h -p gpu -o '%N' | paste -sd, -)" 2>/dev/null); do
  state=$(sinfo -h -n "$n" -o "%T" | head -1)
  gres=$(sinfo -h -n "$n" -o "%G"  | head -1)
  reason=$(sinfo -h -n "$n" -o "%E" | head -1)
  res=""
  while IFS='|' read -r rname rnodes; do
    [[ -z "$rnodes" ]] && continue
    if scontrol show hostnames "$rnodes" 2>/dev/null | grep -qx "$n"; then res="$rname"; fi
  done <<< "$RESMAP"
  [[ "$reason" == "none" || -z "$reason" ]] && reason="-"
  printf "%-8s %-10s %-9s %-13s %-22s\n" "$n" "$state" "$gres" "${res:--}" "${reason:0:22}"
done

echo
echo "==================== RUNNING JOBS ON GPU NODES ===================="
printf "%-8s %-12s %-12s %-14s %-11s %-11s %-8s %s\n" \
       JOBID USER ACCOUNT NAME ELAPSED "TIME LEFT" GRES NODE
printf '%.0s-' {1..96}; echo
squeue -h -p gpu -t RUNNING -o "%i|%u|%a|%j|%M|%L|%b|%N" 2>/dev/null | \
while IFS='|' read -r id u a j m l b nn; do
  printf "%-8s %-12s %-12s %-14s %-11s %-11s %-8s %s\n" \
    "$id" "${u:0:12}" "${a:0:12}" "${j:0:14}" "$m" "$l" "${b:-—}" "$nn"
done
[[ -z "$(squeue -h -p gpu -t RUNNING 2>/dev/null)" ]] && echo "  (none)"

echo
echo "==================== END TIMES (when nodes free up) ===================="
for id in $(squeue -h -p gpu -t RUNNING -o "%i" 2>/dev/null); do
  scontrol show job "$id" 2>/dev/null | tr ' ' '\n' | \
    grep -E "^(JobId|UserId|EndTime|TimeLimit|NodeList)=" | paste -sd' ' -
done
[[ -z "$(squeue -h -p gpu -t RUNNING 2>/dev/null)" ]] && echo "  (none)"

echo
echo "==================== RESERVATIONS COVERING GPU NODES ===================="
scontrol show reservation 2>/dev/null | awk '
  /ReservationName=/ {
    name=$1; sub("ReservationName=","",name)
    end=""; for(i=1;i<=NF;i++) if($i ~ /^EndTime=/) { end=$i; sub("EndTime=","",end) }
  }
  /Nodes=/ && /gpu/ {
    for(i=1;i<=NF;i++) if($i ~ /^Nodes=/) { n=$i; sub("Nodes=","",n);
      printf "  %-14s until %-20s nodes=%s\n", name, end, n }
  }'

echo
echo "==================== SUMMARY ===================="
TOT=$(sinfo -h -p gpu -o "%D" | awk '{s+=$1} END{print s+0}')
sinfo -h -p gpu -o "%D %t" | awk '{printf "  %-12s %s node(s)\n", $2, $1}'
echo "  ------------------------------"
printf "  %-12s %s node(s)\n" TOTAL "$TOT"
# CAREFUL: `sinfo -t idle` is WRONG here. In SLURM a drained node keeps base state
# "idle" with a DRAIN flag on top, so -t idle counts drained nodes as available. Match on
# the rendered state string instead, which shows "drained"/"reserved"/"mixed"/"allocated".
FREE=$(sinfo -h -p gpu -N -o "%T" 2>/dev/null | grep -cx "idle")
echo
echo "  GPU nodes genuinely free : ${FREE:-0}   =  $(( ${FREE:-0} * 2 )) V100(s)"
echo
echo "  (a node shown as drained/reserved/mixed/allocated is NOT available to you;"
echo "   'mixed' means some resources on it are already taken)"

# Per-GPU accounting: gres allocated vs installed, which is what actually matters.
INST=$(sinfo -h -p gpu -N -o "%G" | grep -o '[0-9]*$' | awk '{s+=$1} END{print s+0}')
USED=$(squeue -h -p gpu -t RUNNING -o "%b" | grep -o '[0-9]*$' | awk '{s+=$1} END{print s+0}')
BLOCKED=$(sinfo -h -p gpu -N -o "%T %G" | grep -E "drain|reserved" | grep -o '[0-9]*$' \
          | awk '{s+=$1} END{print s+0}')
echo
printf "  V100s installed in partition : %s\n" "${INST:-?}"
printf "  V100s held by running jobs   : %s\n" "${USED:-0}"
printf "  V100s in drained/reserved    : %s\n" "${BLOCKED:-0}"
printf "  V100s actually free for you  : %s\n" "$(( ${INST:-0} - ${USED:-0} - ${BLOCKED:-0} ))"
echo
echo "  my queued jobs and estimated starts:"
squeue --me --start -h -o "    %i %P %S %R" 2>/dev/null || echo "    (none)"
