#!/bin/bash
# One-shot cluster resource report: what exists, what is blocked and why, who is using it.
#
#   bash scripts/param/cluster_status.sh          # everything
#   bash scripts/param/cluster_status.sh gpu      # focus one partition
#
# Safe on a login node: a handful of scheduler queries, no compute.
set -uo pipefail
FOCUS="${1:-}"
line() { printf '%.0s─' {1..78}; echo; }
hdr()  { echo; line; echo "  $*"; line; }

hdr "PARTITIONS — capacity and state"
# %F is allocated/idle/other/total, which is the single most useful column here.
sinfo -o "%-12P %5D %14F %6c %8m %12G %10l" | head -20
echo
echo "  A/I/O/T = Allocated / Idle / Other(down,drain,resv) / Total"

hdr "WHY NODES ARE BLOCKED  (the important one)"
if out=$(sinfo -R --noheader 2>/dev/null) && [[ -n "$out" ]]; then
  sinfo -R -o "%-32H %-10u %-12n %S" 2>/dev/null | head -40 \
    || sinfo -R | head -40
else
  echo "  no nodes are down/drained/failed"
fi

hdr "GPU NODES — per-node detail"
sinfo -p gpu -N -o "%-10n %-10T %8c %8m %-14G %-16E" 2>/dev/null | head -20
echo
echo "  T=state  G=generic resources (gres)  E=reason if unavailable"

hdr "GPUS IN USE RIGHT NOW"
squeue -p gpu -t RUNNING -o "%.10i %-12u %-16j %.6D %-14b %.11M %.11L %R" 2>/dev/null | head -20
echo
GTOT=$(sinfo -h -p gpu -o "%D" | awk '{s+=$1} END{print s+0}')
GALLOC=$(squeue -h -p gpu -t RUNNING -o "%b" 2>/dev/null | grep -o '[0-9]*$' | awk '{s+=$1} END{print s+0}')
echo "  gpu partition nodes: ${GTOT:-?}    GPUs allocated by running jobs: ${GALLOC:-0}"

hdr "RESERVATIONS  (these block nodes from normal jobs)"
if scontrol show reservation 2>/dev/null | grep -q ReservationName; then
  scontrol show reservation 2>/dev/null \
    | grep -E "ReservationName|Nodes=|StartTime|Users=|State" | head -40
else
  echo "  none visible to this account"
fi

hdr "QUEUE — everyone"
squeue -o "%.10i %-10P %-12u %-16j %.2t %.11M %.11L %.5D %-14b %R" 2>/dev/null | head -25
echo
echo "  pending by partition:"
squeue -h -t PD -o "%P" 2>/dev/null | sort | uniq -c | sort -rn | sed 's/^/    /'

hdr "MY JOBS"
squeue --me -o "%.10i %-10P %-18j %.2t %.11M %.11L %.5D %-12b %R" 2>/dev/null
echo
echo "  estimated starts:"
squeue --me --start -o "%.10i %-10P %20S %R" 2>/dev/null | tail -n +1

hdr "MY LIMITS AND SHARE"
echo "association:"; sacctmgr -P show associations user="$USER" \
  format=Account,Partition,QOS,GrpTRES,MaxJobs,MaxSubmit 2>/dev/null | sed 's/^/  /'
echo; echo "fairshare:"; sshare -U -u "$USER" 2>/dev/null | head -6 | sed 's/^/  /'
echo; echo "QOS limits:"
QOS=$(sacctmgr -n -P show associations user="$USER" format=QOS 2>/dev/null | head -1)
[[ -n "$QOS" ]] && sacctmgr -P show qos "${QOS%%,*}" \
  format=Name,MaxWall,MaxTRESPerUser,MaxJobsPerUser,GrpTRES 2>/dev/null | sed 's/^/  /'

hdr "MY USAGE (last 7 days)"
sacct -u "$USER" -S now-7days -X \
  -o JobID%10,JobName%18,Partition%10,State%14,Elapsed%11,AllocTRES%34 2>/dev/null | head -25
echo
echo "  total node-seconds consumed:"
sacct -u "$USER" -S now-7days -X -n -P -o ElapsedRaw,NNodes 2>/dev/null \
  | awk -F'|' '{s+=$1*$2} END{printf "    %.2f node-hours\n", s/3600}'

hdr "STORAGE"
for d in "$HOME" "${DSCTM_SCRATCH:-/scratch/$USER}" /scratch; do
  [[ -d "$d" ]] && df -h "$d" 2>/dev/null | tail -1 | awk -v p="$d" \
    '{printf "  %-34s %6s used of %-6s (%s)  avail %s\n", p, $3, $2, $5, $4}'
done
command -v lfs >/dev/null 2>&1 && { echo; echo "  lustre quota:"; \
  lfs quota -h -u "$USER" /scratch 2>/dev/null | sed 's/^/    /'; }

if [[ -n "$FOCUS" ]]; then
  hdr "FOCUS: $FOCUS"
  sinfo -p "$FOCUS" -N -l 2>/dev/null | head -30
fi

echo; line
echo "  Reading this: 'Other' in A/I/O/T means down+drained+reserved -- capacity you"
echo "  cannot use. If a partition shows 0 Idle and high Other, the queue estimate is"
echo "  optimistic. 'WHY NODES ARE BLOCKED' gives the actual reason per node."
line
