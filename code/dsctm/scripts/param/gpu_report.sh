#!/bin/bash
# GPU occupancy on PARAM: per node how many GPUs exist, how many are allocated, how many
# are genuinely available TO YOU, and when a block of N frees up on a single node.
#
#   bash scripts/param/gpu_report.sh            # can I run anything right now?
#   bash scripts/param/gpu_report.sh -n 2       # when can --gres=gpu:2 schedule?
#   bash scripts/param/gpu_report.sh -p standard
#
# Read-only scheduler queries. Output kept under 100 columns so it screenshots cleanly.
#
# THREE COUNTING TRAPS this script exists to avoid. Each one has reported free GPUs that
# were not free:
#
#   1. `sinfo -t idle` counts DRAINED nodes. In SLURM a drained node keeps base state
#      "idle" with a DRAIN flag layered on top.
#   2. A reserved node renders as "mixed"/"allocated" the moment it has a running job, so
#      matching the state string alone lets reserved GPUs be counted as free. Reservation
#      membership must be resolved per node, not read off the state column.
#   3. Summing `squeue -o %b` misses GPUs consumed by anything that is not a plain running
#      job in this partition. AllocTRES is what the scheduler actually enforces.
#
# So: gres/gpu is read from CfgTRES/AllocTRES per node, and reservation entitlement is
# resolved against your username and your accounts.
set -uo pipefail

PART="gpu"
NEED=2
while true; do
  case "${1:-}" in
    -p|--partition) PART="${2:?--partition needs a value}"; shift 2 ;;
    -n|--need)      NEED="${2:?--need needs a value}"; shift 2 ;;
    -h|--help)      sed -n '2,9p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) break ;;
  esac
done

command -v sinfo >/dev/null 2>&1 || { echo "no SLURM here - run this on a PARAM login node"; exit 2; }

NODES=$(sinfo -h -p "$PART" -N -o "%N" 2>/dev/null | sort -u)
[[ -z "$NODES" ]] && { echo "partition '$PART' has no nodes visible to you"; exit 2; }

MY_ACCOUNTS=$(sacctmgr -n -P show associations user="$USER" format=Account 2>/dev/null \
              | grep -v '^$' | sort -u)

# ---------------------------------------------------------------- helpers
# gres/gpu out of a TRES string: "cpu=40,mem=192000M,gres/gpu=2" -> 2. Handles the typed
# form (gres/gpu:v100=2) too, which some sites configure instead.
gpu_of() {
  local v
  v=$(printf '%s' "${1:-}" | tr ',' '\n' | sed -n 's|^gres/gpu=||p'          | head -1)
  [[ -z "$v" ]] && \
  v=$(printf '%s' "${1:-}" | tr ',' '\n' | sed -n 's|^gres/gpu:[^=]*=||p'    | head -1)
  printf '%s' "${v:-0}"
}
field() { printf '%s' "${1:-}" | tr ' ' '\n' | sed -n "s|^$2=||p" | head -1; }

# ---------------------------------------------------------------- reservations
# name|nodes|entitled(yes/no)|endtime  -- entitlement decides whether a reserved GPU counts
# as available to you or to somebody else.
RES=$(scontrol -o show reservation 2>/dev/null | while read -r line; do
  [[ -z "$line" ]] && continue
  rname=$(field "$line" ReservationName); [[ -z "$rname" ]] && continue
  rnodes=$(field "$line" Nodes)
  rusers=$(field "$line" Users)
  raccts=$(field "$line" Accounts)
  rend=$(field "$line" EndTime)
  ent="no"
  [[ ",$rusers," == *",$USER,"* ]] && ent="yes"
  while read -r a; do
    [[ -n "$a" && ",$raccts," == *",$a,"* ]] && ent="yes"
  done <<< "$MY_ACCOUNTS"
  printf '%s|%s|%s|%s\n' "$rname" "$rnodes" "$ent" "$rend"
done)

# node -> "resname:entitled", empty if unreserved
res_for() {
  local n="$1" out=""
  while IFS='|' read -r rname rnodes ent _; do
    [[ -z "${rnodes:-}" || "$rnodes" == "(null)" ]] && continue
    if scontrol show hostnames "$rnodes" 2>/dev/null | grep -qx "$n"; then out="$rname:$ent"; fi
  done <<< "$RES"
  printf '%s' "$out"
}

# ---------------------------------------------------------------- per-node table
echo "==================== GPU NODES (partition: $PART) ===================="
printf "%-8s %-11s %5s %5s %5s  %-13s %s\n" NODE STATE GPUS ALLOC FREE RESERVATION "REASON / NOTE"
printf '%.0s-' {1..92}; echo

INST=0; ALLOC=0; BLOCKED=0; USABLE=0; BEST=0
declare -a USABLE_NODES=()

for n in $NODES; do
  line=$(scontrol -o show node "$n" 2>/dev/null)
  [[ -z "$line" ]] && continue
  state=$(field "$line" State)
  reason=$(printf '%s' "$line" | sed -n 's|.*Reason=\(.*\)|\1|p' | cut -d'[' -f1 | sed 's/ *$//')
  cfg=$(gpu_of "$(field "$line" CfgTRES)")
  alc=$(gpu_of "$(field "$line" AllocTRES)")
  free=$(( cfg - alc ))
  (( free < 0 )) && free=0

  rinfo=$(res_for "$n"); rname="${rinfo%%:*}"; rent="${rinfo##*:}"

  # Unavailable if the state carries DRAIN/DOWN/FAIL/MAINT/NOT_RESPONDING, or if a
  # reservation you are not entitled to covers the node.
  note="-"; avail=1
  case "$state" in
    *DRAIN*|*DOWN*|*FAIL*|*MAINT*|*NOT_RESPONDING*|*INVAL*) avail=0; note="${reason:--}" ;;
  esac
  if [[ -n "$rname" && "$rent" == "no" ]]; then
    avail=0; note="reserved for others"
  elif [[ -n "$rname" && "$rent" == "yes" ]]; then
    note="reservation is yours"
  fi

  INST=$(( INST + cfg )); ALLOC=$(( ALLOC + alc ))
  if (( avail )); then
    USABLE=$(( USABLE + free ))
    (( free > BEST )) && BEST=$free
    USABLE_NODES+=("$n")
  else
    BLOCKED=$(( BLOCKED + free ))
  fi

  printf "%-8s %-11s %5s %5s %5s  %-13s %s\n" \
    "$n" "${state:0:11}" "$cfg" "$alc" "$free" "${rname:--}" "${note:0:28}"
done

# ---------------------------------------------------------------- running jobs
echo
echo "==================== RUNNING JOBS ON $PART ===================="
printf "%-8s %-12s %-12s %-14s %-11s %-11s %-7s %s\n" \
       JOBID USER ACCOUNT NAME ELAPSED "TIME LEFT" GRES NODE
printf '%.0s-' {1..96}; echo
if [[ -n "$(squeue -h -p "$PART" -t RUNNING 2>/dev/null)" ]]; then
  squeue -h -p "$PART" -t RUNNING -o "%i|%u|%a|%j|%M|%L|%b|%N" 2>/dev/null | \
  while IFS='|' read -r id u a j m l b nn; do
    printf "%-8s %-12s %-12s %-14s %-11s %-11s %-7s %s\n" \
      "$id" "${u:0:12}" "${a:0:12}" "${j:0:14}" "$m" "$l" "${b:-—}" "$nn"
  done
else
  echo "  (none)"
fi

# ---------------------------------------------------------------- next window
# For each usable node: end its running jobs oldest-deadline-first and find the moment its
# free count first reaches NEED. That, not "when is the node empty", is when your job can
# actually start.
echo
echo "==================== NEXT WINDOW FOR ${NEED} GPU(s) ON ONE NODE ===================="
if (( BEST >= NEED )); then
  echo "  available NOW - ${BEST} free on a single usable node"
else
  found=0
  for n in "${USABLE_NODES[@]:-}"; do
    [[ -z "$n" ]] && continue
    line=$(scontrol -o show node "$n" 2>/dev/null)
    cfg=$(gpu_of "$(field "$line" CfgTRES)")
    alc=$(gpu_of "$(field "$line" AllocTRES)")
    (( cfg < NEED )) && continue          # this node can never satisfy the request
    when=$(for id in $(squeue -h -w "$n" -t RUNNING -o "%i" 2>/dev/null); do
             jl=$(scontrol show job "$id" 2>/dev/null | tr ' ' '\n')
             e=$(printf '%s' "$jl" | sed -n 's|^EndTime=||p' | head -1)
             g=$(gpu_of "$(printf '%s' "$jl" | sed -n 's|^TRES=||p' | head -1)")
             [[ -n "$e" ]] && echo "$e ${g:-0}"
           done | sort | awk -v free="$(( cfg - alc ))" -v need="$NEED" '
             { free += $2; if (free >= need) { print $1; exit } }')
    if [[ -n "$when" ]]; then
      printf "  %-8s frees %s GPU(s) at  %s\n" "$n" "$NEED" "$when"
      found=1
    fi
  done
  (( found )) || echo "  no usable node can reach ${NEED} free GPU(s) from current jobs alone"
  echo
  echo "  These are job TIME LIMITS, not predictions - jobs often end early, and the slot"
  echo "  is contended. Submit now and let SLURM backfill rather than waiting to submit."
fi

# ---------------------------------------------------------------- reservations
echo
echo "==================== RESERVATIONS COVERING $PART NODES ===================="
shown=0
while IFS='|' read -r rname rnodes ent rend; do
  [[ -z "${rnodes:-}" || "$rnodes" == "(null)" ]] && continue
  if scontrol show hostnames "$rnodes" 2>/dev/null | grep -qE "^($(echo "$NODES" | paste -sd'|' -))$"; then
    mark="NOT yours"; [[ "$ent" == "yes" ]] && mark="YOURS"
    printf "  %-14s %-9s until %-20s nodes=%s\n" "$rname" "$mark" "$rend" "${rnodes:0:40}"
    shown=1
  fi
done <<< "$RES"
(( shown )) || echo "  (none)"

# ---------------------------------------------------------------- summary
echo
echo "==================== SUMMARY ===================="
printf "  V100s installed in %-13s : %s\n" "$PART" "$INST"
printf "  held by running jobs            : %s\n" "$ALLOC"
printf "  idle but drained/down/reserved  : %s\n" "$BLOCKED"
printf "  ACTUALLY AVAILABLE TO YOU       : %s\n" "$USABLE"
printf "  largest block on a single node  : %s\n" "$BEST"
echo
if (( BEST >= NEED )); then
  echo "  --gres=gpu:${NEED} CAN schedule now."
else
  echo "  --gres=gpu:${NEED} CANNOT schedule now (needs ${NEED} free on ONE node, best is ${BEST})."
  (( USABLE > 0 )) && echo "  A --gres=gpu:${USABLE} job would still start immediately."
fi
echo
echo "  my queued jobs and estimated starts:"
if [[ -n "$(squeue --me -h -t PENDING 2>/dev/null)" ]]; then
  squeue --me --start -h -o "    %i %P %S %R" 2>/dev/null
else
  echo "    (none pending)"
fi
