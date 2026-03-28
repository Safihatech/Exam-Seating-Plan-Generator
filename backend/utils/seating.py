import pandas as pd
import math
import re
from collections import deque, defaultdict

def parse_roll_number(roll_value):
    if pd.isna(roll_value):
        return float('inf')
    text = str(roll_value).strip()
    if not text:
        return float('inf')
    try:
        return int(text)
    except ValueError:
        digits = ''.join(ch for ch in text if ch.isdigit())
        if digits:
            return int(digits)
        return float('inf')

def normalize_branch(branch):
    if branch is None:
        return 'UNKNOWN'
    text = str(branch).strip()
    return text.upper() if text else 'UNKNOWN'


def is_year_valid(year):
    return isinstance(year, int)


def pop_student_with_year_constraint(queue, excluded_years=None):
    if not queue:
        return None
    if not excluded_years:
        return queue.popleft()
    for idx, student in enumerate(queue):
        year_val = student.get('year')
        if year_val is None or year_val not in excluded_years:
            queue.rotate(-idx)
            student = queue.popleft()
            queue.rotate(idx)
            return student
    return None


def branch_has_candidate(branch_queues, branch, excluded_years=None):
    if branch not in branch_queues or not branch_queues[branch]:
        return False
    if not excluded_years:
        return True
    for student in branch_queues[branch]:
        year_val = student.get('year')
        if year_val is None or year_val not in excluded_years:
            return True
    return False


def get_adjacent_years(grid, row_idx, col_idx):
    years = set()
    for dr, dc in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
        nr, nc = row_idx + dr, col_idx + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]):
            neighbor = grid[nr][nc]
            if neighbor:
                for student in neighbor:
                    year_val = student.get('year')
                    if isinstance(year_val, int):
                        years.add(year_val)
    return years


def pick_next_branch(branches, branch_queues, start_idx, disallowed):
    if not branches:
        return None, start_idx
    total = len(branches)
    idx = start_idx % total
    fallback = None
    fallback_idx = None
    checked = 0

    while checked < total:
        branch = branches[idx]
        if branch_queues[branch]:
            if branch not in disallowed:
                return branch, (idx + 1) % total
            if fallback is None:
                fallback = branch
                fallback_idx = idx
        idx = (idx + 1) % total
        checked += 1

    if fallback is not None:
        return fallback, (fallback_idx + 1) % total
    return None, start_idx

def generate_seating_plan(student_csv_path, room_csv_path, students_per_desk=1, include_detained=False, sequential_fill=True, fill_order='row'):
    """
    Generate seating plan with alternating branches and sequential roll numbers.
    """
    try:
        # small helper to safely parse integers (handles NaN, empty strings, etc.)
        def safe_int(value, default):
            try:
                if value is None:
                    return default
                if pd.isna(value):
                    return default
                if isinstance(value, str):
                    text = value.strip()
                    if text == '':
                        return default
                    # allow numbers inside strings
                    return int(float(text))
                # numeric types
                return int(value)
            except Exception:
                return default

        students_df = pd.read_csv(student_csv_path)
        rooms_df = pd.read_csv(room_csv_path)

        students_df.columns = students_df.columns.str.lower().str.strip()
        rooms_df.columns = rooms_df.columns.str.lower().str.strip()

        detained_column = None
        for col in students_df.columns:
            if 'detained' in col.lower():
                detained_column = col
                break

        if include_detained or not detained_column:
            filtered_students = students_df.copy()
        else:
            detained_mask = students_df[detained_column].astype(str).str.upper().isin(['TRUE', '1', 'YES', 'Y'])
            filtered_students = students_df[~detained_mask]

        if len(filtered_students) == 0:
            return [], [], "No students available for seating."

        total_students_available = len(filtered_students)

        students_by_branch = defaultdict(list)
        for _, student in filtered_students.iterrows():
            roll_value = student.get('roll_number', student.get('id', 'N/A'))
            roll_text = str(roll_value).strip() if not pd.isna(roll_value) else 'N/A'
            branch = normalize_branch(student.get('branch', student.get('department', '')))

            year_value = student.get('year', student.get('academic_year', None))
            year = safe_int(year_value, None) if pd.notna(year_value) else None
            student_info = {
                'roll_number': roll_text if roll_text else 'N/A',
                'name': str(student.get('name', 'Unknown')).strip() or 'Unknown',
                'branch': branch,
                'year': year,
                'section': str(student.get('section', 'N/A')).strip() or 'N/A',
                'detained': False if not detained_column else str(student.get(detained_column, 'FALSE')).upper() in ['TRUE', '1', 'YES', 'Y']
            }
            students_by_branch[branch].append(student_info)

        if not students_by_branch:
            return [], [], "No valid students found after grouping by branch."

        for branch, records in students_by_branch.items():
            records.sort(key=lambda s: (parse_roll_number(s['roll_number']), str(s['roll_number']).strip()))

        branch_queues = {branch: deque(records) for branch, records in students_by_branch.items()}
        # preserve the branch order as encountered in the students file (do not sort)
        branches = list(branch_queues.keys())

        if not branches:
            return [], [], "No valid branches found."

        # helper to parse allowed branches from rooms CSV
        def parse_allowed_branches(value):
            if pd.isna(value):
                return None
            text = str(value).strip()
            if not text:
                return None
            # support separators ; or , and preserve ordering
            parts = [p.strip() for p in re.split('[;,]', text) if p.strip()]
            # normalize each part to the same format used for student branches,
            # dedupe while preserving order
            norm_parts = []
            for p in parts:
                npart = normalize_branch(p)
                if npart not in norm_parts:
                    norm_parts.append(npart)
            return norm_parts if norm_parts else None

        rooms = []
        total_capacity = 0
        for idx, room in rooms_df.iterrows():
            room_number = str(room.get('room_number', room.get('room_no', f"Room_{idx + 1}")))
            capacity = safe_int(room.get('capacity', room.get('seats', 30)), 30)
            total_capacity += capacity
            # try to find explicit rows/cols in room data
            def find_int_field(r, candidates):
                for c in candidates:
                    if c in rooms_df.columns:
                        val = r.get(c)
                        v = safe_int(val, None)
                        if v is not None and v > 0:
                            return v
                return None

            cols_val = find_int_field(room, ['cols', 'columns', 'num_cols', 'col_count', 'columns_count', 'seats_per_row'])
            rows_val = find_int_field(room, ['rows', 'num_rows', 'row_count', 'rows_count'])

            room_info = {
                'room_number': room_number,
                'room_name': str(room.get('room_name', room_number)),
                'capacity': capacity,
                'building': str(room.get('building', 'Main Building')),
                'floor': str(room.get('floor', '1')),
                'rr_idx': 0,
                'cols': cols_val,
                'rows': rows_val
            }
            # read allowed_branches column if present (e.g. "CS;ME" or "CS,ME")
            allowed_val = None
            # try common column names
            for colname in ['allowed_branches', 'allowed', 'allowed_branches_list', 'branches']:
                if colname in rooms_df.columns:
                    allowed_val = room.get(colname)
                    break
            # also check for a column that mentions allowed and branch
            if allowed_val is None:
                for col in rooms_df.columns:
                    if 'allow' in col.lower() and 'branch' in col.lower():
                        allowed_val = room.get(col)
                        break
            # parse allowed branches into a list of uppercase names (preserve order) or None
            room_info['allowed_branches'] = parse_allowed_branches(allowed_val)
            rooms.append(room_info)

        rooms.sort(key=lambda x: x['room_number'])

        try:
            students_per_desk = int(students_per_desk)
        except (TypeError, ValueError):
            students_per_desk = 1
        students_per_desk = max(1, students_per_desk)

        seating_plan = []
        total_seated = 0
        branch_distribution = defaultdict(int)
        branch_idx = 0
    # keep a short history of recently used branches to avoid repeating the same branch
    # across consecutive desks (helps distribute branches more evenly)
        # You can change recent_history_len to increase/decrease how many recent desks to avoid
        recent_history_len = 3
        recent_branches = deque(maxlen=recent_history_len)

        for room in rooms:
            if not any(branch_queues[branch] for branch in branches):
                break

            room_capacity = room['capacity']

            # Determine seating grid based on per-room rows/cols if provided.
            cols = room.get('cols')
            rows = room.get('rows')

            if cols and rows:
                # explicit grid provided
                seats_per_row = int(cols)
                grid_rows = int(rows)
                max_slots = grid_rows * seats_per_row
                # effective capacity limits seating to the smaller of declared capacity and grid slots
                effective_capacity = min(room_capacity, max_slots)
                seat_grid = [[None for _ in range(seats_per_row)] for _ in range(grid_rows)]
            else:
                # fallback: use cols if only cols provided
                if cols:
                    seats_per_row = int(cols)
                    grid_rows = math.ceil(room_capacity / seats_per_row) if room_capacity else 0
                else:
                    # default behavior (legacy): 6 seats per row
                    seats_per_row = 6
                    grid_rows = math.ceil(room_capacity / seats_per_row) if room_capacity else 0

                seat_grid = [[None for _ in range(seats_per_row)] for _ in range(grid_rows)]
                effective_capacity = room_capacity
            seat_assignments = []
            seat_number = 1
            last_branch_used = None

            # compute allowed branches for this room (None means all branches allowed)
            allowed = room.get('allowed_branches')
            # iterate over seat positions up to the effective capacity and available students
            while seat_number <= (effective_capacity if 'effective_capacity' in locals() else room_capacity) and any(branch_queues[branch] for branch in branches):
                if fill_order == 'column':
                    col_idx = (seat_number - 1) // grid_rows if grid_rows > 0 else 0
                    row_idx = (seat_number - 1) % grid_rows if grid_rows > 0 else 0
                else:
                    row_idx = (seat_number - 1) // seats_per_row
                    col_idx = (seat_number - 1) % seats_per_row
                adjacent_years = get_adjacent_years(seat_grid, row_idx, col_idx)

                active_branches = {b for b in branches if branch_queues[b]}
                if not active_branches:
                    break

                # If only one branch remains overall for this room and the current row
                # doesn't have enough seats left to form a full desk, skip to next row
                # (leave this partially filled row empty) to reduce clustering.
                remaining_allowed = [b for b in branches if branch_queues[b] and (allowed is None or b in allowed)]
                if len(remaining_allowed) == 1:
                    seats_left_in_row = seats_per_row - ((seat_number - 1) % seats_per_row)
                    # if current row is partially filled and can't fit a full desk
                    if seats_left_in_row != seats_per_row and seats_left_in_row < students_per_desk:
                        # compute start of next row
                        next_row_start = (((seat_number - 1) // seats_per_row) + 1) * seats_per_row + 1
                        if next_row_start <= room_capacity:
                            seat_number = next_row_start
                            # restart loop with updated seat_number
                            continue

                # determine available branches for this room
                if allowed is not None:
                    allowed_list = allowed
                    # round-robin primary selection using per-room index and adjacent year constraints
                    found = False
                    for i in range(len(allowed_list)):
                        idx = (room['rr_idx'] + i) % len(allowed_list)
                        b = allowed_list[idx]
                        if b in branch_queues and branch_has_candidate(branch_queues, b, adjacent_years):
                            branch_choice = b
                            # advance rr_idx for next primary pick
                            room['rr_idx'] = (idx + 1) % len(allowed_list)
                            found = True
                            break
                    if not found:
                        # fallback to any allowed branch with students
                        for i in range(len(allowed_list)):
                            idx = (room['rr_idx'] + i) % len(allowed_list)
                            b = allowed_list[idx]
                            if b in branch_queues and branch_queues[b]:
                                branch_choice = b
                                room['rr_idx'] = (idx + 1) % len(allowed_list)
                                found = True
                                break
                    if not found:
                        # no allowed branch has students; stop seating in this room
                        break
                else:
                    # no allowed list: fall back to previous behavior (balanced pick)
                    available_branches = [b for b in branches if branch_queues[b]]
                    if not available_branches:
                        break
                    candidates = [b for b in available_branches if b not in recent_branches and branch_has_candidate(branch_queues, b, adjacent_years)]
                    if not candidates:
                        candidates = [b for b in available_branches if b not in recent_branches]
                    if not candidates:
                        candidates = [b for b in available_branches if branch_has_candidate(branch_queues, b, adjacent_years)]
                    if not candidates:
                        candidates = available_branches[:]
                    branch_choice = candidates[0]

                primary_student = pop_student_with_year_constraint(branch_queues[branch_choice], adjacent_years)
                if primary_student is None:
                    # no safe primary student for this seat due year constraints
                    seat_number += 1
                    continue
                desk_students = [primary_student]
                desk_branches = {branch_choice}
                desk_years = {primary_student['year']} if is_year_valid(primary_student.get('year')) else set()
                while len(desk_students) < students_per_desk:
                    disallowed_years = set(desk_years) | adjacent_years
                    if allowed is not None:
                        allowed_list = allowed
                        extra_choice = None
                        for j in range(len(allowed_list)):
                            idx = (room['rr_idx'] + j) % len(allowed_list)
                            b = allowed_list[idx]
                            if b not in desk_branches and b in branch_queues and branch_has_candidate(branch_queues, b, disallowed_years):
                                extra_choice = b
                                break
                        if extra_choice is None:
                            alt_candidates = [b for b in allowed_list if b in branch_queues and branch_has_candidate(branch_queues, b, disallowed_years)]
                            if not alt_candidates:
                                break
                            branch_choice_extra = alt_candidates[0]
                        else:
                            branch_choice_extra = extra_choice
                    else:
                        extra_candidates = [b for b in branches if b not in desk_branches and branch_queues[b] and branch_has_candidate(branch_queues, b, disallowed_years)]
                        if not extra_candidates:
                            alt_candidates = [b for b in branches if branch_queues[b] and branch_has_candidate(branch_queues, b, disallowed_years)]
                            if not alt_candidates:
                                break
                            branch_choice_extra = alt_candidates[0]
                        else:
                            branch_choice_extra = extra_candidates[0]

                    extra_student = pop_student_with_year_constraint(branch_queues[branch_choice_extra], disallowed_years)
                    if extra_student is None:
                        break
                    desk_students.append(extra_student)
                    desk_branches.add(branch_choice_extra)
                    if is_year_valid(extra_student.get('year')):
                        desk_years.add(extra_student['year'])

                # Simple sequential placement (no adjacency/spacing enforcement)
                # compute row/col depending on fill_order
                if fill_order == 'column':
                    # column-major: fill down columns then across
                    col_idx = (seat_number - 1) // grid_rows if grid_rows > 0 else 0
                    row_idx = (seat_number - 1) % grid_rows if grid_rows > 0 else 0
                else:
                    # default row-major: fill across rows then down
                    row_idx = (seat_number - 1) // seats_per_row
                    col_idx = (seat_number - 1) % seats_per_row
                # only assign if position exists in the grid (protect against mismatched capacity/grid)
                if 0 <= row_idx < len(seat_grid) and 0 <= col_idx < len(seat_grid[row_idx]):
                    seat_grid[row_idx][col_idx] = desk_students
                placed_seat = seat_number

                # For display only: reorder students at each desk according to branch display order
                room_allowed = room.get('allowed_branches')
                if room_allowed:
                    display_order = room_allowed
                else:
                    display_order = list(branches)
                def branch_index(b):
                    try:
                        return display_order.index(b)
                    except ValueError:
                        return len(display_order)

                desk_students_sorted = sorted(
                    desk_students,
                    key=lambda s: (branch_index(s['branch']), parse_roll_number(s['roll_number']))
                )

                seat_assignments.append({
                    'seat_number': placed_seat,
                    'students': desk_students_sorted
                })

                for assigned in desk_students:
                    branch_distribution[assigned['branch']] += 1
                total_seated += len(desk_students)
                last_branch_used = desk_students[0]['branch']
                seat_number += 1

            room_students_count = sum(len(seat['students']) for seat in seat_assignments)
            room_entry = {
                'room_number': room['room_number'],
                'room_name': room['room_name'],
                'building': room['building'],
                'floor': room['floor'],
                'capacity': room_capacity,
                'seats': seat_grid,
                'seat_assignments': seat_assignments,
                'students_count': room_students_count
            }
            seating_plan.append(room_entry)

        unseated_students = []
        for branch in branches:
            unseated_students.extend(dict(student) for student in branch_queues[branch])

        summary_stats = {
            'total_students_available': total_students_available,
            'total_students_seated': total_seated,
            'total_unseated': len(unseated_students),
            'total_capacity': total_capacity,
            'total_rooms': sum(1 for room in seating_plan if room['students_count'] > 0),
            'overall_utilization': round((total_seated / total_capacity * 100), 2) if total_capacity else 0,
            'branch_distribution': dict(branch_distribution),
            'students_per_desk': students_per_desk,
            'include_detained': include_detained
        }

        print("\n=== SEATING PLAN SUMMARY ===")
        print(f"Total students available: {total_students_available}")
        print(f"Total students seated: {total_seated}")
        print(f"Total unseated students: {len(unseated_students)}")
        print(f"Total rooms used: {summary_stats['total_rooms']}")
        print(f"Overall utilization: {summary_stats['overall_utilization']}%")
        print(f"Branch distribution: {summary_stats['branch_distribution']}")
        print(f"Students per desk setting: {students_per_desk}")

        return (seating_plan, summary_stats), unseated_students, None

    except Exception as e:
        print(f"Error in generate_seating_plan: {e}")
        import traceback
        traceback.print_exc()
        return [], [], f"Error generating seating plan: {str(e)}"

def get_room_summary(seating_plan):
    summary = []
    total_students = 0
    for room in seating_plan:
        count = room['students_count']
        total_students += count
        summary.append({
            'room_number': room['room_number'],
            'room_name': room['room_name'],
            'capacity': room['capacity'],
            'students_seated': count,
            'utilization': f"{(count / room['capacity'] * 100):.1f}%" if room['capacity'] else "0.0%"
        })

    return summary, total_students
