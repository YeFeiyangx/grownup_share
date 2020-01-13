#%%
import collections

# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model


def MinimalJobshopSat():
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = [  # task = (machine_id, processing_time).
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 3), (1, 2), (2, 2)],  # Job0 需要在哪些机器上处理，以及处理时间，每个机器负责不同的工序
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(0, 2), (2, 1), (1, 4)],  # Job1
        [(1, 4), (2, 3)]  # Job2
    ]

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    # 所有job在所有机器上的总处理时间，这是一个变量的上届
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    ## 任务类型 和 持续时间
    ## @@ collections.namedtuple 十分管用
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    ## 已被分配的任务，任务索引的开始时间
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    # @@ 这玩意是个字典，但是初始化时，key都是个空list
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            ## 笛卡尔集里的每一个任务的起点时间和终点时间，以及任务持续时间
            ### NewIntVar 是CpModel中的一个类方法，IntVar在cp_model本身就是个类
            ### IntVar在pywraplp.Solver中是一个类方法
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            # all_tasks 是一个集合,也可以是个字典，关键看怎么用
            ## 这。。卧槽，运维起来方便爆了啊！！
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)
    print("all_tasksall_tasks",all_tasks)
    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))

        # Create per machine output lines.
        output = ''
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print('Optimal Schedule Length: %i' % solver.ObjectiveValue())
        print(output)


MinimalJobshopSat()


# %%
