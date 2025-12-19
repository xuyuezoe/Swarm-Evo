import asyncio
import time
import json
from typing import List, Dict, Any, Optional
import uuid
from core.execution.pipeline import Pipeline
from core.agent.agent_pool import AgentPool
from core.agent.base_agent import BaseReActAgent
from core.execution.journal import Journal, Node
from core.agent.prompt_manager import PromptContext
from core.memory.local_experience import generate_local_report, select_top_nodes
from utils.logger_system import log_msg

class IterationController:
    def __init__(
        self,
        agent_pool: AgentPool,
        task_pipeline: Pipeline,
        journal: Journal,
        config: Any,
        competition_description: str = "",
        global_prior: str = "",
    ):
        self.agent_pool = agent_pool
        self.task_pipeline = task_pipeline
        self.journal = journal
        self.config = config
        self.competition_description = competition_description
        self.global_prior = global_prior

        self.current_epoch = 0
        self.start_time = time.time()

    async def run_competition(self):
        log_msg("INFO", "Starting competition loop...")
        while self.current_epoch < self.config.mle_bench_epoch_limit:
            self.current_epoch += 1
            log_msg("INFO", f"--- Starting Epoch {self.current_epoch} ---")
            await self.run_epoch()
        
        self.task_pipeline.finalize_task(current_step=self.current_epoch)
        log_msg("INFO", "Competition loop finished.")

    async def run_epoch(self):
        """
        Run one epoch:
        1. Identify available agents.
        2. Fetch tasks from pipeline.
        3. Execute tasks concurrently.
        4. Save results to journal.
        """
        tasks_coroutines = []
        
        agents = list(self.agent_pool.agents.values())
        
        # Concurrency control: Limit the number of active agents
        max_concurrent = self.config.max_concurrent_agents
        active_agents = agents[:max_concurrent]
        
        log_msg("INFO", f"Epoch {self.current_epoch}: concurrency limit = {max_concurrent}, active agents = {len(active_agents)}")

        for agent in active_agents:
            task_item = self.task_pipeline.get_task()
            if task_item:
                task_item['agent_name'] = agent.name
                tasks_coroutines.append(self._run_single_task(agent, task_item))
            else:
                pass
        
        if tasks_coroutines:
            log_msg("INFO", f"Epoch {self.current_epoch}: Executing {len(tasks_coroutines)} tasks.")
            await asyncio.gather(*tasks_coroutines)
        else:
            log_msg("INFO", f"Epoch {self.current_epoch}: No tasks to execute.")
            await asyncio.sleep(1)

    async def _run_single_task(self, agent: BaseReActAgent, task: Dict[str, Any]):
        """
        Execute a single task with a specific agent.
        """
        task_id = task['id']
        task_type = task['type']
        log_msg("INFO", f"Agent {agent.name} starting task {task_id} ({task_type})")
        
        # 1. 准备上下文 (Merge 任务需要先获取 gene_plan)
        if task_type == 'merge':
            source_id = task.get('dependencies', {}).get('gene_plan_source')
            if source_id:
                gene_plan = self.task_pipeline.retrieve_result(source_id, pop=True)
                if gene_plan:
                    task['payload']['gene_plan'] = gene_plan
                else:
                    log_msg("WARNING", f"Merge task {task_id} missing gene_plan from {source_id}")

        prompt_context = self._construct_prompt_context(task)
        
        task_description = self._get_task_description(task)
        
        agent_input_state = {
            "task_description": task_description,
            "prompt_context": prompt_context
        }

        try:
            result = await agent(agent_input_state)
            
            # 2. 根据任务类型处理结果
            result_nodes = []
            update_data = None

            if task_type == 'select':
                # Select 任务: 暂存结果，创建 Merge 任务
                agent_output = result.get('agent_output', {})
                gene_plan = agent_output 
                
                # 暂存 Plan
                self.task_pipeline.store_result(task_id, gene_plan)
                
                # 创建 Merge 任务
                merge_payload = {
                    "candidate_ids": task['payload'].get('candidate_ids', [])
                }
                
                # 手动构建 Merge 任务并优先添加
                merge_task = {
                    "id": str(uuid.uuid4()),
                    "type": "merge",
                    "priority": 0, # Merge 优先
                    "payload": merge_payload,
                    "status": "pending",
                    "created_at": time.time(),
                    "agent_name": None,
                    "dependencies": {"gene_plan_source": task_id}
                }
                
                self.task_pipeline.add_urgent_task(merge_task)
                log_msg("INFO", f"Select task finished. Merge task {merge_task['id']} created.")

            elif task_type == 'review':
                # Review 任务: 准备更新数据传给 Pipeline
                target_id = task['payload'].get('target_node_id')
                if target_id:
                    agent_output = result.get('agent_output', {})
                    update_data = {
                        "score": agent_output.get('score'),
                        "summary": agent_output.get('summary', ""),
                        "is_bug": agent_output.get('is_bug', False),
                        "agent_success": result.get('agent_success')
                    }
                else:
                     log_msg("WARNING", f"Review task target node {target_id} not found.")

            else:
                # Explore / Merge 任务: 创建新 Node 对象但不直接添加
                result_nodes = self._create_nodes_from_result(result, task)
                # self.journal.add_node(result_node) -> 移交给 pipeline
                
                log_msg("INFO", f"Agent {agent.name} finished task {task_id}. Generated {len(result_nodes)} Nodes: {[n.id for n in result_nodes]}")

            # 3. 完成任务，并将结果传递给 Pipeline 处理 (Journal 更新, 后续任务触发)
            self.task_pipeline.complete_task(
                task_id=task_id,
                result_nodes=result_nodes if result_nodes else [], # Pass list of nodes
                update_data=update_data
            )
            
        except Exception as e:
            log_msg("ERROR", f"Agent {agent.name} failed task {task_id}: {e}")
            import traceback
            log_msg("ERROR", traceback.format_exc())

    def _construct_prompt_context(self, task: Dict[str, Any]) -> PromptContext:
        """
        Build PromptContext from task payload and global config.
        """
        payload = task.get('payload', {})
        
        elapsed = time.time() - self.start_time
        remaining = self.config.time_limit_seconds - elapsed

        if payload.get('template_name') == None:
            if task['type'] == 'review':
                payload['template_name']  = "evaluate_user_prompt.j2"
            elif task['type'] == 'select':
                payload['template_name']  = "select_user_prompt.j2"
            elif task['type'] == 'merge':
                payload['template_name']  = "merge_user_prompt.j2"
            else:
                payload['template_name'] = "explore_user_prompt.j2"
        
        # 动态填充数据
        candidates_data = {}
        gene_plan_data = {}
        solution_code = None
        execution_logs = None

        local_report = ""
        context_candidates = None
        if task['type'] == 'select':
            gene_normalizer = getattr(self.task_pipeline, "gene_normalizer", None)
            if gene_normalizer:
                top_nodes = select_top_nodes(self.journal, gene_normalizer, topn=5)
            else:
                top_nodes = []
            candidates_data = {n.id: n.code for n in top_nodes if n.code}
            payload['candidate_ids'] = list(candidates_data.keys())
            local_report = generate_local_report(top_nodes, gene_normalizer) if gene_normalizer else ""
            context_candidates = candidates_data
        
        elif task['type'] == 'merge':
            # 从 Journal 获取 Candidates
            candidate_ids = payload.get('candidate_ids', [])
            for cid in candidate_ids:
                node = self.journal.get_node(cid)
                if node and node.code:
                    candidates_data[cid] = node.code
            
            # Gene Plan 从 Payload (run_single_task 中注入)
            gene_plan_data = payload.get('gene_plan')

        elif task['type'] == 'review':
            target_id = payload.get('target_node_id')
            if target_id:
                node = self.journal.get_node(target_id)
                if node:
                    solution_code = node.code
                    execution_logs = node.logs # 假设 logs 存在 Node 中

        if context_candidates is None:
            context_candidates = candidates_data if candidates_data else payload.get('candidates')

        return PromptContext(
            workspace_root=self.config.mle_bench_workspace_dir,
            conda_env_name=self.config.conda_env_name,
            time_limit_seconds=self.config.time_limit_seconds,
            total_iterations=self.config.mle_bench_epoch_limit,
            iteration=self.current_epoch,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining,
            conda_packages="", 
            task_description=self._get_task_description(task),
            
            parent_code=payload.get('parent_code'),
            parent_feedback=payload.get('parent_feedback'),
            candidates=context_candidates,
            gene_plan=gene_plan_data if gene_plan_data else payload.get('gene_plan'),
            solution_code=solution_code if solution_code else payload.get('solution_code'),
            execution_logs=execution_logs if execution_logs else payload.get('execution_logs'),
            global_prior=self.global_prior,
            local_report=local_report,
            template_name=payload.get('template_name') 
        )

    def _get_task_description(self, task: Dict[str, Any]) -> str:
        """
        Generate a human-readable task description based on task type.
        Prepends competition description as background context.
        """
        t_type = task['type']
        if t_type == 'select':
            task_instruction = "Please select the best solution strategy from the candidates."
        elif t_type == 'explore':
            task_instruction = "Please explore a new solution based on the plan."
        elif t_type == 'merge':
            task_instruction = "Please merge the selected strategies into a new solution."
        elif t_type == 'review':
            task_instruction = "Please review the solution and provide feedback."
        elif t_type == 'debug':
            task_instruction = "Please debug the current solution."
        else:
            task_instruction = f"Execute task of type {t_type}"
        
        # Prepend competition description if available
        if self.competition_description:
            return f"# Competition Background\n{self.competition_description}\n\n---\n\n# Your Task\n{task_instruction}"
        return task_instruction

    def _create_nodes_from_result(self, result: Dict[str, Any], task: Dict[str, Any]) -> List[Node]:
        """
        Convert agent execution result into a list of Journal Nodes.
        Extracts multiple versions of solution.py if available in history.
        """
        nodes = []
        raw_session = result.get('raw_session')
        
        # Parent handling
        parent_ids = []
        if task['type'] == 'merge':
             # Merge node parents = All Candidates
             parent_ids = task['payload'].get('candidate_ids', [])
        else:
             parent_id = task.get('payload', {}).get('parent_id')
             if parent_id:
                 parent_ids = [parent_id]
        
        # Extract logs once
        logs = ""
        if raw_session:
            logs = json.dumps([h.get('observation') for h in raw_session.history], ensure_ascii=False)

        # Strategy 1: Parse history for write_file to solution.py (Only for Explore tasks)
        if task['type'] in ['explore', 'merge'] and raw_session:
            history = raw_session.history
            seen_content = set()
            
            # Iterate through history to find all write_file actions for solution.py
            for i, step in enumerate(history):
                action = step.get('action') or step.get('tool')
                tool_input = step.get('input') or step.get('tool_input', {})
                
                if action == 'write_file' and isinstance(tool_input, dict):
                    path = tool_input.get('path', '')
                    content = tool_input.get('content', '')
                    
                    if path.endswith('solution.py') and content:
                        if content not in seen_content:
                            seen_content.add(content)
                            
                            # Look ahead for execution logs
                            execution_log = ""
                            for j in range(i + 1, len(history)):
                                next_step = history[j]
                                next_action = next_step.get('action') or next_step.get('tool')
                                next_input = next_step.get('input') or next_step.get('tool_input', {})
                                
                                # Check for execution commands
                                if next_action in ['run_python', 'python', 'bash', 'cmd_line', 'execute_script', 'terminal']:
                                    # Very loose check: if it runs python, assume it runs the solution
                                    # Or check if 'solution.py' is in the command?
                                    # For now, take the first execution result as related.
                                    execution_log = next_step.get('observation', "")
                                    break
                                
                                # Stop if we hit another write to solution.py
                                if next_action == 'write_file' and isinstance(next_input, dict):
                                    next_path = next_input.get('path', '')
                                    if next_path.endswith('solution.py'):
                                        break

                            # Create a node for this version
                            node = Node(
                                parent_ids=parent_ids,
                                code=content,
                                score=None, 
                                step=self.current_epoch,
                                action_type=task['type'],
                                logs=execution_log if execution_log else logs, # Prefer specific execution log, fallback to full session
                                metadata={
                                    "agent_name": result.get('agent_name'),
                                    "task_id": task['id'],
                                    "success": result.get('agent_success'),
                                    "version": "history_snapshot"
                                }
                            )
                            nodes.append(node)
                            
        # Strategy 2: Fallback to final agent output if no nodes found (or for Merge tasks)
        if not nodes:
            agent_output = result.get('agent_output', {})
            code_content = ""
            if isinstance(agent_output, dict):
                code_content = agent_output.get('code', "")
            
            # If code is present, create a node
            if code_content:
                node = Node(
                    parent_ids=parent_ids,
                    code=code_content,
                    score=None if not isinstance(agent_output, dict) else agent_output.get('score'),
                    step=self.current_epoch,
                    action_type=task['type'],
                    logs=logs,
                    metadata={
                        "agent_name": result.get('agent_name'),
                        "task_id": task['id'],
                        "success": result.get('agent_success'),
                        "version": "final_output"
                    }
                )
                nodes.append(node)
                
        return nodes
