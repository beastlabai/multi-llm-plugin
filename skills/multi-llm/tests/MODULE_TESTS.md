<!-- GENERATED FILE - DO NOT EDIT.
     Regenerate with `make tests-map` (from the repository root), or
     `python skills/multi-llm/tests/tools/tests_for.py --write`.
     tests/test_module_tests_map.py fails when this file is stale, i.e. when a
     test file's imports or a declared route change the mapping. -->

# Module to tests map

Which `tests/test_*.py` files cover each source module, from a static AST scan:
`import x`, `from x import y`, `from utils import foo`, a literal
`importlib.import_module("x")` / `__import__("x")`, plus two declared tables in
`tests/tools/tests_for.py` (listed at the end). A `tests/support` module a test
imports counts its imports as the test's. **Advisory:** the scan still misses
coverage through `runpy`, through any other helper module that re-exports a
source module, through a suite-wide fixture a test names without importing, and
through another source module that imports the one you changed; it never
replaces the CI command. For a ranked list that also weighs
file size, and to run just those files, use `make test-for MODULE=utils/foo.py`.

Files are ordered by name match, then a `from x import names` import, then name;
at most 12 are listed per module. A **hub** (more than 40 files)
cannot be narrowed by this map: choose by the file names or run the CI command.

| Module | Test files | Top files |
|--------|-----------:|-----------|
| `apply_code_fixes_orchestrator.py` | 24 | `test_e2e_apply_code_fixes.py`, `test_e2e_batching.py`, `test_e2e_commit_modes.py`, `test_e2e_improve_modes.py`, `test_apply_code_fixes_unit.py`, `test_apply_orchestrator_base.py`, `test_clipped_previews.py`, `test_human_decision_overlay.py`, `test_report_markdown.py`, `test_scope_report_tags.py`, `test_validation_context_by_path.py`, `test_validation_subagent_orchestrators.py`, +12 more |
| `apply_commit_fixes_orchestrator.py` | 2 | `test_e2e_commit_modes.py`, `test_instruction_references.py` |
| `apply_suggestions_orchestrator.py` | 37 | `test_e2e_full_chain.py`, `test_e2e_html_report.py`, `test_e2e_improve_modes.py`, `test_e2e_validation.py`, `test_e2e_validation_edge_cases.py`, `test_e2e_workflow.py`, `test_apply_orchestrator_base.py`, `test_apply_suggestions_integration.py`, `test_clipped_previews.py`, `test_instruction_portability.py`, `test_validation_subagent_orchestrators.py`, `test_write_executor_orchestrators.py`, +25 more |
| `apply_task_suggestions_orchestrator.py` | 9 | `test_e2e_improve_modes.py`, `test_apply_orchestrator_base.py`, `test_apply_task_suggestions.py`, `test_relay_says_it_once.py`, `test_tasks_graph_check.py`, `test_write_executor_orchestrators.py`, `test_apply_json_output.py`, `test_scope_policy_apply.py`, `test_scope_policy_cli.py` |
| `ask_orchestrator.py` | 19 | `test_ask.py`, `test_ask_run.py`, `test_damaged_checkpoint_recovery.py`, `test_omp_stream_reading.py`, `test_pi_json_mode.py`, `test_provider_housekeeping.py`, `test_slot_guard_parity.py`, `test_status_file.py`, `test_ask_answer_persistence.py`, `test_ask_raised_call.py`, `test_ask_withheld_answers.py`, `test_checked_bytes_consumed.py`, +7 more |
| `check_workflow_prerequisites.py` | 24 | `test_empty_review_outcomes.py`, `test_review_tasks_integration.py`, `test_task_suggestion_counts.py`, `test_apply_orchestrator_base.py`, `test_apply_phase_in_progress.py`, `test_batch_done.py`, `test_check_task_findings.py`, `test_completion_predates.py`, `test_detach.py`, `test_full_workflow_orchestrator.py`, `test_improve_orchestrator.py`, `test_instruction_portability.py`, +12 more |
| `code_review_orchestrator.py` | 55 | **hub**, cannot narrow: `test_code_review_orchestrator.py`, `test_e2e_code_review.py`, `test_e2e_commit_modes.py`, `test_e2e_full_chain.py`, `test_e2e_html_report.py`, `test_e2e_improve_modes.py`, `test_e2e_validation.py`, `test_apply_code_fixes_unit.py`, `test_code_review_integration.py`, `test_duplicate_run_prevention.py`, `test_format_issue.py`, `test_html_report_integration.py`, +43 more |
| `display_decisions.py` | 1 | `test_judge_vocab.py` |
| `finalize_tracking.py` | 5 | `test_commit_review_paths.py`, `test_no_commit_repository.py`, `test_tracking_write_integrity.py`, `test_write_executor.py`, `test_write_executor_orchestrators.py` |
| `full_workflow_orchestrator.py` | 13 | `test_full_workflow_orchestrator.py`, `test_e2e_improve_modes.py`, `test_commit_spelling_sites.py`, `test_consolidation_not_recommended_twice.py`, `test_empty_review_outcomes.py`, `test_improve_commits.py`, `test_improve_orchestrator.py`, `test_improve_steps.py`, `test_mode_sequence.py`, `test_partial_detached_result.py`, `test_scope_policy_cli.py`, `test_scope_policy_composites.py`, +1 more |
| `generate_tasks_orchestrator.py` | 13 | `test_generate_tasks_orchestrator.py`, `test_e2e_improve_modes.py`, `test_e2e_validation.py`, `test_continuation_plan_spelling.py`, `test_dispatch_from_a_subdirectory.py`, `test_full_workflow_orchestrator.py`, `test_native_prompt_files.py`, `test_next_phase.py`, `test_plan_dir_writers_held.py`, `test_reply_without_schema_version.py`, `test_scope_policy_cli.py`, `test_scope_policy_composites.py`, +1 more |
| `hosts/render_capabilities.py` | 0 | none (no direct importer) |
| `implement_orchestrator.py` | 108 | **hub**, cannot narrow: `test_implement_orchestrator.py`, `test_e2e_code_review.py`, `test_e2e_full_chain.py`, `test_e2e_validation.py`, `test_e2e_workflow.py`, `test_implement_decisions.py`, `test_notice_clause_join.py`, `test_write_executor.py`, `test_accepted_check_rerun.py`, `test_accepted_finding_note.py`, `test_accepted_finding_restate.py`, `test_accepted_findings_scope.py`, +96 more |
| `improve_code_orchestrator.py` | 6 | `test_e2e_failures.py`, `test_e2e_improve_modes.py`, `test_improve_commits.py`, `test_improve_orchestrator.py`, `test_improve_steps.py`, `test_scope_policy_composites.py` |
| `improve_commits_orchestrator.py` | 2 | `test_e2e_commit_modes.py`, `test_e2e_improve_modes.py` |
| `improve_plan_orchestrator.py` | 5 | `test_e2e_improve_modes.py`, `test_improve_commits.py`, `test_improve_orchestrator.py`, `test_improve_steps.py`, `test_scope_policy_composites.py` |
| `init_config.py` | 7 | `test_init_config.py`, `test_availability.py`, `test_install_host.py`, `test_provider_housekeeping.py`, `test_provider_registry.py`, `test_subagent_executor.py`, `test_workflow_prerequisites.py` |
| `modes.py` | 51 | **hub**, cannot narrow: `test_block_reason_says_one_thing.py`, `test_claude_package.py`, `test_cli_surface_emitter.py`, `test_codex_long_call_rule.py`, `test_codex_package.py`, `test_commit_dispatch.py`, `test_consolidation_not_recommended_twice.py`, `test_core_file.py`, `test_core_pages.py`, `test_detach.py`, `test_dispatch_from_a_subdirectory.py`, `test_doctor.py`, +39 more |
| `multi_llm.py` | 109 | **hub**, cannot narrow: `test_check_reply.py`, `test_tasks_graph_check.py`, `test_abort_signal_handlers.py`, `test_accepted_check_rerun.py`, `test_accepted_finding_note.py`, `test_accepted_finding_restate.py`, `test_accepted_findings_scope.py`, `test_already_completed_rerun.py`, `test_an_answer_answers_one_stop.py`, `test_apply_json_output.py`, `test_apply_phases_close_like_implement.py`, `test_batch_check_on_record.py`, +97 more |
| `review_commits_orchestrator.py` | 1 | `test_e2e_commit_modes.py` |
| `review_plan_orchestrator.py` | 27 | `test_review_plan_orchestrator.py`, `test_e2e_code_review.py`, `test_e2e_exec_subagents.py`, `test_e2e_failures.py`, `test_e2e_full_chain.py`, `test_e2e_html_report.py`, `test_e2e_improve_modes.py`, `test_e2e_validation.py`, `test_e2e_validation_edge_cases.py`, `test_e2e_workflow.py`, `test_duplicate_run_prevention.py`, `test_bypassed_slot_failure_provenance.py`, +15 more |
| `review_tasks_orchestrator.py` | 9 | `test_review_tasks_orchestrator.py`, `test_e2e_failures.py`, `test_e2e_improve_modes.py`, `test_review_tasks_integration.py`, `test_checked_bytes_consumed.py`, `test_damaged_checkpoint_recovery.py`, `test_scope_policy_cli.py`, `test_scope_prompts_and_validation.py`, `test_validation_subagent_orchestrators.py` |
| `update_plan_tasks.py` | 6 | `test_update_plan_tasks.py`, `test_executor_role_sweep.py`, `test_task_decomposer.py`, `test_tasks_graph_check.py`, `test_text_regex_linearity.py`, `test_generate_tasks_orchestrator.py` |
| `utils/__init__.py` | 338 | **hub**, cannot narrow: `test_utils_imports.py`, `test_git_utils.py`, `test_git_utils_moved_names.py`, `test_abort_signal_handlers.py`, `test_accepted_findings_scope.py`, `test_advisory.py`, `test_aider_provider.py`, `test_an_answer_answers_one_stop.py`, `test_answer_text.py`, `test_applied_without_changes.py`, `test_apply_fix_known_checks.py`, `test_apply_json_output.py`, +326 more |
| `utils/advisory.py` | 2 | `test_advisory.py`, `test_env_vars.py` |
| `utils/apply_orchestrator_base.py` | 15 | `test_apply_orchestrator_base.py`, `test_apply_json_output.py`, `test_batch_snapshot.py`, `test_human_decisions.py`, `test_prior_changes.py`, `test_subagent_executor.py`, `test_unattended.py`, `test_validation_subagent_orchestrators.py`, `test_write_executor.py`, `test_batch_own_check.py`, `test_completion_predates.py`, `test_host_index.py`, +3 more |
| `utils/apply_output_helpers.py` | 2 | `test_apply_orchestrator_base.py`, `test_host_index.py` |
| `utils/apply_path_helpers.py` | 3 | `test_apply_code_fixes_unit.py`, `test_apply_orchestrator_base.py`, `test_apply_task_suggestions.py` |
| `utils/apply_report.py` | 2 | `test_apply_report.py`, `test_relay_says_it_once.py` |
| `utils/apply_selection_helpers.py` | 4 | `test_apply_code_fixes_unit.py`, `test_apply_orchestrator_base.py`, `test_apply_task_suggestions.py`, `test_scope_policy_apply.py` |
| `utils/apply_suggestion_phase.py` | 1 | `test_apply_orchestrator_base.py` |
| `utils/atomic_io.py` | 5 | `test_atomic_io.py`, `test_ask_answer_persistence.py`, `test_batched_validation.py`, `test_held_folder.py`, `test_scheduler_record_writers.py` |
| `utils/attempt_classifier.py` | 5 | `test_provider_kilocode.py`, `test_attempt_backpressure.py`, `test_opencode_loop_watch.py`, `test_probe_output_redaction.py`, `test_subagent_executor_moved_names.py` |
| `utils/backup.py` | 1 | `test_backup.py` |
| `utils/batch_done.py` | 23 | `test_batch_done.py`, `test_accepted_finding_restate.py`, `test_commit_apply.py`, `test_selected_tasks_and_check_trees.py`, `test_apply_fix_known_checks.py`, `test_batch_snapshot.py`, `test_check_always_reported.py`, `test_clipped_previews.py`, `test_declined_reason_punctuation.py`, `test_empty_review_outcomes.py`, `test_implement_continues_a_started_run.py`, `test_native_apply_count_over_the_run.py`, +11 more |
| `utils/batch_ledger.py` | 2 | `test_batch_ledger.py`, `test_write_legs_parity.py` |
| `utils/batch_prompt.py` | 2 | `test_scheduler_null_state.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/batch_prompt_builder.py` | 2 | `test_scheduler_null_state.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/batch_reply.py` | 1 | `test_subagent_executor_moved_names.py` |
| `utils/check_records.py` | 2 | `test_scheduler_null_state.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/check_reply.py` | 3 | `test_check_reply.py`, `test_scope_prompts_and_validation.py`, `test_validation_reply_printed_once.py` |
| `utils/check_tree.py` | 0 | none (no direct importer) |
| `utils/cli_common.py` | 91 | **hub**, cannot narrow: `test_batch_own_check.py`, `test_check_findings_hold_dependents.py`, `test_clipped_previews.py`, `test_commit_dispatch.py`, `test_commit_front_door.py`, `test_commit_spelling.py`, `test_commit_spelling_sites.py`, `test_consolidation_not_recommended_twice.py`, `test_continuation_plan_spelling.py`, `test_core_file.py`, `test_core_pages.py`, `test_dispatch_from_a_subdirectory.py`, +79 more |
| `utils/cli_version_cache.py` | 2 | `test_cli_version_cache.py`, `test_doctor.py` |
| `utils/clock.py` | 1 | `test_clock.py` |
| `utils/code_fix_batcher.py` | 11 | `test_code_fix_batcher.py`, `test_apply_code_fixes_unit.py`, `test_code_fix_prior_batches.py`, `test_commit_apply.py`, `test_commit_write_policy.py`, `test_decline_reason_quotes_the_decision.py`, `test_e2e_batching.py`, `test_executor_role_sweep.py`, `test_prior_batches_named_by_id.py`, `test_prior_changes_not_quoted_twice.py`, `test_write_executor_orchestrators.py` |
| `utils/codex_trust.py` | 1 | `test_codex_trust.py` |
| `utils/commit_apply.py` | 10 | `test_commit_apply.py`, `test_e2e_commit_modes.py`, `test_result_contract.py`, `test_state_file.py`, `test_clock.py`, `test_commit_front_door.py`, `test_commit_write_policy.py`, `test_improve_commits.py`, `test_scope_policy_commit_doors.py`, `test_text_fmt.py` |
| `utils/commit_attribution.py` | 1 | `test_commit_attribution.py` |
| `utils/commit_brief.py` | 8 | `test_commit_brief.py`, `test_e2e_commit_modes.py`, `test_commit_apply.py`, `test_git_two_commit.py`, `test_commit_front_door.py`, `test_improve_commits.py`, `test_scope_policy_commit_doors.py`, `test_text_fmt.py` |
| `utils/commit_front_door.py` | 3 | `test_commit_front_door.py`, `test_clock.py`, `test_review_round.py` |
| `utils/commit_paths.py` | 1 | `test_commit_review_paths.py` |
| `utils/commit_range.py` | 16 | `test_commit_range.py`, `test_codex_stream_audit.py`, `test_commit_apply.py`, `test_commit_attribution.py`, `test_commit_brief.py`, `test_commit_review_prompt.py`, `test_commit_review_report.py`, `test_commit_write_policy.py`, `test_e2e_commit_modes.py`, `test_git_two_commit.py`, `test_result_contract.py`, `test_review_code_head_ref.py`, +4 more |
| `utils/commit_report.py` | 5 | `test_commit_review_prompt.py`, `test_commit_review_report.py`, `test_report_templates_js.py`, `test_scope_report_tags.py`, `test_text_fmt.py` |
| `utils/commit_review.py` | 9 | `test_e2e_improve_modes.py`, `test_commit_brief.py`, `test_commit_front_door.py`, `test_commit_spelling_sites.py`, `test_e2e_commit_modes.py`, `test_improve_commits.py`, `test_cli_surface_emitter.py`, `test_instruction_portability.py`, `test_scope_policy_commit_doors.py` |
| `utils/commit_spelling.py` | 4 | `test_commit_spelling.py`, `test_commit_front_door.py`, `test_improve_commits.py`, `test_instruction_portability.py` |
| `utils/commit_write_guard.py` | 1 | `test_write_batch_scheduler_moved_names.py` |
| `utils/concurrency.py` | 0 | none (no direct importer) |
| `utils/consolidation.py` | 16 | `test_consolidation.py`, `test_consolidation_integration.py`, `test_consolidation_not_recommended_twice.py`, `test_consolidation_group_shape.py`, `test_full_workflow_orchestrator.py`, `test_host_index.py`, `test_html_report_scripts.py`, `test_html_report_view.py`, `test_replay_flags.py`, `test_report_markdown.py`, `test_report_templates_js.py`, `test_scope_prompts_and_validation.py`, +4 more |
| `utils/detach.py` | 56 | **hub**, cannot narrow: `test_detach.py`, `test_partial_detached_result.py`, `test_ask.py`, `test_abort_signal_handlers.py`, `test_already_completed_rerun.py`, `test_apply_json_output.py`, `test_atomic_io_callers.py`, `test_batch_done.py`, `test_codex_mcp_server.py`, `test_codex_trust.py`, `test_commit_front_door.py`, `test_continuation_plan_spelling.py`, +44 more |
| `utils/dispatch_spawn.py` | 24 | `test_applied_without_changes.py`, `test_attempt_backpressure.py`, `test_dispatch_executor_hooks.py`, `test_dispatch_steps.py`, `test_doctor.py`, `test_failure_classifier.py`, `test_fx_provider.py`, `test_generate_tasks_orchestrator.py`, `test_jail_and_audit_helpers.py`, `test_lane_a_ladder.py`, `test_lane_b_structured_output.py`, `test_opencode_config_providers.py`, +12 more |
| `utils/doctor.py` | 6 | `test_doctor.py`, `test_cli_version_cache.py`, `test_failure_classifier.py`, `test_plugin_landlock_surfaces.py`, `test_provider_omp.py`, `test_result_contract.py` |
| `utils/env_vars.py` | 2 | `test_env_vars.py`, `test_codex_package.py` |
| `utils/failure_capture.py` | 7 | `test_scheduler_record_writers.py`, `test_dispatch_steps.py`, `test_held_folder.py`, `test_plan_dir_writers_held.py`, `test_probe_output_redaction.py`, `test_reaggregate_identity.py`, `test_subagent_executor_moved_names.py` |
| `utils/file_discovery.py` | 1 | `test_file_discovery.py` |
| `utils/filtering.py` | 6 | `test_filtering.py`, `test_apply_code_fixes_unit.py`, `test_apply_task_suggestions.py`, `test_importance_lanes.py`, `test_judge_vocab.py`, `test_unattended.py` |
| `utils/git_diff.py` | 1 | `test_git_utils_moved_names.py` |
| `utils/git_utils.py` | 19 | `test_git_utils.py`, `test_git_utils_moved_names.py`, `test_code_review_integration.py`, `test_code_review_orchestrator.py`, `test_commit_brief.py`, `test_commit_range.py`, `test_finalize_tracking.py`, `test_html_report_functional.py`, `test_implement_orchestrator.py`, `test_no_commit_repository.py`, `test_provider_registry.py`, `test_review_code_head_ref.py`, +7 more |
| `utils/held_folder.py` | 2 | `test_held_folder.py`, `test_damaged_checkpoint_recovery.py` |
| `utils/host_capabilities.py` | 20 | `test_host_capabilities.py`, `test_ask.py`, `test_cli_surface_emitter.py`, `test_codex_mcp_server.py`, `test_codex_package.py`, `test_codex_route.py`, `test_credential_strip.py`, `test_detach.py`, `test_improve_steps.py`, `test_install_host.py`, `test_instruction_portability.py`, `test_instruction_references.py`, +8 more |
| `utils/host_install.py` | 11 | `test_codex_mcp_server.py`, `test_availability.py`, `test_cli_surface_emitter.py`, `test_codex_package.py`, `test_codex_route.py`, `test_core_file.py`, `test_core_pages.py`, `test_cursor_auth_probe.py`, `test_install_host.py`, `test_opencode_package.py`, `test_shim_correctness.py` |
| `utils/host_paths.py` | 0 | none (no direct importer) |
| `utils/host_routes/__init__.py` | 1 | `test_codex_route.py` |
| `utils/host_routes/codex.py` | 1 | `test_codex_route.py` |
| `utils/host_routes/npm.py` | 0 | none (no direct importer) |
| `utils/host_shim.py` | 1 | `test_install_host.py` |
| `utils/html_report_generator.py` | 23 | `test_html_report_generator.py`, `test_e2e_html_report.py`, `test_commit_review_report.py`, `test_consolidation_group_shape.py`, `test_consolidation_integration.py`, `test_html_report_functional.py`, `test_html_report_integration.py`, `test_html_report_scripts.py`, `test_html_report_view.py`, `test_human_decision_overlay.py`, `test_human_decisions.py`, `test_importance_lanes.py`, +11 more |
| `utils/human_decisions.py` | 5 | `test_human_decisions.py`, `test_judge_vocab.py`, `test_reply_without_schema_version.py`, `test_subagent_executor.py`, `test_state_save_merge.py` |
| `utils/human_task_plan.py` | 3 | `test_write_legs_parity.py`, `test_scheduler_null_state.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/importance.py` | 2 | `test_importance.py`, `test_importance_lanes.py` |
| `utils/improve_steps.py` | 7 | `test_improve_steps.py`, `test_e2e_improve_modes.py`, `test_dispatch_from_a_subdirectory.py`, `test_improve_commits.py`, `test_report_check_composite.py`, `test_scope_policy_cli.py`, `test_scope_policy_composites.py` |
| `utils/improve_workflow.py` | 13 | `test_commit_front_door.py`, `test_commit_spelling_sites.py`, `test_completion_predates.py`, `test_e2e_failures.py`, `test_e2e_improve_modes.py`, `test_improve_commits.py`, `test_improve_orchestrator.py`, `test_improve_steps.py`, `test_next_phase.py`, `test_scope_policy_apply.py`, `test_scope_policy_cli.py`, `test_scope_policy_commit_doors.py`, +1 more |
| `utils/interactive.py` | 4 | `test_interactive.py`, `test_provider_registry.py`, `test_availability.py`, `test_init_config.py` |
| `utils/json_extractor.py` | 28 | `test_json_extractor.py`, `test_ask.py`, `test_ask_answer_persistence.py`, `test_ask_raised_call.py`, `test_ask_run.py`, `test_ask_withheld_answers.py`, `test_duplicate_run_prevention.py`, `test_e2e_failures.py`, `test_envelope_locator.py`, `test_exit_zero_prose_matrix.py`, `test_llm_client.py`, `test_llm_robustness.py`, +16 more |
| `utils/judge_vocab.py` | 1 | `test_judge_vocab.py` |
| `utils/known_checks.py` | 6 | `test_apply_fix_known_checks.py`, `test_clipped_previews.py`, `test_scheduler_null_state.py`, `test_state_save_merge.py`, `test_subagent_executor_moved_names.py`, `test_tests_line_leads_with_what_ran.py` |
| `utils/landlock_dispatch.py` | 19 | `test_landlock_dispatch.py`, `test_sandbox_rung_reporting.py`, `test_atomic_io_callers.py`, `test_codex_jail_rig.py`, `test_doctor.py`, `test_jail_and_audit_helpers.py`, `test_landlock_repo_wide.py`, `test_plugin_landlock_acceptance.py`, `test_plugin_landlock_executor.py`, `test_plugin_landlock_fanout.py`, `test_plugin_landlock_surfaces.py`, `test_plugin_landlock_write.py`, +7 more |
| `utils/landlock_exec.py` | 5 | `test_landlock_exec.py`, `test_codex_jail_rig.py`, `test_landlock_dispatch.py`, `test_landlock_repo_wide.py`, `test_plugin_landlock_acceptance.py` |
| `utils/launch_limiter.py` | 2 | `test_launch_limiter.py`, `test_subagent_executor.py` |
| `utils/llm_client.py` | 57 | **hub**, cannot narrow: `test_llm_client.py`, `test_llm_client_moved_names.py`, `test_aider_provider.py`, `test_answer_text.py`, `test_availability.py`, `test_bypassed_slot_failure_provenance.py`, `test_codex_jail_rig.py`, `test_codex_stream_audit.py`, `test_codex_trust.py`, `test_commit_review_prompt.py`, `test_e2e_failures.py`, `test_env_home_expansion.py`, +45 more |
| `utils/manual_tasks.py` | 0 | none (no direct importer) |
| `utils/metrics.py` | 15 | `test_metrics.py`, `test_commit_front_door.py`, `test_commit_write_policy.py`, `test_executor_role_sweep.py`, `test_apply_phases_close_like_implement.py`, `test_batch_done.py`, `test_e2e_validation.py`, `test_eta_closes_every_ending.py`, `test_eta_names_what_did_not_run.py`, `test_eta_says_what_happened.py`, `test_judge_vocab.py`, `test_plan_dir_writers_held.py`, +3 more |
| `utils/next_phase.py` | 35 | `test_next_phase.py`, `test_continuation_plan_spelling.py`, `test_next_command_emitters.py`, `test_result_contract.py`, `test_already_completed_rerun.py`, `test_batch_done.py`, `test_check_task_findings.py`, `test_cli_surface_emitter.py`, `test_codex_mcp_server.py`, `test_commit_spelling_sites.py`, `test_implement_continues_a_started_run.py`, `test_improve_steps.py`, +23 more |
| `utils/next_phase_words.py` | 0 | none (no direct importer) |
| `utils/output_handler.py` | 148 | **hub**, cannot narrow: `test_output_handler.py`, `test_all_failed_fanout_is_recorded.py`, `test_applied_without_changes.py`, `test_apply_code_fixes_unit.py`, `test_apply_orchestrator_base.py`, `test_apply_task_suggestions.py`, `test_apply_task_suggestions_integration.py`, `test_ask.py`, `test_ask_raised_call.py`, `test_check_task_findings.py`, `test_commit_apply.py`, `test_commit_review_paths.py`, +136 more |
| `utils/phase_walk.py` | 1 | `test_phase_walk.py` |
| `utils/plan_updater.py` | 4 | `test_plan_updater.py`, `test_commit_brief.py`, `test_tasks_file.py`, `test_text_regex_linearity.py` |
| `utils/positionals.py` | 0 | none (no direct importer) |
| `utils/prior_changes.py` | 16 | `test_prior_changes.py`, `test_prior_changes_empty_id.py`, `test_prior_changes_not_quoted_twice.py`, `test_code_fix_batcher.py`, `test_decline_reason_quotes_the_decision.py`, `test_applied_without_changes.py`, `test_batch_done.py`, `test_check_findings_hold_dependents.py`, `test_clock.py`, `test_code_fix_prior_batches.py`, `test_implement_continues_a_started_run.py`, `test_native_apply_count_over_the_run.py`, +4 more |
| `utils/probe_state.py` | 4 | `test_jail_and_audit_helpers.py`, `test_plugin_landlock_write.py`, `test_subagent_executor_moved_names.py`, `test_write_ceiling_cache.py` |
| `utils/process_capture.py` | 2 | `test_process_capture.py`, `test_opencode_loop_watch.py` |
| `utils/process_control.py` | 37 | `test_git_utils.py`, `test_abort_signal_handlers.py`, `test_ask.py`, `test_batch_snapshot.py`, `test_codex_jail_rig.py`, `test_codex_trust.py`, `test_commit_front_door.py`, `test_continuation_plan_spelling.py`, `test_damaged_checkpoint_recovery.py`, `test_detach.py`, `test_eta_closes_every_ending.py`, `test_full_workflow_orchestrator.py`, +25 more |
| `utils/process_state.py` | 1 | `test_process_state.py` |
| `utils/progress_tracker.py` | 17 | `test_progress_tracker.py`, `test_ask.py`, `test_ask_answer_persistence.py`, `test_ask_run.py`, `test_codex_stream_audit.py`, `test_reaggregate_progress_line.py`, `test_review_lane_parity.py`, `test_ask_raised_call.py`, `test_code_review_ladder.py`, `test_code_review_orchestrator.py`, `test_detach.py`, `test_fanout_raised_and_cancelled_slots.py`, +5 more |
| `utils/prompt_channels.py` | 10 | `test_commit_review_prompt.py`, `test_fake_lane_output_channel.py`, `test_prompt_loader.py`, `test_provider_contract.py`, `test_review_code_head_ref.py`, `test_review_plan_orchestrator.py`, `test_review_prompt_rules.py`, `test_scope_prompts_and_validation.py`, `test_code_review_orchestrator.py`, `test_llm_client_moved_names.py` |
| `utils/prompt_files.py` | 3 | `test_batch_snapshot.py`, `test_core_file.py`, `test_windows_tree_jobs.py` |
| `utils/prompt_loader.py` | 9 | `test_prompt_loader.py`, `test_ask.py`, `test_ask_run.py`, `test_commit_review_prompt.py`, `test_implement_orchestrator.py`, `test_llm_client.py`, `test_provider_housekeeping.py`, `test_scope_prompts_and_validation.py`, `test_slot_guard_parity.py` |
| `utils/provider_outcome.py` | 10 | `test_provider_outcome.py`, `test_envelope_locator.py`, `test_exit_zero_prose_matrix.py`, `test_failure_classifier.py`, `test_lane_a_ladder.py`, `test_opencode_loop_watch.py`, `test_provider_fixture_rows.py`, `test_provider_kilocode.py`, `test_vendor_text_drops_the_answer.py`, `test_llm_client.py` |
| `utils/provider_registry.py` | 188 | **hub**, cannot narrow: `test_provider_registry.py`, `test_answer_text.py`, `test_ask.py`, `test_availability.py`, `test_commit_review_paths.py`, `test_container_argv.py`, `test_container_live_provider.py`, `test_credential_strip.py`, `test_cursor_auth_probe.py`, `test_detach.py`, `test_doctor.py`, `test_e2e_exec_subagents.py`, +176 more |
| `utils/providers/__init__.py` | 53 | **hub**, cannot narrow: `test_opencode_config_providers.py`, `test_ask.py`, `test_codex_stream_audit.py`, `test_commit_review_prompt.py`, `test_copilot_home_isolation.py`, `test_credential_strip.py`, `test_cursor_argv_and_residue.py`, `test_cursor_profile_isolation.py`, `test_doctor.py`, `test_env_allowlist.py`, `test_failure_classifier.py`, `test_fx_provider.py`, +41 more |
| `utils/providers/agy.py` | 4 | `test_provider_agy.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py` |
| `utils/providers/aider.py` | 10 | `test_aider_provider.py`, `test_provider_aider.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_ask.py`, `test_fanout_workspace_caches.py`, `test_provider_contract.py`, `test_provider_housekeeping.py`, `test_provider_vendor_text.py`, `test_salvage_text.py` |
| `utils/providers/base.py` | 100 | **hub**, cannot narrow: `test_provider_base_moved_names.py`, `test_pi_family_stream.py`, `test_retry_ladder.py`, `test_aider_provider.py`, `test_ask.py`, `test_ask_raised_call.py`, `test_availability.py`, `test_bypassed_slot_failure_provenance.py`, `test_code_review_ladder.py`, `test_code_review_orchestrator.py`, `test_codex_jail_rig.py`, `test_codex_mcp_server.py`, +88 more |
| `utils/providers/child_env.py` | 7 | `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_credential_strip.py`, `test_env_allowlist.py`, `test_provider_base_moved_names.py`, `test_provider_omp.py`, `test_version_gate.py` |
| `utils/providers/claude_code.py` | 14 | `test_provider_claude_code.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_credential_strip.py`, `test_failed_slot_reporting.py`, `test_lane_b_structured_output.py`, `test_provider_contract.py`, `test_provider_housekeeping.py`, `test_result_envelope.py`, `test_served_model_reporting.py`, `test_stderr_diagnosis.py`, `test_structured_output.py`, +2 more |
| `utils/providers/cline.py` | 7 | `test_cline_reducer.py`, `test_provider_cline.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py`, `test_salvage_text.py`, `test_stderr_diagnosis.py` |
| `utils/providers/codex.py` | 36 | `test_codex_jail_rig.py`, `test_codex_stream_audit.py`, `test_codex_trust.py`, `test_provider_codex.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_ask_raised_call.py`, `test_bypassed_slot_failure_provenance.py`, `test_code_review_ladder.py`, `test_code_review_orchestrator.py`, `test_fanout_raised_and_cancelled_slots.py`, `test_fanout_workspace_caches.py`, +24 more |
| `utils/providers/codex_stream_audit.py` | 4 | `test_codex_stream_audit.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_codex.py` |
| `utils/providers/copilot.py` | 7 | `test_copilot_home_isolation.py`, `test_provider_copilot.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py`, `test_salvage_text.py`, `test_private_home.py` |
| `utils/providers/cursor_agent.py` | 15 | `test_provider_cursor_agent.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_ask.py`, `test_cursor_argv_and_residue.py`, `test_cursor_auth_probe.py`, `test_cursor_profile_isolation.py`, `test_launch_limiter.py`, `test_llm_client.py`, `test_probe_output_redaction.py`, `test_provider_contract.py`, `test_result_envelope.py`, +3 more |
| `utils/providers/failure_classifier.py` | 7 | `test_failure_classifier.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_lane_a_ladder.py`, `test_provider_base_moved_names.py`, `test_provider_fixture_rows.py`, `test_provider_vendor_text.py` |
| `utils/providers/fx.py` | 6 | `test_fx_provider.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_dockerfile_pins.py`, `test_provider_contract.py`, `test_wrapper_env_contract.py` |
| `utils/providers/gemini.py` | 5 | `test_provider_gemini.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py`, `test_stderr_diagnosis.py` |
| `utils/providers/goose.py` | 8 | `test_goose_error_kind.py`, `test_provider_goose.py`, `test_reasoning_effort_ladder.py`, `test_provider_contract.py`, `test_provider_opencode.py`, `test_provider_vendor_text.py`, `test_retry_ladder.py`, `test_salvage_text.py` |
| `utils/providers/grok.py` | 6 | `test_grok_dispatch_env.py`, `test_provider_grok.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py`, `test_salvage_text.py` |
| `utils/providers/kilocode.py` | 8 | `test_provider_kilocode.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_ask.py`, `test_live_lane_variant_echo.py`, `test_llm_client.py`, `test_provider_contract.py`, `test_stderr_diagnosis.py` |
| `utils/providers/kimi.py` | 7 | `test_kimi_home_isolation.py`, `test_provider_kimi.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py`, `test_salvage_text.py`, `test_private_home.py` |
| `utils/providers/muse.py` | 8 | `test_muse_terminal_state.py`, `test_provider_muse.py`, `test_reasoning_effort_ladder.py`, `test_lane_a_ladder.py`, `test_provider_contract.py`, `test_retry_ladder.py`, `test_salvage_text.py`, `test_stderr_diagnosis.py` |
| `utils/providers/omp.py` | 13 | `test_omp_containment.py`, `test_omp_stream_reading.py`, `test_provider_omp.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_failure_classifier.py`, `test_llm_client.py`, `test_pi_family_stream.py`, `test_pi_json_mode.py`, `test_provider_contract.py`, `test_provider_housekeeping.py`, `test_provider_timeouts.py`, +1 more |
| `utils/providers/opencode.py` | 23 | `test_opencode_loop_watch.py`, `test_opencode_package.py`, `test_opencode_version_pins.py`, `test_opencode_config_providers.py`, `test_opencode_hermetic.py`, `test_opencode_permission_block.py`, `test_opencode_write_profile.py`, `test_provider_opencode.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_container_host_lane.py`, `test_json_extractor.py`, +11 more |
| `utils/providers/parse_kit.py` | 3 | `test_parse_kit.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py` |
| `utils/providers/pi.py` | 13 | `test_pi_family_stream.py`, `test_pi_json_mode.py`, `test_provider_housekeeping.py`, `test_provider_pi.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_ask.py`, `test_container_provider_lane.py`, `test_e2e_failures.py`, `test_fx_provider.py`, `test_llm_client.py`, `test_provider_contract.py`, +1 more |
| `utils/providers/pi_family_stream.py` | 3 | `test_pi_family_stream.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py` |
| `utils/providers/private_home.py` | 4 | `test_private_home.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_qwen_home_isolation.py` |
| `utils/providers/qwen_code.py` | 7 | `test_provider_qwen_code.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py`, `test_qwen_home_isolation.py`, `test_salvage_text.py`, `test_private_home.py` |
| `utils/providers/reasoning.py` | 3 | `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_base_moved_names.py` |
| `utils/providers/result_envelope.py` | 10 | `test_result_envelope.py`, `test_result_envelope_verdicts.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_envelope_locator.py`, `test_failed_slot_reporting.py`, `test_json_extractor.py`, `test_llm_robustness.py`, `test_provider_claude_code.py`, `test_provider_qwen_code.py` |
| `utils/providers/sandbox_rungs.py` | 3 | `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_base_moved_names.py` |
| `utils/providers/stdout_shape.py` | 3 | `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_base_moved_names.py` |
| `utils/providers/types.py` | 3 | `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_base_moved_names.py` |
| `utils/providers/version_gate.py` | 13 | `test_version_gate.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_cursor_argv_and_residue.py`, `test_failure_classifier.py`, `test_opencode_config_providers.py`, `test_opencode_hermetic.py`, `test_provider_base_moved_names.py`, `test_provider_contract.py`, `test_provider_cursor_agent.py`, `test_provider_housekeeping.py`, `test_provider_omp.py`, +1 more |
| `utils/providers/vibe.py` | 7 | `test_vibe_home_isolation.py`, `test_provider_vibe.py`, `test_reasoning_effort_ladder.py`, `test_retry_ladder.py`, `test_provider_contract.py`, `test_salvage_text.py`, `test_private_home.py` |
| `utils/readonly_audit.py` | 3 | `test_ask.py`, `test_code_review_orchestrator.py`, `test_codex_stream_audit.py` |
| `utils/redact.py` | 4 | `test_probe_output_redaction.py`, `test_redaction.py`, `test_redaction_pem_scanner.py`, `test_cursor_auth_probe.py` |
| `utils/report_check.py` | 14 | `test_report_check.py`, `test_report_check_composite.py`, `test_report_check_session.py`, `test_report_check_stop_answers.py`, `test_block_reason_says_one_thing.py`, `test_claude_package.py`, `test_hook_transcript_settle.py`, `test_improve_commits.py`, `test_manual_task_gate.py`, `test_manual_tasks_on_completion.py`, `test_next_step_markdown.py`, `test_resume_why.py`, +2 more |
| `utils/report_markdown.py` | 5 | `test_report_markdown.py`, `test_commit_review_report.py`, `test_consolidation_group_shape.py`, `test_duplicate_run_prevention.py`, `test_review_round.py` |
| `utils/report_parser.py` | 16 | `test_report_parser.py`, `test_apply_code_fixes_unit.py`, `test_apply_orchestrator_base.py`, `test_apply_task_suggestions.py`, `test_code_review_orchestrator.py`, `test_commit_review_report.py`, `test_consolidation_integration.py`, `test_html_report_functional.py`, `test_html_report_generator.py`, `test_html_report_integration.py`, `test_judge_vocab.py`, `test_report_markdown.py`, +4 more |
| `utils/report_view.py` | 5 | `test_html_report_view.py`, `test_html_report_functional.py`, `test_html_report_generator.py`, `test_html_report_integration.py`, `test_scope_report_tags.py` |
| `utils/restore_batch.py` | 4 | `test_external_rollback_storage.py`, `test_plugin_landlock_write.py`, `test_write_executor.py`, `test_write_executor_orchestrators.py` |
| `utils/result_files.py` | 12 | `test_misnamed_result_files.py`, `test_host_index.py`, `test_model_result_shapes.py`, `test_status_file.py`, `test_subagent_executor.py`, `test_ask_answer_persistence.py`, `test_ask_withheld_answers.py`, `test_checked_bytes_consumed.py`, `test_damaged_checkpoint_recovery.py`, `test_model_result_paths.py`, `test_rejected_slot_answers.py`, `test_scope_report_tags.py` |
| `utils/resume_state.py` | 4 | `test_resume_state.py`, `test_damaged_checkpoint_recovery.py`, `test_rejected_slot_answers.py`, `test_review_round.py` |
| `utils/retry_ladder.py` | 27 | `test_retry_ladder.py`, `test_applied_without_changes.py`, `test_dispatch_steps.py`, `test_envelope_locator.py`, `test_eta_says_what_happened.py`, `test_exit_zero_prose_matrix.py`, `test_failure_classifier.py`, `test_grok_dispatch_env.py`, `test_lane_a_ladder.py`, `test_launch_limiter.py`, `test_llm_client.py`, `test_opencode_loop_watch.py`, +15 more |
| `utils/review_diff.py` | 4 | `test_review_diff_spill.py`, `test_code_review_orchestrator.py`, `test_git_two_commit.py`, `test_git_utils_moved_names.py` |
| `utils/review_lane.py` | 5 | `test_review_lane_parity.py`, `test_ask_withheld_answers.py`, `test_checked_bytes_consumed.py`, `test_damaged_checkpoint_recovery.py`, `test_rejected_slot_answers.py` |
| `utils/review_orchestrator_base.py` | 77 | **hub**, cannot narrow: `test_already_completed_rerun.py`, `test_apply_suggestions_integration.py`, `test_ask.py`, `test_consolidation_not_recommended_twice.py`, `test_detach.py`, `test_duplicate_run_prevention.py`, `test_e2e_failures.py`, `test_failure_classifier.py`, `test_fast_backoff.py`, `test_format_suggestion.py`, `test_host_index.py`, `test_json_recovery_sites.py`, +65 more |
| `utils/review_round.py` | 8 | `test_review_round.py`, `test_checked_bytes_consumed.py`, `test_damaged_checkpoint_recovery.py`, `test_duplicate_run_prevention.py`, `test_plan_dir_writers_held.py`, `test_rejected_slot_answers.py`, `test_review_lane_parity.py`, `test_validation_batch_binding.py` |
| `utils/routes.py` | 0 | none (no direct importer) |
| `utils/run_record.py` | 14 | `test_clock.py`, `test_detach.py`, `test_eta_closes_every_ending.py`, `test_full_workflow_orchestrator.py`, `test_generate_tasks_orchestrator.py`, `test_implement_resolution_replay.py`, `test_progress_tracker.py`, `test_review_tasks_orchestrator.py`, `test_signal_identity_posix.py`, `test_status_recorded_route.py`, `test_subagent_executor.py`, `test_unattended.py`, +2 more |
| `utils/run_registry.py` | 6 | `test_atomic_io_callers.py`, `test_damaged_checkpoint_recovery.py`, `test_detach.py`, `test_review_round.py`, `test_signal_identity_posix.py`, `test_windows_run_jobs.py` |
| `utils/run_status.py` | 64 | **hub**, cannot narrow: `test_run_status_moved_names.py`, `test_misnamed_result_files.py`, `test_accepted_findings_scope.py`, `test_advisory.py`, `test_apply_fix_known_checks.py`, `test_ask.py`, `test_batch_check_on_record.py`, `test_batch_own_check.py`, `test_check_findings_hold_dependents.py`, `test_check_gate_fixture.py`, `test_check_repeats_finding.py`, `test_check_task_findings.py`, +52 more |
| `utils/run_sweep.py` | 4 | `test_detach.py`, `test_review_round.py`, `test_signal_identity_posix.py`, `test_windows_run_jobs.py` |
| `utils/runner_preflight.py` | 10 | `test_cursor_auth_probe.py`, `test_failure_classifier.py`, `test_generate_tasks_orchestrator.py`, `test_lane_b_structured_output.py`, `test_opencode_config_providers.py`, `test_opencode_permission_block.py`, `test_probe_output_redaction.py`, `test_subagent_executor.py`, `test_subagent_executor_moved_names.py`, `test_version_gate.py` |
| `utils/runner_selection.py` | 8 | `test_checked_bytes_consumed.py`, `test_generate_tasks_orchestrator.py`, `test_plugin_landlock_executor.py`, `test_plugin_landlock_write.py`, `test_subagent_executor.py`, `test_subagent_executor_moved_names.py`, `test_validation_batch_binding.py`, `test_validation_context_by_path.py` |
| `utils/sandbox_walk.py` | 14 | `test_code_review_orchestrator.py`, `test_doctor.py`, `test_lane_b_structured_output.py`, `test_opencode_permission_block.py`, `test_plugin_landlock_acceptance.py`, `test_plugin_landlock_executor.py`, `test_plugin_landlock_write.py`, `test_probe_guard_production_wrapper.py`, `test_process_state.py`, `test_sandbox_fallback_ladder.py`, `test_sandbox_rung_retirement.py`, `test_subagent_executor.py`, +2 more |
| `utils/schema_validator.py` | 19 | `test_schema_validator.py`, `test_batch_check_on_record.py`, `test_check_reply.py`, `test_check_task_findings.py`, `test_code_review_integration.py`, `test_commit_review_prompt.py`, `test_human_decisions.py`, `test_json_recovery_sites.py`, `test_lane_b_structured_output.py`, `test_processed_items_both_legs.py`, `test_reply_without_schema_version.py`, `test_result_contract.py`, +7 more |
| `utils/scope_apply.py` | 4 | `test_apply_report.py`, `test_clock.py`, `test_scope_policy_apply.py`, `test_state_save_merge.py` |
| `utils/scope_policy.py` | 15 | `test_scope_policy.py`, `test_scope_policy_apply.py`, `test_scope_policy_cli.py`, `test_scope_policy_composites.py`, `test_commit_review_prompt.py`, `test_prompt_loader.py`, `test_review_plan_orchestrator.py`, `test_review_prompt_rules.py`, `test_subagent_prompts.py`, `test_apply_report.py`, `test_clock.py`, `test_same_content_group_identity.py`, +3 more |
| `utils/scope_policy_runs.py` | 5 | `test_clock.py`, `test_scope_policy_cli.py`, `test_scope_policy_composites.py`, `test_scope_policy_leaf_modes.py`, `test_state_save_merge.py` |
| `utils/scope_report.py` | 1 | `test_scope_report_tags.py` |
| `utils/scope_round.py` | 3 | `test_batched_validation.py`, `test_same_content_group_identity.py`, `test_scope_prompts_and_validation.py` |
| `utils/selection_migration.py` | 0 | none (no direct importer) |
| `utils/slot_guard.py` | 4 | `test_slot_guard_parity.py`, `test_ask_answer_persistence.py`, `test_ask_withheld_answers.py`, `test_rejected_slot_answers.py` |
| `utils/state_file.py` | 6 | `test_state_file.py`, `test_batch_done.py`, `test_held_folder.py`, `test_judge_vocab.py`, `test_plan_dir_writers_held.py`, `test_state_save_merge.py` |
| `utils/state_manager.py` | 184 | **hub**, cannot narrow: `test_state_manager_decisions.py`, `test_accepted_check_rerun.py`, `test_accepted_finding_note.py`, `test_accepted_finding_restate.py`, `test_accepted_findings_scope.py`, `test_an_answer_answers_one_stop.py`, `test_applied_without_changes.py`, `test_apply_code_fixes_unit.py`, `test_apply_json_output.py`, `test_apply_orchestrator_base.py`, `test_apply_phase_in_progress.py`, `test_apply_phases_close_like_implement.py`, +172 more |
| `utils/status_commands.py` | 2 | `test_run_status_moved_names.py`, `test_status_result_line_on_a_fault.py` |
| `utils/status_file.py` | 88 | **hub**, cannot narrow: `test_status_file.py`, `test_batch_ledger.py`, `test_consolidation_integration.py`, `test_html_report_view.py`, `test_misnamed_result_files.py`, `test_reaggregate_identity.py`, `test_review_priority_reset.py`, `test_review_round.py`, `test_served_model_reporting.py`, `test_usage_reporting.py`, `test_abort_signal_handlers.py`, `test_advisory.py`, +76 more |
| `utils/status_native.py` | 3 | `test_write_legs_parity.py`, `test_run_status_moved_names.py`, `test_status_result_line_on_a_fault.py` |
| `utils/status_text.py` | 4 | `test_advisory.py`, `test_detach.py`, `test_run_status_moved_names.py`, `test_status_result_line_on_a_fault.py` |
| `utils/status_views.py` | 4 | `test_commit_dispatch.py`, `test_next_command_emitters.py`, `test_run_status_moved_names.py`, `test_status_result_line_on_a_fault.py` |
| `utils/stop_answers.py` | 2 | `test_scheduler_null_state.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/stream_bootstrap.py` | 2 | `test_stream_bootstrap.py`, `test_eta_closes_every_ending.py` |
| `utils/structured_output.py` | 3 | `test_structured_output.py`, `test_lane_b_structured_output.py`, `test_strict_schema_subset.py` |
| `utils/subagent_executor.py` | 184 | **hub**, cannot narrow: `test_subagent_executor.py`, `test_subagent_executor_moved_names.py`, `test_accepted_finding_note.py`, `test_accepted_finding_restate.py`, `test_apply_fix_known_checks.py`, `test_batch_ledger.py`, `test_bypassed_slot_failure_provenance.py`, `test_check_task_findings.py`, `test_clipped_previews.py`, `test_declined_reason_punctuation.py`, `test_e2e_exec_subagents.py`, `test_envelope_locator.py`, +172 more |
| `utils/subagent_prompts.py` | 15 | `test_subagent_prompts.py`, `test_accepted_finding_restate.py`, `test_check_always_reported.py`, `test_code_review_orchestrator.py`, `test_dispatch_steps.py`, `test_generate_tasks_orchestrator.py`, `test_lane_b_structured_output.py`, `test_report_repair_preserves_checks.py`, `test_review_code_head_ref.py`, `test_scope_prompts_and_validation.py`, `test_validation_reply_printed_once.py`, `test_write_executor.py`, +3 more |
| `utils/subagent_tasks.py` | 49 | **hub**, cannot narrow: `test_subagent_tasks.py`, `test_applied_without_changes.py`, `test_apply_fix_known_checks.py`, `test_attempt_backpressure.py`, `test_batch_own_check.py`, `test_batch_snapshot.py`, `test_checked_bytes_consumed.py`, `test_clipped_previews.py`, `test_code_fix_prior_batches.py`, `test_code_review_orchestrator.py`, `test_commit_write_policy.py`, `test_dispatch_steps.py`, +37 more |
| `utils/suggestion_batcher.py` | 7 | `test_suggestion_batcher.py`, `test_apply_task_suggestions.py`, `test_e2e_batching.py`, `test_subagent_tasks.py`, `test_tasks_graph_check.py`, `test_write_executor_orchestrators.py`, `test_text_regex_linearity.py` |
| `utils/suggestion_processor.py` | 11 | `test_suggestion_processor.py`, `test_apply_code_fixes_unit.py`, `test_code_review_integration.py`, `test_commit_attribution.py`, `test_consolidation_group_shape.py`, `test_importance_lanes.py`, `test_json_recovery_sites.py`, `test_orchestrator_integration.py`, `test_reaggregate_identity.py`, `test_scope_prompts_and_validation.py`, `test_text_regex_linearity.py` |
| `utils/task_batch.py` | 7 | `test_attempt_backpressure.py`, `test_checked_bytes_consumed.py`, `test_jail_and_audit_helpers.py`, `test_review_code_head_ref.py`, `test_subagent_executor.py`, `test_subagent_executor_moved_names.py`, `test_validation_batch_binding.py` |
| `utils/task_decomposer.py` | 10 | `test_task_decomposer.py`, `test_executor_role_sweep.py`, `test_human_task_answers.py`, `test_implement_orchestrator.py`, `test_lane_b_structured_output.py`, `test_tasks_file.py`, `test_tasks_graph_check.py`, `test_update_plan_tasks.py`, `test_generate_tasks_orchestrator.py`, `test_text_regex_linearity.py` |
| `utils/task_dispatch.py` | 9 | `test_attempt_backpressure.py`, `test_checked_bytes_consumed.py`, `test_dispatch_steps.py`, `test_generate_tasks_orchestrator.py`, `test_lane_b_structured_output.py`, `test_subagent_executor.py`, `test_subagent_executor_moved_names.py`, `test_validation_batch_binding.py`, `test_validation_context_by_path.py` |
| `utils/task_file_runner.py` | 6 | `test_checked_bytes_consumed.py`, `test_damaged_checkpoint_recovery.py`, `test_e2e_failures.py`, `test_rejected_slot_answers.py`, `test_subagent_executor.py`, `test_subagent_executor_moved_names.py` |
| `utils/task_ledger.py` | 8 | `test_task_ledger.py`, `test_batch_own_check.py`, `test_check_findings_hold_dependents.py`, `test_implement_stop_account.py`, `test_refused_answers.py`, `test_status_after_a_stop.py`, `test_write_executor.py`, `test_write_legs_parity.py` |
| `utils/task_outcome.py` | 2 | `test_dispatch_steps.py`, `test_subagent_executor_moved_names.py` |
| `utils/task_result.py` | 7 | `test_attempt_backpressure.py`, `test_checked_bytes_consumed.py`, `test_dispatch_steps.py`, `test_generate_tasks_orchestrator.py`, `test_plan_dir_writers_held.py`, `test_subagent_executor.py`, `test_subagent_executor_moved_names.py` |
| `utils/tasks_file.py` | 5 | `test_tasks_file.py`, `test_apply_task_suggestions.py`, `test_generate_tasks_orchestrator.py`, `test_judge_vocab.py`, `test_update_plan_tasks.py` |
| `utils/tasks_graph_check.py` | 1 | `test_tasks_graph_check.py` |
| `utils/tasks_marker.py` | 0 | none (no direct importer) |
| `utils/text_clip.py` | 9 | `test_text_clip.py`, `test_clipped_previews.py`, `test_commit_range.py`, `test_declined_reason_punctuation.py`, `test_full_workflow_orchestrator.py`, `test_native_apply_view_names_declined_items.py`, `test_notice_clause_join.py`, `test_opencode_stop_title.py`, `test_relay_says_it_once.py` |
| `utils/text_fmt.py` | 1 | `test_text_fmt.py` |
| `utils/tracking.py` | 2 | `test_finalize_tracking.py`, `test_write_executor.py` |
| `utils/unattended.py` | 18 | `test_unattended.py`, `test_provider_registry.py`, `test_claude_package.py`, `test_cli_surface_emitter.py`, `test_codex_package.py`, `test_detach.py`, `test_full_workflow_orchestrator.py`, `test_host_capabilities.py`, `test_implement_resolution_replay.py`, `test_instruction_portability.py`, `test_opencode_package.py`, `test_partial_detached_result.py`, +6 more |
| `utils/validation.py` | 33 | `test_validation_batch_binding.py`, `test_validation_context_by_path.py`, `test_validation_group_id.py`, `test_validation_recovery.py`, `test_validation_sections.py`, `test_validation_subagent.py`, `test_validation_subagent_orchestrators.py`, `test_batched_validation.py`, `test_e2e_validation_edge_cases.py`, `test_reopen_for_validation.py`, `test_scope_prompts_and_validation.py`, `test_apply_code_fixes_unit.py`, +21 more |
| `utils/validation_batcher.py` | 1 | `test_validation_batcher.py` |
| `utils/validation_context.py` | 4 | `test_validation_context_by_path.py`, `test_validation_context_surfacing.py`, `test_result_contract.py`, `test_validation_sections.py` |
| `utils/validation_sections.py` | 1 | `test_validation_sections.py` |
| `utils/verification_lines.py` | 1 | `test_write_batch_scheduler_moved_names.py` |
| `utils/wakeup_check.py` | 3 | `test_wakeup_check.py`, `test_hook_transcript_settle.py`, `test_report_check_composite.py` |
| `utils/write_batch_scheduler.py` | 56 | **hub**, cannot narrow: `test_write_batch_scheduler_moved_names.py`, `test_accepted_check_rerun.py`, `test_accepted_finding_note.py`, `test_accepted_finding_restate.py`, `test_accepted_findings_scope.py`, `test_apply_fix_known_checks.py`, `test_apply_orchestrator_base.py`, `test_batch_own_check.py`, `test_check_always_reported.py`, `test_check_repeats_finding.py`, `test_code_fix_prior_batches.py`, `test_commit_write_policy.py`, +44 more |
| `utils/write_check_holds.py` | 3 | `test_resume_why.py`, `test_scheduler_record_writers.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/write_commit_leg.py` | 3 | `test_commit_apply.py`, `test_write_batch_scheduler_moved_names.py`, `test_write_verdict_table.py` |
| `utils/write_context.py` | 3 | `test_scheduler_null_state.py`, `test_scheduler_options.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/write_dry_run.py` | 1 | `test_write_batch_scheduler_moved_names.py` |
| `utils/write_eta.py` | 1 | `test_write_batch_scheduler_moved_names.py` |
| `utils/write_grant.py` | 8 | `test_dispatch_counts.py`, `test_dispatch_executor_hooks.py`, `test_external_rollback_storage.py`, `test_plugin_landlock_write.py`, `test_write_batch_scheduler_moved_names.py`, `test_write_ceiling_cache.py`, `test_write_executor.py`, `test_write_verdict_table.py` |
| `utils/write_outcomes.py` | 2 | `test_write_outcomes.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/write_phase_bridge.py` | 46 | **hub**, cannot narrow: `test_apply_code_fixes_unit.py`, `test_apply_fix_known_checks.py`, `test_apply_suggestions_integration.py`, `test_batch_done.py`, `test_batch_snapshot.py`, `test_clipped_previews.py`, `test_code_fix_prior_batches.py`, `test_commit_write_policy.py`, `test_implement_continues_a_started_run.py`, `test_implement_orchestrator.py`, `test_native_apply_count_over_the_run.py`, `test_native_apply_view_names_declined_items.py`, +34 more |
| `utils/write_prompt.py` | 11 | `test_write_prompt.py`, `test_selected_tasks_and_check_trees.py`, `test_apply_fix_known_checks.py`, `test_apply_suggestions_integration.py`, `test_batch_snapshot.py`, `test_code_fix_prior_batches.py`, `test_commit_write_policy.py`, `test_prior_batches_named_by_id.py`, `test_prompt_files_stay_inside.py`, `test_scope_policy_apply.py`, `test_write_executor_orchestrators.py` |
| `utils/write_record.py` | 1 | `test_write_record.py` |
| `utils/write_reports.py` | 2 | `test_plan_dir_writers_held.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/write_resume.py` | 3 | `test_scheduler_record_writers.py`, `test_completion_predates.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/write_safety.py` | 27 | `test_batch_ledger.py`, `test_check_gate_fixture.py`, `test_commit_apply.py`, `test_commit_write_policy.py`, `test_implement_continues_a_started_run.py`, `test_next_step_relay_rules.py`, `test_plugin_landlock_acceptance.py`, `test_policy_way_forward.py`, `test_resume_why.py`, `test_review_cycle_rule.py`, `test_selected_tasks_and_check_trees.py`, `test_write_executor.py`, +15 more |
| `utils/write_snapshots.py` | 2 | `test_write_batch_scheduler_moved_names.py`, `test_write_verdict_table.py` |
| `utils/write_stop.py` | 1 | `test_write_batch_scheduler_moved_names.py` |
| `utils/write_sweep.py` | 6 | `test_applied_without_changes.py`, `test_commit_write_policy.py`, `test_stop_resume_wording.py`, `test_write_batch_scheduler_moved_names.py`, `test_write_executor.py`, `test_write_verdict_table.py` |
| `utils/write_verdict.py` | 1 | `test_write_verdict_table.py` |
| `utils/write_verify.py` | 3 | `test_write_verify_kill_verdict.py`, `test_kill_tree_posix.py`, `test_write_batch_scheduler_moved_names.py` |
| `utils/yaml_io.py` | 3 | `test_yaml_io.py`, `test_process_state.py`, `test_read_only_modes.py` |

## Declared routes

Tests that load modules by a computed name (`SWEEPS`):

| Test file | Covers | Why |
|-----------|--------|-----|
| `test_retry_ladder.py` | `utils.providers.*` | parametrized over adapter modules, loaded by dotted name from the parameters |
| `test_reasoning_effort_ladder.py` | `utils.providers.*` | loads the adapter module behind every registry id through the roster |
| `test_pi_family_stream.py` | `utils.providers.base`, `utils.providers.pi_family_stream`, `utils.providers.pi`, `utils.providers.omp` | imports the pi-family modules in several orders in a fresh interpreter, by a name built from the order argument |

Subprocess suites (`HARNESS_SUITES`). The orchestrators a suite drives are
derived from the mode words and orchestrator names it passes; the modules
column is what the row adds:

| Test file | Adds | Why |
|-----------|------|-----|
| `test_e2e_apply_code_fixes.py` | - | apply_code_fixes_orchestrator subprocess |
| `test_e2e_batching.py` | - | imports the batchers directly; the scan sees them |
| `test_e2e_code_review.py` | - | code_review_orchestrator subprocess and multi_llm.py |
| `test_e2e_commit_modes.py` | `utils.commit_review`, `utils.commit_apply`, `utils.commit_brief`, `utils.commit_range` | multi_llm.py subprocesses for the three commit modes: the front door, the brief, the range and the apply path run for real |
| `test_e2e_exec_subagents.py` | `utils.subagent_executor` | --exec-subagents drives the executor in the subprocess |
| `test_e2e_failures.py` | - | review_plan and code_review subprocesses on failing providers |
| `test_e2e_full_chain.py` | - | the phase chain through the orchestrator subprocesses |
| `test_e2e_html_report.py` | `utils.html_report_generator` | the HTML report written and reimported by a phase run |
| `test_e2e_improve_modes.py` | `utils.improve_workflow`, `utils.improve_steps`, `utils.commit_review` | improve-plan, improve-code and improve-commits through multi_llm.py |
| `test_e2e_mock_infrastructure.py` | - | tests the mock provider binaries, not a source module |
| `test_e2e_runs_retention.py` | - | tests the container harness's run pruning, not a source module |
| `test_e2e_validation.py` | - | implement_orchestrator subprocess on validation edge cases |
| `test_e2e_validation_edge_cases.py` | - | imports utils.validation directly; the scan sees it |
| `test_e2e_workflow.py` | - | review_plan and implement subprocesses end to end |
