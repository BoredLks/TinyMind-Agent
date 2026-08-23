"""WebSocket chat endpoint (M4.1: concurrent generation with cancel).

A turn runs as a background asyncio task so the receive loop stays responsive:
it can handle `cancel` (cancels the task) and `tool_approval_response` (resolves
the pending approval future) WHILE the model is generating.

Client -> server:
    {"type":"user_message","request_id","session_id?","content"}
    {"type":"cancel","request_id"}
    {"type":"tool_approval_response","request_id","call_id","approved":bool}

Server -> client:
    status / token / tool_call / tool_approval_request / tool_result /
    workflow_stage / cancelled / done / error
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agent.context import build_context
from app.agent.loop import run_agent_loop
from app.agent.memory import format_memory_narrative
from app.agent.subagent import SubagentRunner
from app.agent.model.base import ChatMessage
from app.agent.system_prompt import build_system_prompt, suggest_skill
from app.agent.usage import estimate_tokens, estimate_tokens_for_messages
from app.agent.workflow import stage_for_skill
from app.core import config as app_config
from app.core.config import load_settings
from app.storage import dao
from app.tools.base import ToolContext
from app.tools.workspace import ensure_workspace

router = APIRouter()


def _to_openai(messages: List[ChatMessage], keep_multimodal: bool = False) -> List[dict]:
    """Convert internal messages to OpenAI format.

    For historical messages (all but the last user message), multimodal JSON is
    always converted to plain text to avoid breaking non-vision models.
    Only the last user message keeps multimodal format when keep_multimodal=True.
    """
    result = []
    # Find the index of the last user message
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break

    for idx, m in enumerate(messages):
        # Check if content is multimodal JSON
        try:
            parsed = json.loads(m.content)
            if isinstance(parsed, dict) and parsed.get("type") == "multimodal" and "images" in parsed:
                # Only keep multimodal format for the last user message when keep_multimodal is True
                if keep_multimodal and idx == last_user_idx:
                    content_parts = []
                    if parsed.get("text"):
                        content_parts.append({"type": "text", "text": parsed["text"]})
                    for img in parsed.get("images", []):
                        content_parts.append(img)
                    result.append({"role": m.role, "content": content_parts})
                    continue
                else:
                    # Historical image messages: extract text only
                    text_only = parsed.get("text", "")
                    if text_only:
                        result.append({"role": m.role, "content": text_only + "\n[图片已省略]"})
                    continue
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        result.append({"role": m.role, "content": m.content})
    return result


@router.websocket("/ws/chat")
async def chat_ws(ws: WebSocket) -> None:
    await ws.accept()
    app = ws.app
    conn = getattr(app.state, "db", None)
    skills = getattr(app.state, "skills", None)
    ephemeral: List[ChatMessage] = []
    usage_state = {"ephemeral": 0}
    # Concurrency model: one running turn per request_id; different sessions run
    # in parallel, a new message in the SAME session supersedes (cancels) its
    # previous still-running turn. Events are routed by (session_id, request_id).
    active_tasks: Dict[str, asyncio.Task] = {}   # request_id -> running turn task
    session_active: Dict[str, str] = {}          # session_id -> current request_id
    pending_approvals: Dict[str, asyncio.Future] = {}
    pending_interactions: Dict[str, asyncio.Future] = {}

    async def emit(event: dict, request_id: str, session_id: Optional[str]) -> None:
        # Advance workflow stage when a stage-bearing skill loads.
        if (
            event.get("type") == "tool_call"
            and event.get("name") == "load_skill"
            and session_id
            and conn is not None
        ):
            try:
                skill_name = json.loads(event.get("args") or "{}").get("name")
            except json.JSONDecodeError:
                skill_name = None
            new_stage = stage_for_skill(skill_name)
            if new_stage:
                current = dao.get_session(conn, session_id)
                if current and current.get("workflow_stage") != new_stage:
                    dao.set_workflow_stage(conn, session_id, new_stage)
                    await ws.send_json({"type": "workflow_stage", "request_id": request_id,
                                        "session_id": session_id, "stage": new_stage})
        # Don't overwrite session_id if the event already has one (e.g. subagent events)
        await ws.send_json({**event, "request_id": request_id, "session_id": event.get("session_id", session_id)})

    def make_request_approval(request_id: str, turn_session_id: Optional[str], ctx):
        async def request_approval(call) -> bool:
            fut: asyncio.Future = asyncio.get_event_loop().create_future()
            pending_approvals[call.id] = fut
            subagent_depth = int(getattr(ctx, "subagent_depth", 0) or 0)
            # session_id MUST be the turn's top-level session (where the tool card
            # is shown), not ctx.session_id (which is mutated during subagent runs);
            # without it the frontend's (session,request) routing drops the prompt
            # and the tool hangs waiting for an approval the user can never give.
            await ws.send_json({"type": "tool_approval_request", "request_id": request_id,
                                "session_id": turn_session_id,
                                "call_id": call.id, "name": call.name, "args": call.arguments,
                                "subagent": subagent_depth > 0})
            try:
                return await fut
            finally:
                pending_approvals.pop(call.id, None)
        return request_approval

    def make_interact(request_id: str, turn_session_id: Optional[str], ctx):
        async def interact(payload: dict):
            interaction_id = uuid.uuid4().hex
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            pending_interactions[interaction_id] = fut
            await ws.send_json({
                "type": "interaction_request",
                "request_id": request_id,
                "session_id": turn_session_id,
                "interaction_id": interaction_id,
                "call_id": payload.get("call_id") or getattr(ctx, "current_call_id", None),
                "subagent": bool(getattr(ctx, "subagent_depth", 0) or 0),
                **payload,
            })
            try:
                return await fut
            finally:
                pending_interactions.pop(interaction_id, None)
        return interact

    def cancel_request(request_id: Optional[str]) -> None:
        task = active_tasks.get(request_id) if request_id else None
        if task is not None and not task.done():
            task.cancel()

    def cancel_all() -> None:
        for task in list(active_tasks.values()):
            if not task.done():
                task.cancel()

    def make_adapter(provider_id: Optional[str] = None):
        try:
            return app.state.adapter_factory(provider_id)
        except TypeError:
            return app.state.adapter_factory()

    async def run_turn(
        request_id: str,
        session_id: Optional[str],
        content: str,
        provider_id: Optional[str] = None,
    ) -> None:
        settings = load_settings(conn)
        # A session bound to a project writes into that project's directory;
        # otherwise it falls back to the default sandbox workspace.
        root = settings.workspace_root
        if session_id and conn is not None:
            project_path = dao.project_path_for_session(conn, session_id)
            if project_path:
                root = project_path
        workspace_root = ensure_workspace(root)

        if session_id and conn is not None:
            session = dao.get_session(conn, session_id)
            if session is None:
                await ws.send_json({"type": "error", "request_id": request_id,
                                    "session_id": session_id,
                                    "message": "session not found", "retryable": False})
                return
            dao.add_message(conn, session_id, "user", content)
            rows = dao.list_messages(conn, session_id)
            # Filter out tool_event messages — those are for UI reconstruction only
            history = [ChatMessage(r["role"], r["content"]) for r in rows if r["role"] != "tool_event"]
            user_prompt = session.get("system_prompt") or settings.system_prompt
        else:
            ephemeral.append(ChatMessage("user", content))
            history = ephemeral
            user_prompt = settings.system_prompt

        enabled_skills = (
            [m for m in skills.list_metas() if conn is None or dao.is_skill_enabled(conn, m.name)]
            if skills is not None else []
        )
        hint = suggest_skill(content, [m.name for m in enabled_skills]) if enabled_skills else None
        memory_text = format_memory_narrative(conn) if settings.memory_enabled and conn is not None else None
        system_prompt = build_system_prompt(
            enabled_skills, user_prompt, hint, workspace_root=workspace_root, memory=memory_text
        )
        context = build_context(history, system_prompt=system_prompt,
                                max_messages=settings.context_max_messages,
                                strategy=settings.context_strategy,
                                max_tokens=settings.context_max_tokens)
        # Check if current message has images BEFORE building base_messages
        has_images = False
        try:
            parsed = json.loads(content)
            has_images = isinstance(parsed, dict) and parsed.get("type") == "multimodal" and bool(parsed.get("images"))
        except (json.JSONDecodeError, TypeError):
            pass

        base_messages = _to_openai(context, keep_multimodal=has_images)
        ctx = ToolContext(
            workspace_root=workspace_root,
            skills=skills,
            registry=app.state.registry,
            session_id=session_id,
            db=conn,
        )
        disabled = (
            list(settings.disabled_tools) if settings.tools_enabled else app.state.registry.names()
        )

        # Track tool events for persistence
        tool_events: list[dict] = []

        async def emit_rid(event: dict) -> None:
            # Persist tool events so they survive session switching
            if session_id and conn is not None and event.get("type") in (
                "tool_call", "tool_result",
            ):
                dao.add_message(
                    conn, session_id, "tool_event",
                    json.dumps(event, ensure_ascii=False),
                )
            await emit(event, request_id, session_id)

        approval = make_request_approval(request_id, session_id, ctx)
        ctx.interact = make_interact(request_id, session_id, ctx)
        ctx.subagent_runner = SubagentRunner(
            app.state.adapter_factory, app.state.registry, ctx, emit_rid, approval,
            base_disabled=disabled,
            conn=conn,
        )

        await ws.send_json({"type": "status", "request_id": request_id, "session_id": session_id, "state": "generating"})
        if len(history) > settings.context_max_messages:
            await ws.send_json({"type": "context_truncated", "request_id": request_id,
                                "session_id": session_id,
                                "kept": settings.context_max_messages, "total": len(history)})
        providers = [p for p in app_config.load_provider_records(conn) if p.enabled]
        requested_provider = provider_id or settings.active_provider_id
        ordered_provider_ids = []
        for pid in [requested_provider, *[p.id for p in providers]]:
            if pid and pid not in ordered_provider_ids:
                ordered_provider_ids.append(pid)
        if not ordered_provider_ids:
            ordered_provider_ids = [None]
        final_messages = base_messages
        final_text = ""
        last_exc: Exception | None = None

        for index, pid in enumerate(ordered_provider_ids):
            adapter = make_adapter(pid)
            if hasattr(adapter, "set_retry_callback"):
                async def _on_retry(attempt: int) -> None:
                    await ws.send_json({"type": "retrying", "request_id": request_id,
                                        "session_id": session_id, "attempt": attempt})

                adapter.set_retry_callback(_on_retry)
            attempt_messages = [dict(m) for m in base_messages]
            try:
                if index > 0:
                    msg = f"当前模型不支持图片，切换到备用模型 {pid}" if has_images else f"主模型连接失败，切换到备用模型 {pid}"
                    await ws.send_json({
                        "type": "provider_fallback",
                        "request_id": request_id,
                        "session_id": session_id,
                        "provider_id": pid,
                        "reason": "image_not_supported" if has_images else "error",
                    })
                    await ws.send_json({"type": "token", "request_id": request_id,
                                        "session_id": session_id, "delta": f"\n\n> ⚠️ {msg}\n\n"})
                final_text = await asyncio.wait_for(
                    run_agent_loop(
                        adapter, app.state.registry, ctx, attempt_messages,
                        emit=emit_rid, request_approval=approval,
                        disabled_tools=disabled,
                        max_iterations=settings.max_tool_iterations,
                        max_tool_arg_len=settings.max_tool_arg_len,
                        max_tool_result_len=settings.max_tool_result_len,
                        max_tool_text_len=settings.max_tool_text_len,
                    ),
                    timeout=settings.turn_timeout,
                )
                final_messages = attempt_messages
                last_exc = None
                break
            except asyncio.TimeoutError:
                last_exc = Exception(f"模型 {pid} 响应超时（{settings.turn_timeout}秒）")
                # Only fallback on timeout for image messages
                if has_images and index < len(ordered_provider_ids) - 1:
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # ONLY fallback for image messages when error is vision-related
                if has_images:
                    err_str = str(exc).lower()
                    is_vision_error = any(k in err_str for k in [
                        "image", "vision", "multimodal", "unsupported_content",
                        "not support image", "invalid_image",
                    ])
                    if is_vision_error and index < len(ordered_provider_ids) - 1:
                        continue
                # For all other cases (text messages, non-vision errors) — do NOT fallback
                break
        if last_exc is not None:
            raise last_exc
        if session_id and conn is not None:
            dao.add_message(conn, session_id, "assistant", final_text)
            if settings.memory_enabled:
                worker = getattr(app.state, "memory_worker", None)
                if worker is not None:
                    worker.enqueue(user=content, assistant=final_text)
        else:
            ephemeral.append(ChatMessage("assistant", final_text))

        prompt_tokens = estimate_tokens_for_messages(final_messages)
        completion_tokens = estimate_tokens(final_text)
        turn_total = prompt_tokens + completion_tokens
        if session_id and conn is not None:
            session_total = dao.add_session_tokens(conn, session_id, turn_total)
        else:
            usage_state["ephemeral"] += turn_total
            session_total = usage_state["ephemeral"]

        await ws.send_json({
            "type": "done", "request_id": request_id, "session_id": session_id,
            "usage": {"prompt": prompt_tokens, "completion": completion_tokens,
                      "turn_total": turn_total, "session_total": session_total, "estimated": True},
        })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "invalid JSON", "retryable": False})
                continue
            mtype = msg.get("type")

            if mtype == "user_message":
                request_id = msg.get("request_id") or uuid.uuid4().hex
                _session_id = msg.get("session_id")
                _content = msg.get("content", "")
                _provider_id = msg.get("provider_id")
                # One running turn per session: a new message in the same session
                # supersedes its previous, still-running turn. Different sessions
                # run concurrently and never share turn state.
                if _session_id:
                    prev_rid = session_active.get(_session_id)
                    if prev_rid:
                        cancel_request(prev_rid)

                # Bind the per-turn identifiers as defaults so each task captures
                # its own values (never the latest loop variables).
                async def _safe_run_turn(rid=request_id, sid=_session_id,
                                         content=_content, provider_id=_provider_id):
                    try:
                        await run_turn(rid, sid, content, provider_id)
                    except asyncio.CancelledError:
                        try:
                            await ws.send_json({"type": "cancelled", "request_id": rid, "session_id": sid})
                        except Exception:
                            pass
                        return
                    except Exception as exc:  # noqa: BLE001 — catch any unhandled error
                        try:
                            await ws.send_json({
                                "type": "error",
                                "request_id": rid,
                                "session_id": sid,
                                "message": f"生成失败: {exc}",
                                "retryable": True,
                            })
                        except Exception:
                            pass
                    finally:
                        active_tasks.pop(rid, None)
                        if sid and session_active.get(sid) == rid:
                            session_active.pop(sid, None)

                task = asyncio.create_task(_safe_run_turn())
                active_tasks[request_id] = task
                if _session_id:
                    session_active[_session_id] = request_id
            elif mtype == "cancel":
                cancel_request(msg.get("request_id"))
            elif mtype == "tool_approval_response":
                fut = pending_approvals.get(msg.get("call_id"))
                if fut is not None and not fut.done():
                    fut.set_result(bool(msg.get("approved")))
            elif mtype == "interaction_response":
                fut = pending_interactions.get(msg.get("interaction_id"))
                if fut is not None and not fut.done():
                    fut.set_result(msg.get("response"))
    except WebSocketDisconnect:
        cancel_all()
        return
