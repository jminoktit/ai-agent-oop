import json

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt

from .agents import AgentOrchestrator
from .models import Conversation, Message, UploadedFile, TrainingJob
from .tools import RuleEngine

rule_engine = RuleEngine()

orchestrator = AgentOrchestrator()
orchestrator.create_default_agents()

AGENT_NAME_TO_KEY = {
    "ChatBot": "chat",
    "CodeBot": "code",
    "DataBot": "data",
    "ResearchBot": "research",
    "PlannerBot": "planner",
    "MediaBot": "media",
}

AGENT_KEY_TO_NAME = {v: k for k, v in AGENT_NAME_TO_KEY.items()}


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("index")
    else:
        form = AuthenticationForm()
    return render(request, "agent_app/login.html", {"form": form})


def register_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("index")
    else:
        form = UserCreationForm()
    return render(request, "agent_app/register.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required
def index(request):
    conversations = Conversation.objects.filter(user=request.user)[:20]
    conv_list = [{
        "id": c.id,
        "display_name": c.display_name(),
        "agent_name": c.agent_name,
        "is_pinned": c.is_pinned,
        "created_at": c.created_at.isoformat(),
        "msg_count": c.messages.count(),
    } for c in conversations]
    if request.headers.get("Accept") == "application/json":
        return JsonResponse({"conversations": conv_list})
    return render(request, "agent_app/index.html", {
        "agents": orchestrator.list_agents(),
        "active_agent": orchestrator.active_agent,
        "conversations": conversations,
        "conv_list_json": json.dumps(conv_list),
        "user_name": request.user.username,
    })


@login_required
def conversations_list(request):
    """Return conversations as JSON."""
    conversations = Conversation.objects.filter(user=request.user)[:20]
    return JsonResponse({
        "conversations": [{
            "id": c.id,
            "display_name": c.display_name(),
            "agent_name": c.agent_name,
            "is_pinned": c.is_pinned,
            "created_at": c.created_at.isoformat(),
            "msg_count": c.messages.count(),
        } for c in conversations],
    })


@login_required
def user_info(request):
    """Return current user info as JSON."""
    return JsonResponse({
        "username": request.user.username,
        "email": request.user.email,
    })


@login_required
@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    data = json.loads(request.body)
    user_input = data.get("message", "").strip()
    conversation_id = data.get("conversation_id")
    stream = data.get("stream", False)

    if not user_input:
        return JsonResponse({"error": "Message is required"}, status=400)

    rule_result = rule_engine.check(user_input)
    if rule_result is not None:
        agent = orchestrator.get_agent(orchestrator.active_agent or "chat")
        if agent and agent.conversation_id:
            Message.objects.create(
                conversation_id=agent.conversation_id, role="user", content=user_input
            )
            Message.objects.create(
                conversation_id=agent.conversation_id, role="assistant", content=rule_result
            )
        return JsonResponse({
            "response": rule_result,
            "active_agent": orchestrator.active_agent,
            "conversation_id": agent.conversation_id if agent else None,
            "messages": [{"role": "user", "content": user_input}, {"role": "assistant", "content": rule_result}],
        })

    # Handle learn/rules commands via RuleEngine
    if user_input.lower().startswith(("learn:", "learn ", "rules:")):
        engine_result = rule_engine.execute(user_input)
        return JsonResponse({
            "response": engine_result,
            "active_agent": orchestrator.active_agent,
            "conversation_id": None,
            "messages": [{"role": "user", "content": user_input}, {"role": "assistant", "content": engine_result}],
        })

    agent_key = data.get("agent")
    if not agent_key and conversation_id:
        conv = Conversation.objects.filter(id=conversation_id, user=request.user).first()
        if conv:
            agent_key = AGENT_NAME_TO_KEY.get(conv.agent_name, conv.agent_name)

    if agent_key:
        if agent_key in orchestrator.list_agents():
            orchestrator.active_agent = agent_key
        result = orchestrator.run(agent_key, user_input, conversation_id)
    else:
        result = orchestrator.auto_route(user_input, conversation_id)
    agent = orchestrator.get_agent(orchestrator.active_agent or "chat")

    response_type = "text"
    image_url = None
    video_url = None

    try:
        result_data = json.loads(result)
        if result_data.get("type") == "image":
            response_type = "image"
            image_url = result_data.get("url") or result_data.get("local_path")
            result = result_data.get("content", result_data.get("error", ""))
        elif result_data.get("type") == "video":
            response_type = "video"
            video_url = result_data.get("url") or result_data.get("local_path")
            result = result_data.get("content", result_data.get("error", ""))
    except (json.JSONDecodeError, TypeError):
        pass

    if agent and agent.conversation_id:
        conv = Conversation.objects.filter(id=agent.conversation_id, user=request.user).first()
        if conv and not conv.user_id:
            conv.user = request.user
            conv.save()

    messages = []
    if agent and agent.conversation_id:
        qs = Message.objects.filter(conversation_id=agent.conversation_id).order_by("created_at")
        messages = [{"role": m.role, "content": m.content} for m in qs]

    # Streaming response (SSE)
    if stream:
        def event_stream():
            # Send response in chunks for streaming effect
            chunk_size = 20
            full_text = result if isinstance(result, str) else str(result)
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i + chunk_size]
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            # Send completion event
            yield f"data: {json.dumps({'done': True, 'conversation_id': agent.conversation_id if agent else None})}\n\n"
            yield "data: [DONE]\n\n"

        response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
        return response

    response_data = {
        "response": result,
        "active_agent": orchestrator.active_agent,
        "conversation_id": agent.conversation_id if agent else None,
        "messages": messages,
    }

    if response_type == "image" and image_url:
        response_data["image_url"] = image_url

    if response_type == "video" and video_url:
        response_data["video_url"] = video_url

    return JsonResponse(response_data)


@login_required
@csrf_exempt
def switch_agent(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    data = json.loads(request.body)
    name = data.get("agent", "")
    if name not in orchestrator.list_agents():
        return JsonResponse({"error": f"Agent '{name}' not found"}, status=400)

    orchestrator.active_agent = name
    return JsonResponse({"active_agent": name, "agents": orchestrator.list_agents()})


@login_required
@csrf_exempt
def new_conversation(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    data = json.loads(request.body)
    agent_name = data.get("agent", orchestrator.active_agent)
    agent = orchestrator.get_agent(agent_name)
    if agent:
        agent.conversation_id = None
    return JsonResponse({"conversation_id": None, "agent": agent_name})


@login_required
def conversation_detail(request, conv_id):
    conv = get_object_or_404(Conversation, id=conv_id, user=request.user)
    messages = conv.messages.all().order_by("created_at")

    if request.headers.get("Accept") == "application/json":
        display_name = conv.agent_name
        agent_key = AGENT_NAME_TO_KEY.get(display_name, display_name)
        return JsonResponse({
            "conversation_id": conv.id,
            "agent_name": display_name,
            "agent_key": agent_key,
            "messages": [{"role": m.role, "content": m.content, "id": m.id} for m in messages],
        })

    return render(request, "agent_app/conversation.html", {
        "conversation": conv,
        "messages": messages,
    })


@login_required
def agent_info(request):
    return JsonResponse({
        "agents": orchestrator.list_agents(),
        "active_agent": orchestrator.active_agent,
        "details": {n: a.to_dict() for n, a in orchestrator.agents.items()},
    })


@login_required
@csrf_exempt
def rename_conversation(request, conv_id):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    conv = get_object_or_404(Conversation, id=conv_id, user=request.user)
    data = json.loads(request.body)
    title = data.get("title", "").strip()
    conv.title = title if title else None
    conv.save()
    return JsonResponse({"id": conv.id, "display_name": conv.display_name()})


@login_required
@csrf_exempt
def delete_conversation(request, conv_id):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    conv = get_object_or_404(Conversation, id=conv_id, user=request.user)
    conv.delete()
    return JsonResponse({"deleted": True})


@login_required
@csrf_exempt
def clear_conversation(request, conv_id):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    conv = get_object_or_404(Conversation, id=conv_id, user=request.user)
    conv.messages.all().delete()
    return JsonResponse({"cleared": True, "id": conv.id})


@login_required
@csrf_exempt
def toggle_pin(request, conv_id):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    conv = get_object_or_404(Conversation, id=conv_id, user=request.user)
    conv.is_pinned = not conv.is_pinned
    conv.save()
    return JsonResponse({"id": conv.id, "is_pinned": conv.is_pinned})


@login_required
def list_files(request):
    files = UploadedFile.objects.filter(user=request.user)[:50]
    return JsonResponse({
        "files": [{
            "id": f.id,
            "filename": f.filename,
            "file_type": f.file_type,
            "uploaded_at": f.uploaded_at.isoformat(),
        } for f in files]
    })


@login_required
@csrf_exempt
def upload_file(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    uploaded_file = request.FILES.get("file")
    if not uploaded_file:
        return JsonResponse({"error": "No file provided"}, status=400)

    ext = uploaded_file.name.split(".")[-1].lower() if "." in uploaded_file.name else ""
    file_type_map = {
        "txt": "text", "md": "text", "py": "code", "js": "code", "html": "code", "css": "code",
        "json": "json", "csv": "csv", "xml": "xml", "pdf": "pdf", "doc": "doc", "docx": "doc",
    }
    file_type = file_type_map.get(ext, "unknown")

    content_text = ""
    if file_type in ("text", "code", "json", "csv", "xml"):
        try:
            content_text = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception:
            content_text = ""

    obj = UploadedFile.objects.create(
        user=request.user,
        file=uploaded_file,
        filename=uploaded_file.name,
        file_type=file_type,
        content_text=content_text[:50000],
    )
    return JsonResponse({
        "id": obj.id,
        "filename": obj.filename,
        "file_type": obj.file_type,
        "uploaded_at": obj.uploaded_at.isoformat(),
    })


@login_required
def get_file_content(request, file_id):
    f = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    return JsonResponse({
        "id": f.id,
        "filename": f.filename,
        "file_type": f.file_type,
        "content": f.content_text[:10000],
    })


@login_required
@csrf_exempt
def delete_file(request, file_id):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    f = get_object_or_404(UploadedFile, id=file_id, user=request.user)
    f.file.delete()
    f.delete()
    return JsonResponse({"deleted": True})


# ──────────────── TRAINING ────────────────

import threading
import traceback
from django.utils import timezone


def _run_training_job(job_id):
    """Background thread that runs the actual training."""
    from .models import TrainingJob
    import sys, os

    job = TrainingJob.objects.get(id=job_id)
    job.status = "running"
    job.started_at = timezone.now()
    job.save()

    try:
        trainer_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "AuraTrainer")
        if trainer_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(trainer_dir))

        from AuraTrainer.core import AuraTrainer, TrainingConfig

        config = TrainingConfig(
            model_name=job.model_name,
            dataset_size=job.total_samples,
            batch_size=job.batch_size,
            num_epochs=3,
            learning_rate=2e-4,
            max_seq_length=1024,
            output_dir=os.path.join(trainer_dir, "checkpoints", f"job_{job_id}"),
            notify_email=job.email if job.notify_on_complete else "",
        )

        trainer = AuraTrainer(config)
        metrics = trainer.run()

        job.status = "completed"
        job.completed_at = timezone.now()
        job.current_round = job.total_rounds
        if metrics.get("train_loss"):
            job.current_loss = metrics["train_loss"]
        job.save()

        # Send email
        if job.notify_on_complete and job.email:
            _send_email(
                to_email=job.email,
                subject=f"✅ Training Completed - {job.model_name}",
                body=(
                    f"Training completed successfully!\n\n"
                    f"Model: {job.model_name}\n"
                    f"Total rounds: {job.total_rounds}\n"
                    f"Final loss: {job.current_loss or 'N/A'}\n"
                    f"Checkpoint saved to: {config.output_dir}\n"
                ),
            )

    except Exception as e:
        job.status = "failed"
        job.error_message = traceback.format_exc()
        job.save()
        if job.notify_on_complete and job.email:
            _send_email(
                to_email=job.email,
                subject=f"❌ Training Failed - {job.model_name}",
                body=f"Training failed with error:\n\n{traceback.format_exc()}\n",
            )


def _send_email(to_email, subject, body):
    """Send email via SMTP (Gmail App Password)."""
    import smtplib, os
    from email.mime.text import MIMEText

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")

    if not smtp_user or not smtp_pass:
        return

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.starttls()
            s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    except Exception:
        pass


@login_required
def training_dashboard(request):
    """Training dashboard page or JSON list."""
    jobs = TrainingJob.objects.filter(user=request.user)[:20]
    if request.headers.get("Accept") == "application/json":
        return JsonResponse({
            "jobs": [{
                "id": j.id,
                "model_name": j.model_name,
                "dataset_size": j.dataset_size,
                "total_samples": j.total_samples,
                "batch_size": j.batch_size,
                "status": j.status,
                "current_round": j.current_round,
                "total_rounds": j.total_rounds,
                "current_loss": j.current_loss,
                "email": j.email,
                "progress": j.progress_percent(),
                "error_message": j.error_message,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
                "created_at": j.created_at.isoformat(),
            } for j in jobs],
        })
    return render(request, "agent_app/training.html", {
        "jobs": jobs,
        "user_name": request.user.username,
    })


@login_required
@csrf_exempt
def start_training(request):
    """Start a new training job (async in background thread)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    data = json.loads(request.body)
    model_name = data.get("model_name", "google/gemma-2-2b-it")
    dataset_size = data.get("dataset_size", "100k")
    batch_size = data.get("batch_size", 10000)
    email = data.get("email", "")
    notify = data.get("notify_on_complete", True)

    total_map = {"10k": 10000, "50k": 50000, "100k": 100000, "200k": 200000, "500k": 500000}
    total_samples = total_map.get(dataset_size, 100000)

    job = TrainingJob.objects.create(
        user=request.user,
        model_name=model_name,
        dataset_size=dataset_size,
        total_samples=total_samples,
        batch_size=batch_size,
        email=email,
        notify_on_complete=notify,
    )

    t = threading.Thread(target=_run_training_job, args=(job.id,), daemon=True)
    t.start()

    return JsonResponse({
        "id": job.id,
        "status": job.status,
        "message": "Training started!",
    })


@login_required
@csrf_exempt
def stop_training(request, job_id):
    """Stop a running training job."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)
    job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
    job.status = "failed"
    job.error_message = "Stopped by user"
    job.save()
    return JsonResponse({"status": "stopped"})


@login_required
def training_status(request, job_id):
    """Get current status of a training job."""
    job = get_object_or_404(TrainingJob, id=job_id, user=request.user)
    return JsonResponse({
        "id": job.id,
        "status": job.status,
        "current_round": job.current_round,
        "total_rounds": job.total_rounds,
        "progress": job.progress_percent(),
        "current_loss": job.current_loss,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message,
        "model_name": job.model_name,
    })


# ──────────────── SETTINGS ────────────────

from .models import UserSettings


@login_required
def settings_view(request):
    """User settings page."""
    user_settings = UserSettings.get_or_create(request.user)

    if request.method == "POST":
        # Profile
        user_settings.display_name = request.POST.get("display_name", "")
        user_settings.avatar_url = request.POST.get("avatar_url", "")

        # Theme
        user_settings.theme = request.POST.get("theme", "dark")
        user_settings.language = request.POST.get("language", "en")

        # Chat
        user_settings.default_agent = request.POST.get("default_agent", "chat")
        user_settings.show_timestamps = request.POST.get("show_timestamps") == "on"
        user_settings.send_on_enter = request.POST.get("send_on_enter") == "on"
        try:
            user_settings.font_size = int(request.POST.get("font_size", 14))
        except (ValueError, TypeError):
            user_settings.font_size = 14

        # Notifications
        user_settings.email_notifications = request.POST.get("email_notifications") == "on"
        user_settings.training_notifications = request.POST.get("training_notifications") == "on"
        user_settings.sound_enabled = request.POST.get("sound_enabled") == "on"

        # Training
        user_settings.default_model = request.POST.get("default_model", "google/gemma-2-2b-it")
        user_settings.default_dataset_size = request.POST.get("default_dataset_size", "100k")
        try:
            user_settings.default_epochs = int(request.POST.get("default_epochs", 3))
        except (ValueError, TypeError):
            user_settings.default_epochs = 3
        try:
            user_settings.default_learning_rate = float(request.POST.get("default_learning_rate", 2e-4))
        except (ValueError, TypeError):
            user_settings.default_learning_rate = 2e-4

        # API Keys
        user_settings.openai_api_key = request.POST.get("openai_api_key", "")
        user_settings.huggingface_token = request.POST.get("huggingface_token", "")
        user_settings.smtp_user = request.POST.get("smtp_user", "")
        user_settings.smtp_pass = request.POST.get("smtp_pass", "")

        user_settings.save()

        return JsonResponse({"status": "ok", "message": "Settings saved!"})

    return render(request, "agent_app/settings.html", {
        "settings": user_settings,
        "user_name": request.user.username,
    })


@login_required
@csrf_exempt
def update_profile(request):
    """Update user profile (AJAX)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    user_settings = UserSettings.get_or_create(request.user)
    data = json.loads(request.body)

    if "display_name" in data:
        user_settings.display_name = data["display_name"]
    if "avatar_url" in data:
        user_settings.avatar_url = data["avatar_url"]
    if "theme" in data:
        user_settings.theme = data["theme"]
    if "language" in data:
        user_settings.language = data["language"]
    if "email_notifications" in data:
        user_settings.email_notifications = data["email_notifications"]
    if "training_notifications" in data:
        user_settings.training_notifications = data["training_notifications"]
    if "sound_enabled" in data:
        user_settings.sound_enabled = data["sound_enabled"]

    user_settings.save()
    return JsonResponse({"status": "ok"})


@login_required
@csrf_exempt
def update_chat_settings(request):
    """Update chat settings (AJAX)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    user_settings = UserSettings.get_or_create(request.user)
    data = json.loads(request.body)

    if "default_agent" in data:
        user_settings.default_agent = data["default_agent"]
    if "show_timestamps" in data:
        user_settings.show_timestamps = data["show_timestamps"]
    if "send_on_enter" in data:
        user_settings.send_on_enter = data["send_on_enter"]
    if "font_size" in data:
        user_settings.font_size = data["font_size"]

    user_settings.save()
    return JsonResponse({"status": "ok"})


@login_required
@csrf_exempt
def update_training_settings(request):
    """Update training defaults (AJAX)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    user_settings = UserSettings.get_or_create(request.user)
    data = json.loads(request.body)

    if "default_model" in data:
        user_settings.default_model = data["default_model"]
    if "default_dataset_size" in data:
        user_settings.default_dataset_size = data["default_dataset_size"]
    if "default_epochs" in data:
        user_settings.default_epochs = data["default_epochs"]
    if "default_learning_rate" in data:
        user_settings.default_learning_rate = data["default_learning_rate"]

    user_settings.save()
    return JsonResponse({"status": "ok"})


@login_required
@csrf_exempt
def update_api_keys(request):
    """Update API keys (AJAX)."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    user_settings = UserSettings.get_or_create(request.user)
    data = json.loads(request.body)

    if "openai_api_key" in data:
        user_settings.openai_api_key = data["openai_api_key"]
    if "huggingface_token" in data:
        user_settings.huggingface_token = data["huggingface_token"]
    if "smtp_user" in data:
        user_settings.smtp_user = data["smtp_user"]
    if "smtp_pass" in data:
        user_settings.smtp_pass = data["smtp_pass"]

    user_settings.save()
    return JsonResponse({"status": "ok"})


@login_required
def get_settings(request):
    """Get user settings (AJAX)."""
    user_settings = UserSettings.get_or_create(request.user)
    return JsonResponse({
        "display_name": user_settings.display_name,
        "avatar_url": user_settings.avatar_url,
        "theme": user_settings.theme,
        "language": user_settings.language,
        "default_agent": user_settings.default_agent,
        "show_timestamps": user_settings.show_timestamps,
        "send_on_enter": user_settings.send_on_enter,
        "font_size": user_settings.font_size,
        "email_notifications": user_settings.email_notifications,
        "training_notifications": user_settings.training_notifications,
        "sound_enabled": user_settings.sound_enabled,
        "default_model": user_settings.default_model,
        "default_dataset_size": user_settings.default_dataset_size,
        "default_epochs": user_settings.default_epochs,
        "default_learning_rate": user_settings.default_learning_rate,
    })
