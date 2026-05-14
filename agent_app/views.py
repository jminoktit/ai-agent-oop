import json

from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt

from .agents import AgentOrchestrator
from .models import Conversation, Message
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
    return render(request, "agent_app/index.html", {
        "agents": orchestrator.list_agents(),
        "active_agent": orchestrator.active_agent,
        "conversations": conversations,
        "conv_list_json": json.dumps(conv_list),
        "user_name": request.user.username,
    })


@login_required
@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    data = json.loads(request.body)
    user_input = data.get("message", "").strip()
    conversation_id = data.get("conversation_id")

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
        conv = Conversation.objects.filter(id=conversation_id).first()
        if conv:
            agent_key = AGENT_NAME_TO_KEY.get(conv.agent_name, conv.agent_name)

    if agent_key:
        if agent_key in orchestrator.list_agents():
            orchestrator.active_agent = agent_key
        result = orchestrator.run(agent_key, user_input, conversation_id)
    else:
        result = orchestrator.auto_route(user_input, conversation_id)
    agent = orchestrator.get_agent(orchestrator.active_agent or "chat")

    if agent and agent.conversation_id:
        conv = Conversation.objects.filter(id=agent.conversation_id).first()
        if conv and not conv.user_id:
            conv.user = request.user
            conv.save()

    messages = []
    if agent and agent.conversation_id:
        qs = Message.objects.filter(conversation_id=agent.conversation_id).order_by("created_at")
        messages = [{"role": m.role, "content": m.content} for m in qs]

    return JsonResponse({
        "response": result,
        "active_agent": orchestrator.active_agent,
        "conversation_id": agent.conversation_id if agent else None,
        "messages": messages,
    })


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
    conv = get_object_or_404(Conversation, id=conv_id)
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
