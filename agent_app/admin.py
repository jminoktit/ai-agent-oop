from django.contrib import admin

from .models import Conversation, Message


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ["id", "agent_name", "created_at", "updated_at"]
    list_filter = ["agent_name"]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["id", "conversation", "role", "short_content", "created_at"]
    list_filter = ["role", "conversation"]

    def short_content(self, obj):
        return obj.content[:80]
