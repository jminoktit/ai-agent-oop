from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("chat/", views.chat, name="chat"),
    path("switch-agent/", views.switch_agent, name="switch_agent"),
    path("new-conversation/", views.new_conversation, name="new_conversation"),
    path("conversation/<int:conv_id>/", views.conversation_detail, name="conversation_detail"),
    path("conversation/<int:conv_id>/rename/", views.rename_conversation, name="rename_conversation"),
    path("conversation/<int:conv_id>/delete/", views.delete_conversation, name="delete_conversation"),
    path("conversation/<int:conv_id>/clear/", views.clear_conversation, name="clear_conversation"),
    path("conversation/<int:conv_id>/pin/", views.toggle_pin, name="toggle_pin"),
    path("agent-info/", views.agent_info, name="agent_info"),
]
